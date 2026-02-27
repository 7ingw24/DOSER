import argparse
import numpy as np
import torch
import os
import d4rl
import gym
import random
import wandb
from tqdm import tqdm
import utils
from diffusion.karras import DiffusionModel
from diffusion.mlps import ScoreNetwork

D4RL_SUPPRESS_IMPORT_ERROR = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def load_states_actions(env, no_normalize):
    env = gym.make(env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

    if not no_normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        print("No normalize")

    states = replay_buffer.state
    actions = replay_buffer.action
    
    return states, actions


def get_state_threshold(state_distribution, diffusion_model, env, no_normalize, state_levels, percentile, batch_size):
    states, _ = load_states_actions(env, no_normalize)
    states = torch.tensor(states, dtype=torch.float32, device=device)
    state_error = compute_state_error(state_distribution, diffusion_model, states, state_levels, batch_size)
    return np.percentile(state_error, percentile)


def get_action_threshold(behavior_model, diffusion_model, env, no_normalize, action_levels, percentile, batch_size):
    states, actions = load_states_actions(env, no_normalize)
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.float32, device=device)
    action_error = compute_action_error(behavior_model, diffusion_model, actions, states, action_levels, batch_size)
    return np.percentile(action_error, percentile)


def compute_state_error(state_distribution, diffusion_model, states, state_levels, batch_size=256):
    recon_errors = []
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i + batch_size]
            batch_errors = torch.zeros(len(batch_states), device=states.device)

            for _ in range(state_levels):
                t = diffusion_model.make_sample_density()(shape=(len(batch_states),), device=states.device)
                noise = torch.randn_like(batch_states)

                t_expanded = append_dims(t, batch_states.ndim)
                noisy_states = batch_states + noise * t_expanded

                c_skip, c_out, c_in = [
                    append_dims(x, batch_states.ndim) 
                    for x in diffusion_model.get_diffusion_scalings(t)
                ]

                model_input = noisy_states * c_in
                model_output = state_distribution(model_input, None, torch.log(t) / 4)
                denoised_states = c_skip * noisy_states + c_out * model_output

                error = torch.norm(denoised_states - batch_states, dim=1)
                batch_errors += error

            batch_errors /= state_levels
            recon_errors.append(batch_errors.cpu())

    return torch.cat(recon_errors).numpy()


def compute_action_error(behavior_model, diffusion_model, actions, states, action_levels, batch_size=256):
    recon_errors = []
    with torch.no_grad():
        for i in range(0, len(actions), batch_size):
            batch_actions = actions[i:i + batch_size]
            batch_states = states[i:i + batch_size]
            batch_errors = torch.zeros(len(batch_actions), device=actions.device)

            for _ in range(action_levels):
                t = diffusion_model.make_sample_density()(shape=(len(batch_actions),), device=actions.device)
                noise = torch.randn_like(batch_actions)

                t_expanded = append_dims(t, batch_actions.ndim)
                noisy_actions = batch_actions + noise * t_expanded

                c_skip, c_out, c_in = [
                    append_dims(x, batch_actions.ndim) 
                    for x in diffusion_model.get_diffusion_scalings(t)
                ]

                model_input = noisy_actions * c_in
                model_output = behavior_model(model_input, batch_states, torch.log(t) / 4)
                denoised_actions = c_skip * noisy_actions + c_out * model_output

                error = torch.norm(denoised_actions - batch_actions, dim=1)
                batch_errors += error

            batch_errors /= action_levels
            recon_errors.append(batch_errors.cpu())

    return torch.cat(recon_errors).numpy()


def train(args):
    print(f"Loading dataset: {args.env_name}")
    states, actions = load_states_actions(args.env_name, args.no_normalize)
    state_dim, action_dim = states.shape[1], actions.shape[1]
    
    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)

    diffusion_model = DiffusionModel(
        sigma_data=args.sigma_data,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        device=device,
    )

    state_distribution = ScoreNetwork(
        x_dim=state_dim,
        hidden_dim=256,
        time_embed_dim=16,
        cond_dim=0,
        cond_mask_prob=0.0,
        num_hidden_layers=4,
        output_dim=state_dim,
        device=device,
        cond_conditional=False
    ).to(device)
    state_distribution_optimizer = torch.optim.Adam(state_distribution.parameters(), lr=3e-4)

    behavior_model = ScoreNetwork(
        x_dim=action_dim, 
        hidden_dim=256,
        time_embed_dim=16,
        cond_dim=state_dim,
        cond_mask_prob=0.0,
        num_hidden_layers=4,
        output_dim=action_dim,
        device=device,
        cond_conditional=True
    ).to(device)
    behavior_model_optimizer = torch.optim.Adam(behavior_model.parameters(), lr=3e-4)

    os.makedirs("./Diffusion_models/sd_models", exist_ok=True)
    state_distribution_path = f"./Diffusion_models/sd_models/{args.env_name}.pth"
    if os.path.exists(state_distribution_path):
        print(f"Loading pre-trained state distribution model from {state_distribution_path}")
        state_distribution.load_state_dict(torch.load(state_distribution_path, map_location=device))
    else:
        print("Training state distribution model...")

        wandb.init(
            project="Diffusion-SD",
            name=f"{args.env_name}",
            config=vars(args)
        )

        for epoch in tqdm(range(args.pretrain_epochs)):
            state_distribution_optimizer.zero_grad()
            loss = diffusion_model.diffusion_train_step(state_distribution, states_tensor, None)
            loss.backward()
            state_distribution_optimizer.step()

            wandb.log({"loss": loss.item()})

        torch.save(state_distribution.state_dict(), state_distribution_path)
        print(f"Saved pre-trained state distribution model to {state_distribution_path}")
        wandb.finish()

    os.makedirs("./Diffusion_models/bc_models", exist_ok=True)
    behavior_model_path = f"./Diffusion_models/bc_models/{args.env_name}.pth"
    if os.path.exists(behavior_model_path):
        print(f"Loading pre-trained behavior policy model from {behavior_model_path}")
        behavior_model.load_state_dict(torch.load(behavior_model_path, map_location=device))
    else:
        print("Training behavior policy model...")

        wandb.init(
            project="Diffusion-BC",
            name=f"{args.env_name}",
            config=vars(args)
        )

        for epoch in tqdm(range(args.pretrain_epochs)):
            behavior_model_optimizer.zero_grad()
            loss = diffusion_model.diffusion_train_step(behavior_model, actions_tensor, states_tensor)
            loss.backward()
            behavior_model_optimizer.step()

            wandb.log({"loss": loss.item()})

        torch.save(behavior_model.state_dict(), behavior_model_path)
        print(f"Saved pre-trained behavior policy model to {behavior_model_path}")
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='halfcheetah-medium-expert-v2')
    parser.add_argument('--pretrain_epochs', type=int, default=100000)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_normalize', default=False, action='store_true')
    parser.add_argument("--sigma_max", type=float, default=80)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--action_levels", type=int, default=1)
    parser.add_argument("--state_levels", type=int, default=1)

    args = parser.parse_args()

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(args.seed)
    train(args)