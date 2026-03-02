import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import random
import time
import wandb
from tqdm import trange
from utils import *
from pretrain_diffusion import *
from agents.doser import *
from diffusion.karras import DiffusionModel
from diffusion.mlps import ScoreNetwork
from pretrain_dynamics import DynamicsModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparameters = {
    'halfcheetah-medium-v2':        {'beta': 0.001, 'lam': 0.001, 'eta': 0.9, 'expectile': 0.9, 'percentile': 99, 'state_levels': 1, 'action_levels': 1, 'action_samples': 10, 'Q_min': -366},
    'hopper-medium-v2':             {'beta': 0.001, 'lam': 0.001, 'eta': 0.9, 'expectile': 0.9, 'percentile': 99, 'state_levels': 10, 'action_levels': 1, 'action_samples': 10, 'Q_min': -125},
    'walker2d-medium-v2':           {'beta': 0.01, 'lam': 0.001, 'eta': 0.9, 'expectile': 0.9, 'percentile': 99, 'state_levels': 10, 'action_levels': 1, 'action_samples': 10, 'Q_min': -471},
    'halfcheetah-medium-replay-v2': {'beta': 0.001, 'lam': 0.001, 'eta': 0.9, 'expectile': 0.9, 'percentile': 99, 'state_levels': 10, 'action_levels': 1, 'action_samples': 10, 'Q_min': -366},
    'hopper-medium-replay-v2':      {'beta': 0.001, 'lam': 0.001, 'eta': 0.9, 'expectile': 0.9, 'percentile': 99, 'state_levels': 1, 'action_levels': 1, 'action_samples': 10, 'Q_min': -125},
    'walker2d-medium-replay-v2':    {'beta': 0.001, 'lam': 0.001, 'eta': 0.9, 'expectile': 0.9, 'percentile': 99, 'state_levels': 1, 'action_levels': 1, 'action_samples': 10, 'Q_min': -471},
    'halfcheetah-medium-expert-v2': {'beta': 0.05,  'lam': 0.001, 'eta': 0.9, 'expectile': 0.7, 'percentile': 80, 'state_levels': 1, 'action_levels': 1, 'action_samples': 10, 'Q_min': -366},
    'hopper-medium-expert-v2':      {'beta': 0.05,  'lam': 0.001, 'eta': 0.9, 'expectile': 0.7, 'percentile': 80, 'state_levels': 1, 'action_levels': 1, 'action_samples': 10, 'Q_min': -125},
    'walker2d-medium-expert-v2':    {'beta': 0.05,  'lam': 0.001, 'eta': 0.9, 'expectile': 0.7, 'percentile': 80, 'state_levels': 1, 'action_levels': 1, 'action_samples': 10, 'Q_min': -471},
    'halfcheetah-expert-v2':        {'beta': 0.05,  'lam': 0.001, 'eta': 0.9, 'expectile': 0.7, 'percentile': 80, 'state_levels': 1, 'action_levels': 1, 'action_samples': 10, 'Q_min': -366},
    'hopper-expert-v2':             {'beta': 0.05,  'lam': 0.001, 'eta': 0.9, 'expectile': 0.7, 'percentile': 80, 'state_levels': 1, 'action_levels': 1, 'action_samples': 10, 'Q_min': -125},
    'walker2d-expert-v2':           {'beta': 0.05,  'lam': 0.001, 'eta': 0.9, 'expectile': 0.7, 'percentile': 80, 'state_levels': 1, 'action_levels': 1, 'action_samples': 10, 'Q_min': -471},
    'halfcheetah-random-v2':        {'beta': 0.001, 'lam': 0.001, 'eta': 0.9, 'expectile': 0.9, 'percentile': 99, 'state_levels': 1, 'action_levels': 1, 'action_samples': 10, 'Q_min': -366},
    'hopper-random-v2':             {'beta': 0.001, 'lam': 0.001, 'eta': 0.9, 'expectile': 0.9, 'percentile': 99, 'state_levels': 1, 'action_levels': 1, 'action_samples': 10, 'Q_min': -125},
    'walker2d-random-v2':           {'beta': 0.001, 'lam': 0.001, 'eta': 0.9, 'expectile': 0.9, 'percentile': 99, 'state_levels': 1, 'action_levels': 1, 'action_samples': 10, 'Q_min': -471},
}

def eval_policy(agent, env_name, seed, mean, std, return_states=False, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)
    eval_env.action_space.seed(seed + seed_offset)
    agent.actor.eval()
    avg_reward = 0.
    visit_states = []
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            visit_states.append(state[0])
            action = agent.actor.act(state, device)
            action = np.asarray(action, dtype=np.float32)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    agent.actor.train()

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    if return_states:
        return d4rl_score, np.array(visit_states)
    return d4rl_score


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="halfcheetah-medium-replay-v2")
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval_freq", default=2e4, type=int)
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99)
    parser.add_argument("--tau", default=0.005)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--target_update_freq", default=2, type=int)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--no_schedule", action="store_true")
    args = parser.parse_args()

    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.beta = hyperparameters[args.env_name]['beta']
    args.lam = hyperparameters[args.env_name]['lam']
    args.eta = hyperparameters[args.env_name]['eta']                        # Weight for compensation
    args.expectile = hyperparameters[args.env_name]['expectile']            # Expectile parameter for critic loss
    args.percentile = hyperparameters[args.env_name]['percentile']          # Percentile for OOD detection
    args.state_levels = hyperparameters[args.env_name]['state_levels']      # Number of noise levels for state reconstruction error
    args.action_levels = hyperparameters[args.env_name]['action_levels']    # Number of noise levels for action reconstruction error
    args.action_samples = hyperparameters[args.env_name]['action_samples']  # Number of ID action samples
    args.Q_min = hyperparameters[args.env_name]['Q_min']                    # Theoretical minimum Q value of the MDP

    print("---------------------------------------")
    print(f"Env: {args.env_name}, Seed: {args.seed}")
    print("---------------------------------------")

    wandb.init(
        project="DOSER",
        name = (f"{args.env_name}_beta{args.beta}_lam{args.lam}_eta{args.eta}_expectile{args.expectile}_percentile{args.percentile}"
                f"_slevels{args.state_levels}_alevels{args.action_levels}_asamples{args.action_samples}_seed{args.seed}"),
        config=vars(args)
    )

    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

    if not args.no_normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    dataset_states = replay_buffer.state
    
    # Diffusion model
    bc_model_path = f'./Diffusion_models/bc_models/{args.env_name}.pth'
    sd_model_path = f'./Diffusion_models/sd_models/{args.env_name}.pth'
    diffusion_model = DiffusionModel(
        sigma_data=0.5,
        sigma_min=0.002,
        sigma_max=80,
        device=device,
    )
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

    assert os.path.exists(bc_model_path), f"Behavior policy model {bc_model_path} not found!"
    behavior_model.load_state_dict(torch.load(bc_model_path, map_location=device))
    behavior_model.eval()

    assert os.path.exists(sd_model_path), f"State distribution model {sd_model_path} not found!"
    state_distribution.load_state_dict(torch.load(sd_model_path, map_location=device))
    state_distribution.eval()

    # Dynamics model
    dynamics_model = DynamicsModel(state_dim, action_dim).to(device)
    dynamics_model_path = f'./Dynamics_models/{args.env_name}.pth'
    assert os.path.exists(dynamics_model_path), f"Dynamics model {dynamics_model_path} not found!"
    dynamics_model.load_state_dict(torch.load(dynamics_model_path, map_location=device))
    dynamics_model.eval()

    # OOD state/action threshold
    state_threshold = round(get_state_threshold(state_distribution, diffusion_model, args.env_name, args.no_normalize, args.state_levels, args.percentile, args.batch_size), 6)
    action_threshold = round(get_action_threshold(behavior_model, diffusion_model, args.env_name, args.no_normalize, args.action_levels, args.percentile, args.batch_size), 6)
    print(f"State threshold for env '{args.env_name}' with state_levels {args.state_levels} and percentile {args.percentile}: {state_threshold}")
    print(f"Action threshold for env '{args.env_name}' with action_levels {args.action_levels} and percentile {args.percentile}: {action_threshold}")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "replay_buffer": replay_buffer,
        "behavior_model": behavior_model,
        "state_distribution": state_distribution,
        "diffusion_model": diffusion_model,
        "dynamics_model": dynamics_model,
        "discount": args.discount,
        "tau": args.tau,
        "policy_freq": args.policy_freq,
        "target_update_freq": args.target_update_freq,
        "schedule": not args.no_schedule,
        "Q_min": args.Q_min,
        "beta": args.beta,
        "lam": args.lam,
        "eta": args.eta,
        "expectile": args.expectile,
        "state_levels": args.state_levels,
        "action_levels": args.action_levels,
        "state_threshold": state_threshold,
        "action_threshold": action_threshold,
        "action_samples": args.action_samples
    }

    agent = DOSER(**kwargs)

    for t in trange(int(args.max_timesteps)):
        agent.train(args.batch_size)
        if (t + 1) % args.eval_freq == 0:
            print(f"Time steps: {t+1}")
            d4rl_score = eval_policy(agent, args.env_name, args.seed, mean, std, eval_episodes=args.eval_episodes)
            wandb.log({
                "d4rl_score": d4rl_score
                }, step=t+1)
            
    time.sleep(10)
