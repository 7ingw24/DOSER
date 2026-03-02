import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gym
import d4rl
import random
import wandb
from tqdm import tqdm
from utils import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DynamicsModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1)  # Predict next state and reward  
        )

    def forward(self, state, action):
        h = torch.cat([state, action], dim=1)
        return self.model(h)


def train(args):
    print(f"Loading dataset: {args.env_name}")
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
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

    if not args.no_normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    model = DynamicsModel(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    os.makedirs("./Dynamics_models", exist_ok=True)
    model_path = f"./Dynamics_models/{args.env_name}.pth"

    if os.path.exists(model_path):
        print(f"Loading pre-trained dynamics model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training dynamics model...")

        wandb.init(
            project = "Pretrain-Dynamics",
            name = f"{args.env_name}",
            config = vars(args)
        )

        for epoch in tqdm(range(args.pretrain_epochs)):
            state, action, next_state, reward, _ = replay_buffer.sample(args.batch_size)

            target = torch.cat([next_state, reward], dim=-1)
            pred = model(state, action)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        torch.save(model.state_dict(), model_path)
        print(f"Saved trained dynamics model to {model_path}")

        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='halfcheetah-medium-replay-v2')
    parser.add_argument('--pretrain_epochs', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_normalize', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    train(args)
