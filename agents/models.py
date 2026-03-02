import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class V_Network(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(V_Network, self).__init__()

        self.V = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.V(state)
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.Q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.Q3 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.Q4 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.V1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.V2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        h = torch.cat([state, action], dim=1)
        q1 = self.Q1(h)
        q2 = self.Q2(h)
        q3 = self.Q3(h)
        q4 = self.Q4(h)
        return q1, q2, q3, q4
    
    def q_min(self, state, action):
        q1, q2, q3, q4 = self.forward(state, action)
        return torch.min(torch.min(q1, q2), torch.min(q3, q4))
    
    def v(self, state):
        v1 = self.V1(state)
        v2 = self.V2(state)
        return v1, v2
    
    def v_min(self, state):
        v1, v2 = self.v(state)
        return torch.min(v1, v2)
    

# deterministic policy
# class Actor(nn.Module):
    # def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.max_action = max_action

    # def forward(self, state):
        return self.max_action * torch.tanh(self.actor(state))
    

# Gaussian policy
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, state, deterministic = False, need_log_prob = False):            
        hidden = self.actor(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        log_sigma = torch.clamp(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))
        
        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)
            log_prob = log_prob.unsqueeze(-1)

        return tanh_action * self.max_action, log_prob
    
    @torch.no_grad()
    def log_prob(self, state, action):
        """
        计算给定状态 state 和动作 action 在当前策略下的 log_prob，
        输入的 action 应为经过 tanh 和 max_action 缩放后的动作
        """
        hidden = self.actor(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)
        
        log_sigma = torch.clamp(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))
        
        # 将动作从 [-max_action, max_action] 映射回 pre-tanh 空间
        eps = 1e-6
        clipped_action = torch.clamp(action / self.max_action, -1 + eps, 1 - eps)
        pre_tanh_action = 0.5 * torch.log((1 + clipped_action) / (1 - clipped_action))  # atanh(x)

        # 高斯对 pre-tanh 的 log 概率
        log_prob = policy_dist.log_prob(pre_tanh_action).sum(dim=-1, keepdim=True)

        # 减去 tanh 的雅可比修正项
        log_prob -= torch.log(1 - clipped_action.pow(2) + eps).sum(dim=-1, keepdim=True)

        return log_prob.unsqueeze(-1)
    
    @torch.no_grad()
    def act(self, state, device):
        deterministic = not self.training
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self(state, deterministic=deterministic)[0].cpu().data.numpy().flatten()
        return action
        