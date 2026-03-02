import copy
import torch
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from .models import Actor, Critic

D4RL_SUPPRESS_IMPORT_ERROR=1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DOSER(object):
    def __init__(
            self, 
            state_dim, 
            action_dim, 
            max_action, 
            replay_buffer, 
            behavior_model, 
            state_distribution,
            diffusion_model,
            dynamics_model,
            Q_min, 
            state_levels,
            action_levels,
            state_threshold, 
            action_threshold,
            beta,
            lam,
            eta,
            expectile,
            action_samples,
            discount=0.99, 
            tau=0.005, 
            policy_freq=2,
            target_update_freq=2, 
            schedule=True
        ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = self.log_alpha.exp().detach()

        self.replay_buffer = replay_buffer
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.target_update_freq = target_update_freq
        self.behavior_model = behavior_model
        self.state_distribution = state_distribution
        self.diffusion_model = diffusion_model
        self.dynamics_model = dynamics_model
        self.Q_min = Q_min
        self.state_levels = state_levels
        self.action_levels = action_levels
        self.state_threshold = state_threshold
        self.action_threshold = action_threshold
        self.beta = beta
        self.lam = lam
        self.eta = eta
        self.expectile = expectile
        self.action_samples = action_samples
        self.schedule = schedule
        if schedule:
            self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, int(int(1e6) / policy_freq))
            # self.critic_lr_schedule = CosineAnnealingLR(self.critic_optimizer, int(int(1e6) / policy_freq))
        self.total_it = 0


    def compute_state_error(self, states, batch_size=256):
        recon_errors = []
        with torch.no_grad():
            for i in range(0, len(states), batch_size):
                batch_states = states[i:i + batch_size]
                batch_errors = torch.zeros(len(batch_states), device=states.device)

                for _ in range(self.state_levels):
                    t = self.diffusion_model.make_sample_density()(shape=(len(batch_states),), device=states.device)
                    noise = torch.randn_like(batch_states)

                    t_expanded = t.view(-1, *([1] * (batch_states.ndim - 1)))
                    noisy_states = batch_states + noise * t_expanded

                    c_skip, c_out, c_in = [
                        x.view(-1, *([1] * (batch_states.ndim - 1))) 
                        for x in self.diffusion_model.get_diffusion_scalings(t)
                    ]

                    model_input = noisy_states * c_in
                    model_output = self.state_distribution(model_input, None, torch.log(t) / 4)
                    denoised_states = c_skip * noisy_states + c_out * model_output

                    error = torch.norm(denoised_states - batch_states, dim=1)
                    batch_errors += error

                batch_errors /= self.state_levels
                recon_errors.append(batch_errors)

        return torch.cat(recon_errors)


    def compute_action_error(self, actions, states, batch_size=256):
        recon_errors = []
        with torch.no_grad():
            for i in range(0, len(actions), batch_size):
                batch_actions = actions[i:i + batch_size]
                batch_states = states[i:i + batch_size]
                batch_errors = torch.zeros(len(batch_actions), device=actions.device)

                for _ in range(self.action_levels):
                    t = self.diffusion_model.make_sample_density()(shape=(len(batch_actions),), device=actions.device)
                    noise = torch.randn_like(batch_actions)

                    t_expanded = t.view(-1, *([1] * (batch_actions.ndim - 1)))
                    noisy_actions = batch_actions + noise * t_expanded

                    c_skip, c_out, c_in = [
                        x.view(-1, *([1] * (batch_actions.ndim - 1))) 
                        for x in self.diffusion_model.get_diffusion_scalings(t)
                    ]

                    model_input = noisy_actions * c_in
                    model_output = self.behavior_model(model_input, batch_states, torch.log(t) / 4)
                    denoised_actions = c_skip * noisy_actions + c_out * model_output

                    error = torch.norm(denoised_actions - batch_actions, dim=1)
                    batch_errors += error

                batch_errors /= self.action_levels
                recon_errors.append(batch_errors)

        return torch.cat(recon_errors)


    def select_action(self, state):
        with torch.no_grad():
            self.actor.eval()
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.actor(state)[0].cpu().data.numpy().flatten()
            self.actor.train()
            return action
        

    def select_best_id_action(self, state):
        all_actions = self.diffusion_model.sample(
            model=self.behavior_model, 
            cond=state, 
            action_samples=self.action_samples
        )  # [B, action_samples, action_dim]
        
        batch_size = state.size(0)
        
        # Expand states to match action samples for batched Q-value computation
        states_expanded = state.unsqueeze(1).expand(-1, self.action_samples, -1)  # [B, action_samples, state_dim]
        
        # Reshape states and actions for critic input
        flat_states = states_expanded.reshape(-1, state.shape[-1])  # [B * action_samples, state_dim]
        flat_actions = all_actions.reshape(-1, all_actions.shape[-1])  # [B * action_samples, action_dim]
        
        # Get Q-values from critic
        q1, q2, q3, q4 = self.critic(flat_states, flat_actions)
        q_values = torch.min(torch.min(q1, q2), torch.min(q3, q4))  # [B * action_samples, 1]
        q_values = q_values.view(batch_size, self.action_samples)  # [B, action_samples]
        
        # Select best actions and their Q-values
        best_indices = torch.argmax(q_values, dim=1)  # [B]
        best_id_actions = all_actions[torch.arange(batch_size), best_indices]  # [B, action_dim]
        best_q_values = q_values[torch.arange(batch_size), best_indices]  # [B]
        
        return best_id_actions, best_q_values


    def alpha_loss(self, state):
        with torch.no_grad():
            _, log_prob = self.actor(state, need_log_prob=True)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()
        return alpha_loss
    

    def actor_loss(self, state):
        pi, pi_log_prob = self.actor(state, need_log_prob=True)
        pi_Q1, pi_Q2, pi_Q3, pi_Q4 = self.critic(state, pi)
        pi_Q = torch.cat([pi_Q1, pi_Q2, pi_Q3, pi_Q4], dim=1)
        pi_Q, _ = torch.min(pi_Q, dim=1)
        if pi_Q.mean().item() > 5e4:
            exit(0)
        actor_loss = (self.alpha * pi_log_prob - pi_Q).mean()
        return actor_loss
    

    def critic_loss(self, state, action, reward, next_state, not_done):
        def expectile_loss(diff, expectile):
            weight = torch.where(diff > 0, expectile, (1 - expectile))
            return weight * (diff ** 2)
        
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor_target(next_state, need_log_prob=True)

            next_Q1, next_Q2, next_Q3, next_Q4 = self.critic_target(next_state, next_action)  
            next_Q = torch.min(torch.min(next_Q1, next_Q2), torch.min(next_Q3, next_Q4))
            next_Q = next_Q - self.alpha * next_action_log_prob
            target_Q = reward + not_done * self.discount * next_Q

            next_V1, next_V2 = self.critic_target.v(next_state)
            next_V = torch.min(next_V1, next_V2)
            target_V = reward + not_done * self.discount * next_V
            
        current_Q1, current_Q2, current_Q3, current_Q4 = self.critic(state, action)
        current_Q = torch.cat([current_Q1, current_Q2, current_Q3, current_Q4], dim=1)

        current_V1, current_V2 = self.critic.v(state)
        q = torch.min(torch.min(current_Q1, current_Q2), torch.min(current_Q3, current_Q4))
        value_loss = expectile_loss(target_V - current_V1, self.expectile).mean() + \
                     expectile_loss(target_V - current_V2, self.expectile).mean()

        # with torch.no_grad():
            # next_V = self.critic_target.V(next_state)
            # target_Q = reward + not_done * self.discount * next_V

        # Bellman loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + \
                      F.mse_loss(current_Q3, target_Q) + F.mse_loss(current_Q4, target_Q) + value_loss
        
        # action reconstruction error
        pi, _ = self.actor(state)
        pi_error = self.compute_action_error(pi, state)

        # Predict next state s0
        pred_next_state = self.dynamics_model(state, pi)[..., :-1]  # [B, state_dim]
        value_s_pi = self.critic.v_min(pred_next_state)
        best_id_action, best_id_q = self.select_best_id_action(state)  # [B, action_dim], [B]
        best_id_q = best_id_q.unsqueeze(1)  # [B, 1]
        best_id_next_state = self.dynamics_model(state, best_id_action)[..., :-1]  # [B, state_dim]
        value_s_in = self.critic.v_min(best_id_next_state)

        pred_next_state_error = self.compute_state_error(pred_next_state)

        if torch.isnan(pi_error).any() or torch.isnan(pred_next_state_error).any():
            print("Warning: Reconstruction error contains NaN values!")
        
        ood_action_mask = (pi_error > self.action_threshold)
        ood_next_state_mask = (pred_next_state_error > self.state_threshold)
        negative_value_mask = (value_s_pi < value_s_in).squeeze(-1)
        positive_value_mask = (value_s_pi >= value_s_in).squeeze(-1)
        negative_ood_action_mask = (ood_action_mask & (ood_next_state_mask | negative_value_mask)).float().unsqueeze(1)
        positive_ood_action_mask = (ood_action_mask & (~ood_next_state_mask) & positive_value_mask).float().unsqueeze(1)

        pi_Q1, pi_Q2, pi_Q3, pi_Q4 = self.critic(state, pi)
        pi_Q = torch.cat([pi_Q1, pi_Q2, pi_Q3, pi_Q4], dim=1)
        qmin = (self.Q_min * torch.ones_like(pi_Q)).detach()

        reg_loss = self.beta * (((pi_Q - qmin) ** 2) * negative_ood_action_mask).mean()

        value_diff = (value_s_pi - value_s_in).clamp(min=0.0)
        q_comp_target = self.eta * (best_id_q + value_diff).detach()

        vc_loss = self.lam * (((pi_Q - q_comp_target) ** 2) * positive_ood_action_mask).mean()

        critic_loss += reg_loss + vc_loss
        
        return critic_loss, reg_loss, vc_loss, current_Q, qmin, positive_ood_action_mask, negative_ood_action_mask, ood_action_mask, best_id_q, value_s_pi, value_s_in, pi_Q, pi_error, pred_next_state_error, q_comp_target


    def train(self, batch_size=256):
        self.total_it += 1

        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        # Alpha update
        alpha_loss = self.alpha_loss(state)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        # Actor update
        if self.total_it % self.policy_freq == 0:
            actor_loss = self.actor_loss(state)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            if self.schedule:
                self.actor_lr_schedule.step()

        # Critic update
        critic_loss, reg_loss, vc_loss, current_Q, qmin, positive_ood_action_mask, negative_ood_action_mask, ood_action_mask, best_id_q, value_s_pi, value_s_in, pi_Q, pi_error, pred_next_state_error, q_comp_target = self.critic_loss(state, action, reward, next_state, not_done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        # if self.schedule:
        #     self.critic_lr_schedule.step()

        pos_mask = positive_ood_action_mask.float()
        neg_mask = negative_ood_action_mask.float()
        ood_mask = ood_action_mask.float() 
        
        total_count = batch_size
        ood_count = ood_mask.sum().item()
        pos_count = pos_mask.sum().item()
        neg_count = neg_mask.sum().item()
        id_count = total_count - ood_count

        if total_count > 0:
            id_ratio = id_count / total_count
            ood_ratio = ood_count / total_count
            pos_ratio = pos_count / total_count
            neg_ratio = neg_count / total_count
        else:
            id_ratio = ood_ratio = pos_ratio = neg_ratio = 0.0

        # Target networks update
        if self.total_it % self.target_update_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.total_it % 10000 == 0:
            with torch.no_grad():
                pi = self.actor(state)[0]
                Q_pi1, Q_pi2, Q_pi3, Q_pi4 = self.critic(state, pi)
                Q_pi = torch.cat([Q_pi1, Q_pi2, Q_pi3, Q_pi4],dim=1)

                wandb.log({"train/critic_loss": critic_loss.item(),
                            "train/reg_loss": reg_loss.item(),
                            "train/vc_loss": vc_loss.item(),
                            "train/actor_loss": actor_loss.item(),
                            'Q/Qmin': qmin.mean().item(),
                            'Q/pi': Q_pi.mean().item(),
                            'Q/a': current_Q.mean().item(),
                            "ood/id_count": id_count,
                            "ood/id_ratio": id_ratio,
                            "ood/ood_count": ood_count,
                            "ood/ood_ratio": ood_ratio,
                            "ood/pos_count": pos_count,
                            "ood/pos_ratio": pos_ratio,
                            "ood/neg_count": neg_count,
                            "ood/neg_ratio": neg_ratio,
                            "ood/total_count": total_count
                            }, step=self.total_it)
                