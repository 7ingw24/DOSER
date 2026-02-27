import numpy as np
import torch
import d4rl


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def convert_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]
		return (
			torch.FloatTensor(self.state).to(self.device),
			torch.FloatTensor(self.action).to(self.device),
			torch.FloatTensor(self.next_state).to(self.device),
			torch.FloatTensor(self.reward).to(self.device),
			torch.FloatTensor(self.not_done).to(self.device)
		)

	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std
    

class EpisodeReplayBuffer:
    def __init__(self, replay_buffer, episode_length=1000):
        self.replay_buffer = replay_buffer
        self.dataset = {
            "observations": replay_buffer.state,
            "actions": replay_buffer.action,
            "next_observations": replay_buffer.next_state,
            "rewards": replay_buffer.reward,
            "terminals": 1. - replay_buffer.not_done
        }

        if "timeouts" in self.dataset:
            done_idxs = np.where(self.dataset["timeouts"])[0]
        else:
            done_idxs = np.where(self.dataset["terminals"])[0]

        if len(done_idxs) == 0:
            print("[Warning] No episode boundaries found! Using fixed-length segmentation.")
            N = len(self.dataset["observations"])
            self.episode_starts = np.arange(0, N, episode_length)
            self.episode_ends = np.append(self.episode_starts[1:], N)
        else:
            self.episode_starts = np.insert(done_idxs[:-1] + 1, 0, 0)
            self.episode_ends = done_idxs + 1

        lengths = [e - s for s, e in zip(self.episode_starts, self.episode_ends)]
        print(f"[EpisodeReplayBuffer] Loaded {len(lengths)} episodes, avg length: {np.mean(lengths):.1f}")

    def sample_episode(self):
        ep_idx = np.random.randint(len(self.episode_starts))
        start = self.episode_starts[ep_idx]
        end = self.episode_ends[ep_idx]
        episode = {
            "states": self.dataset["observations"][start:end],
            "actions": self.dataset["actions"][start:end],
            "next_states": self.dataset["next_observations"][start:end],
            "rewards": self.dataset["rewards"][start:end],
            "dones": self.dataset["terminals"][start:end],
        }
        return episode
