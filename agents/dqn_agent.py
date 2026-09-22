import copy
from collections import deque, namedtuple

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from agents.networks import AttentionInvariantCardExtractor, MultiModalQNetwork

# The transition namedtuple expects 'state' and  'next_state' to be dictionaries of numpy arrays.
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done", "next_action_mask"])

class ReplayBuffer:
    """
    Fixed-capacity store of transitions, sampled uniformly for learning.
    """
    def __init__(self, capacity: int,  *, seed: int | None = None):
        self._buffer = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed=seed)

    def push(self, state: dict[str, np.ndarray], action: int, reward: float, next_state: dict[str, np.ndarray], done: bool, next_action_mask: np.ndarray):
        """
        Store one transition.
        """
        state_copy = {k: np.array(v, dtype=np.float32) for k, v in state.items()}
        next_state_copy  = {k: np.array(v, dtype=np.float32) for k, v in next_state.items()}

        self._buffer.append(Transition(
            state_copy,
            action,
            reward,
            next_state_copy,
            done,
            np.array(next_action_mask, dtype=np.float32)
        ))
    
    def sample(self, batch_size: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Draw a uniform random minibatch, returned as batched tensors.
        """
        indices = self._rng.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        
        states = {}
        next_states = {}
        keys = batch[0].state.keys()

        for k in keys:
            states[k] = torch.from_numpy(np.stack([t.state[k] for t in batch]))
            next_states[k] = torch.from_numpy(np.stack([t.next_state[k] for t in batch]))

        actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)
        next_action_masks = torch.tensor(np.stack([t.next_action_mask for t in batch]), dtype=torch.float32)
        return states, actions, rewards, next_states, dones, next_action_masks

    def __len__(self) -> int:
        return len(self._buffer)


class DQNAgent:
    """
    Deep Q-Network: neural Q-leanring + experience replay + target network.
    """

    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Discrete, total_episodes: int, lr_start: float = 1e-3, lr_min: float = 1e-5, gamma: float = 0.99, epsilon_start: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.99975, buffer_capacity: int = 50_000, batch_size: int = 64, min_buffer_size: int = 1000, target_sync_every: int = 500, seed: int | None = None, card_extractor_cls: type[nn.Module] = AttentionInvariantCardExtractor):
        self.n_actions = int(action_space.n)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon = self.epsilon_start

        if seed is not None:
            torch.manual_seed(seed=seed)
        self._rng = np.random.default_rng(seed=seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the Multi-branch fusion network
        self.q_net = MultiModalQNetwork(observation_space, self.n_actions, card_extractor_out_dim=64).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr_start)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_episodes, eta_min=lr_min)

        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.target_sync_every = int(target_sync_every)
        self._update_count = 0

        self.batch_size = int(batch_size)
        self.min_buffer_size = int(min_buffer_size)
        self.buffer = ReplayBuffer(buffer_capacity, seed=seed)

        self.weight_norms = []
        self.grad_norms = []

    def q_values(self, state: dict[str, np.ndarray]) -> np.ndarray:
        """
        Evaluates the Q-Network for a single dictionary state.
        """
        state_t = {k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device) for k, v in  state.items()}
        with torch.no_grad():
            return self.q_net(state_t).squeeze(0).cpu().numpy()

    def select_action(self, state: dict[str, np.ndarray], action_mask: np.ndarray, *, is_greedy: bool = False) -> int:
        """
        Epsilon-greedy polcy respecting the environment's action mask.
        """
        valid_actions = np.flatnonzero(action_mask == 1)
        if not is_greedy and self._rng.random() < self.epsilon:
            return int(self._rng.choice(valid_actions))

        q_vals = self.q_values(state)
        q_vals[action_mask == 0] = -np.inf

        max_value = q_vals.max()
        candidates = np.flatnonzero(q_vals == max_value)
        return int(self._rng.choice(candidates))

    def store_transition(self, state: dict[str, np.ndarray], action: int, reward: float, next_state: dict[str, np.ndarray], terminated: bool, next_mask: np.ndarray) -> None:
        """
        Push to the buffer from train.py's temporal loop.
        """
        self.buffer.push(state=state, action=action, reward=reward, next_state=next_state, done=terminated, next_action_mask=next_mask)
    
    def optimize_model(self) -> None:
        """
        Draws a minibatch from the replay buffer and computes the semi-gradient TD error.
        """
        if len(self.buffer) < self.min_buffer_size:
            return
        
        states, actions, rewards, next_states, dones, next_action_masks = self.buffer.sample(self.batch_size)

        # Push sampled batch to GPU
        states = {k: v.to(self.device) for k, v in states.items()}
        next_states = {k: v.to(self.device) for k, v in next_states.items()}
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        next_action_masks = next_action_masks.to(self.device)

        # q(s, a) estimates
        q_sa = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.q_net(next_states)
            next_q_online = next_q_online.masked_fill(next_action_masks == 0, -float('inf'))
            best_next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states).gather(1, best_next_actions).squeeze(1)
            target = rewards + self.gamma * (1.0 - dones) * next_q_target

        loss = ((q_sa - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0) # Prevents steep updates from destabilizing the network weights.

        grad_sq_sums = [p.grad.pow(2).sum() for p in self.q_net.parameters() if p.grad is not None]
        if grad_sq_sums:
            grad_norm = torch.sqrt(torch.stack(grad_sq_sums).sum())
        else:
            grad_norm = torch.tensor(0.0)

        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.grad_norms.append(float(grad_norm))
        with torch.no_grad():
            weight_sq_sums = [p.pow(2).sum() for p in self.q_net.parameters()]
            total = torch.stack(weight_sq_sums).sum()
            
            self.weight_norms.append(float(torch.sqrt(total)))

    def end_episode(self) -> None:
        """
        Decay epsilon at the end of an episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self._update_count > 0:
            self.scheduler.step()

    def save_weights(self, filepath: str) -> None:
        torch.save(self.q_net.state_dict(), filepath)

    def load_weights(self, filepath: str) -> None:
        self.q_net.load_state_dict(torch.load(filepath))
        self.target_net.load_state_dict(self.q_net.state_dict())
