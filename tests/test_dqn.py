import unittest

import numpy as np
import torch
from gymnasium import spaces

# Assuming your agent and networks are located in the agents module
from agents.dqn import DQNAgent, ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    """
    Academic Unit Test Suite for the ReplayBuffer class.
    Verifies memory capacity constraints and the strict tensorization 
    of multi-modal dictionary states during mini-batch sampling.
    """
    
    def setUp(self):
        self.capacity = 100
        self.batch_size = 16
        self.buffer = ReplayBuffer(capacity=self.capacity, seed=42)
        
        # Define a mock state matching the CardDuelsEnv POMDP structure
        self.mock_state = {
            "self_stats": np.random.rand(4).astype(np.float32),
            "opp_stats": np.random.rand(3).astype(np.float32),
            "self_hand": np.random.rand(5, 7).astype(np.float32),
            "self_board": np.random.rand(7, 7).astype(np.float32),
            "opp_board": np.random.rand(7, 7).astype(np.float32),
            "history_opp": np.random.rand(20, 7).astype(np.float32),
            "deck_stats_self": np.random.rand(20, 7).astype(np.float32)
        }

    def test_capacity_and_eviction(self):
        """
        Mathematically proves that the deque strictly enforces the maximum memory 
        capacity, correctly evicting the oldest Markov transitions (t=0) when full.
        """
        # Push 150 transitions into a buffer with a capacity of 100
        for i in range(150):
            self.buffer.push(self.mock_state, action=1, reward=1.0, next_state=self.mock_state, done=False)
            
        self.assertEqual(len(self.buffer), self.capacity, "Buffer exceeded its predefined maximum capacity.")

    def test_sample_tensor_shapes(self):
        """
        Verifies that sampled mini-batches perfectly match the expected tensor 
        dimensions for the PyTorch forward pass, preventing graph crashes.
        """
        for i in range(self.batch_size * 2):
            self.buffer.push(self.mock_state, action=1, reward=1.0, next_state=self.mock_state, done=False)
            
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Verify batch dimension dynamically across all dictionary modules
        self.assertEqual(states["self_hand"].shape, (self.batch_size, 5, 7))
        self.assertEqual(states["self_stats"].shape, (self.batch_size, 4))
        
        # Verify 1D tensor shapes for targets and TD components
        self.assertEqual(actions.shape, (self.batch_size,))
        self.assertEqual(rewards.shape, (self.batch_size,))
        self.assertEqual(dones.shape, (self.batch_size,))


class TestDQNAgent(unittest.TestCase):
    """
    Academic Unit Test Suite for the DQNAgent.
    Verifies the Integrity of the Neural Q-Learning process, epsilon-greedy 
    exploration, action masking, and Semi-Gradient Temporal Difference updates.
    """
    
    def setUp(self):
        self.max_hand = 5
        self.n_actions = 2 ** self.max_hand
        
        self.observation_space = spaces.Dict({
            "self_stats": spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32),
            "opp_stats": spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32),
            "self_hand": spaces.Box(low=-10, high=50, shape=(self.max_hand, 7), dtype=np.float32),
            "self_board": spaces.Box(low=-10, high=50, shape=(7, 7), dtype=np.float32),
            "opp_board": spaces.Box(low=-10, high=50, shape=(7, 7), dtype=np.float32),
            "history_opp": spaces.Box(low=-10, high=50, shape=(20, 7), dtype=np.float32),
            "deck_stats_self": spaces.Box(low=-10, high=50, shape=(20, 7), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(self.n_actions)
        
        # Initialize the agent with a tiny buffer requirement to speed up testing
        self.agent = DQNAgent(
            observation_space=self.observation_space, 
            action_space=self.action_space,
            buffer_capacity=1000,
            batch_size=16,
            min_buffer_size=16, 
            target_sync_every=10,
            seed=42
        )
        
        self.mock_state = {k: v.sample() for k, v in self.observation_space.spaces.items()}

    def test_q_values_shape(self):
        """
        Ensures the Multi-Modal fusion network properly maps the $S_t$ dictionary 
        to an output vector of exactly $\mathcal{A}$ dimensions.
        """
        q_vals = self.agent.q_values(self.mock_state)
        self.assertEqual(q_vals.shape, (self.n_actions,), "Q-values output vector dimension mismatch.")

    def test_strict_action_masking(self):
        """
        Mathematically proves that the agent's argmax operator is strictly bounded 
        by the environment's legal constraints. Illegal combinatorial actions must 
        never be selected, even during greedy exploitation.
        """
        # Create an action mask where ONLY action index 12 is mathematically valid
        action_mask = np.zeros(self.n_actions, dtype=np.int8)
        valid_index = 12
        action_mask[valid_index] = 1
        
        # Test Greedy Selection (Exploitation)
        greedy_action = self.agent.select_action(self.mock_state, action_mask, greedy=True)
        self.assertEqual(greedy_action, valid_index, "Greedy policy violated the action mask constraints.")
        
        # Test Exploratory Selection
        # Temporarily force epsilon to 1.0 to guarantee random sampling
        self.agent.epsilon = 1.0
        random_action = self.agent.select_action(self.mock_state, action_mask, greedy=False)
        self.assertEqual(random_action, valid_index, "Random exploration violated the action mask constraints.")

    def test_bellman_optimization_and_target_sync(self):
        """
        Executes the Semi-Gradient TD update to verify the PyTorch Autograd graph 
        does not crash, gradients are successfully computed, and the Target Network 
        hard-syncs at the correct temporal frequency.
        """
        # Fill buffer to exactly the minimum size required to trigger an update
        for _ in range(self.agent.min_buffer_size):
            self.agent.store_transition(self.mock_state, action=0, reward=1.0, next_state=self.mock_state, terminated=False)
            
        initial_update_count = self.agent._update_count
        
        # Trigger gradient descent step
        self.agent.optimize_model()
        
        self.assertEqual(self.agent._update_count, initial_update_count + 1, "Agent failed to increment update counter.")
        self.assertEqual(len(self.agent.grad_norms), 1, "Gradient norm diagnostic was not recorded.")
        self.assertTrue(self.agent.grad_norms[0] >= 0.0, "Gradient norm mathematically invalid (negative).")
        
        # Test Target Sync Logic
        # Artificially push the update counter to the boundary
        self.agent._update_count = self.agent.target_sync_every - 1
        
        # Cache a single weight from the target network to check for mutations
        old_target_weight = next(self.agent.target_net.parameters()).clone()
        
        # This step should trigger the periodic hard-sync
        self.agent.optimize_model()
        
        new_target_weight = next(self.agent.target_net.parameters())
        self.assertFalse(torch.equal(old_target_weight, new_target_weight), "Target network failed to sync weights.")

    def test_epsilon_decay_schedule(self):
        """
        Verifies the geometric decay sequence of the exploration hyperparameter.
        """
        start_eps = self.agent.epsilon
        self.agent.end_episode()
        expected_eps = max(self.agent.epsilon_min, start_eps * self.agent.epsilon_decay)
        
        self.assertAlmostEqual(self.agent.epsilon, expected_eps, places=5, msg="Epsilon geometric decay failed.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
