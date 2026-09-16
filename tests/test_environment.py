import unittest

import numpy as np

from env.cards import Card
from env.environment import CardDuelsEnv


class TestCardDuelsEnv(unittest.TestCase):
    """
    Academic Unit Test Suite for the CardDuelsEnv Gymnasium wrapper.
    Verifies POMDP observation constraints, action masking, and MDP transition logic.
    """

    def setUp(self):
        """
        Initializes a controlled environment with a mock card pool.
        """
        self.mock_card_data = [
            {"id": "01", "name": "A", "cost": 1, "atk": 2, "def": 1, "m_draw": 0, "m_burn": 0, "m_heal": 0, "m_wipe": 0},
            {"id": "02", "name": "B", "cost": 3, "atk": 4, "def": 4, "m_draw": 0, "m_burn": 0, "m_heal": 0, "m_wipe": 0},
            {"id": "03", "name": "C", "cost": 5, "atk": 6, "def": 6, "m_draw": 1, "m_burn": 0, "m_heal": 0, "m_wipe": 0},
        ]
        self.card_pool = {data["id"]: Card(data) for data in self.mock_card_data}
        self.deck_list = ["01", "01", "02", "02", "03"]
        
        self.env = CardDuelsEnv(
            card_pool=self.card_pool, 
            deck_list=self.deck_list, 
            max_hand=3, # Reduced for easier combinatorial testing (2^3 = 8 actions)
            max_board=3, 
            total_unique_cards=3
        )

    def test_observation_space_compliance(self):
        """
        Verifies that the generated observation dictionary strictly matches the 
        Gymnasium observation_space definitions (bounds, shapes, and dtypes).
        """
        obs, info = self.env.reset()
        
        # Gymnasium provides a built-in contains() method to mathematically verify the tensor
        is_compliant = self.env.observation_space.contains(obs)
        self.assertTrue(is_compliant, "The generated observation violates the defined observation_space bounds or dtypes.")

    def test_starting_hand_initialization(self):
        """
        Verifies the Starting Hand draw logic.
        The active player (who draws for Turn 1) should hit exactly max_hand.
        The opponent should have exactly max_hand - 1.
        """
        self.env.reset()
        
        p1 = self.env.board.players[0]
        p2 = self.env.board.players[1]
        
        self.assertEqual(len(p1.hand), self.env.max_hand)
        self.assertEqual(len(p2.hand), self.env.max_hand - 1)

    def test_action_masking_logic(self):
        """
        Verifies that the info dictionary correctly masks mathematically illegal actions
        based on combinatorial mana costs and empty hand slots.
        """
        self.env.reset()
        
        p1 = self.env.board.get_active_player()
        
        # Manually force the hand and mana state for controlled testing
        p1.mana = 4
        # Hand holds: Cost 1, Cost 3. (Slot 2 is empty)
        p1.hand = [self.card_pool["01"], self.card_pool["02"]] 
        
        info = self.env._get_info()
        mask = info["action_mask"]
        
        # Action mapping for H=3 (Binary format: '000' to '111')
        # 0: 000 (Cost 0) -> Valid
        # 1: 001 (Requires Slot 2) -> Invalid (Empty Slot)
        # 2: 010 (Requires Slot 1: Cost 3) -> Valid
        # 3: 011 (Requires Slot 1 & 2) -> Invalid (Empty Slot)
        # 4: 100 (Requires Slot 0: Cost 1) -> Valid
        # 5: 101 (Requires Slot 0 & 2) -> Invalid (Empty Slot)
        # 6: 110 (Requires Slot 0 & 1: Cost 4) -> Valid (Mana is exactly 4)
        # 7: 111 (Requires all slots) -> Invalid
        
        expected_mask = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
        np.testing.assert_array_equal(mask, expected_mask)

    def test_pomdp_deck_sorting(self):
        """
        Verifies that the observation matrix hides the true draw order (sorted by ID),
        without permanently sorting the actual engine deck object.
        """
        self.env.reset()
        p1 = self.env.board.get_active_player()
        
        # Force a specific scrambled order in the actual deck
        p1.deck.cards = [self.card_pool["03"], self.card_pool["01"], self.card_pool["02"]]
        
        obs = self.env._get_obs()
        deck_matrix = obs["deck_stats_self"]
        
        # The first row of the observation matrix should be Card "01" (Cost 1), because it's sorted by ID
        self.assertEqual(deck_matrix[0][0], 1.0) # Index 0 of card vector is Cost
        
        # But the ACTUAL deck object should remain untouched (Card "03" on top)
        self.assertEqual(p1.deck.cards[0].id, "03")

    def test_alternating_mdp_step(self):
        """
        Verifies that taking a step successfully hands control over to the opponent.
        """
        obs_t0, _ = self.env.reset()
        
        initial_active_idx = self.env.board.active_p_idx
        
        # Action 0 means "play no cards", effectively passing the turn
        obs_t1, reward, terminated, truncated, info = self.env.step(action=0)
        
        new_active_idx = self.env.board.active_p_idx
        
        self.assertNotEqual(initial_active_idx, new_active_idx)
        self.assertFalse(terminated)
        self.assertEqual(reward, 0.0) # Game hasn't ended

if __name__ == '__main__':
    unittest.main(verbosity=2)
