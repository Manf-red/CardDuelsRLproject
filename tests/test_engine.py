import unittest

import numpy as np

from env.cards import Card, Deck
from env.engine import GameState, Player


class TestCardDuelsEngine(unittest.TestCase):
    """
    Academic Unit Test Suite for the Card Duels MDP State Machine.
    """

    def setUp(self):
        """
        Initializes a deterministic mock card pool before every test to isolate state variables.
        """
        self.mock_card_data = [
            {"id": "01", "name": "Vanilla Attacker", "cost": 2, "atk": 4, "def": 2, "m_draw": 0, "m_burn": 0, "m_heal": 0, "m_wipe": 0},
            {"id": "02", "name": "Vanilla Tank", "cost": 3, "atk": 0, "def": 5, "m_draw": 0, "m_burn": 0, "m_heal": 0, "m_wipe": 0},
            {"id": "03", "name": "Burn Spell", "cost": 1, "atk": 0, "def": 0, "m_draw": 0, "m_burn": 3, "m_heal": 0, "m_wipe": 0},
            {"id": "04", "name": "Heal Spell", "cost": 1, "atk": 0, "def": 0, "m_draw": 0, "m_burn": 0, "m_heal": 5, "m_wipe": 0},
            {"id": "05", "name": "Wipe Spell", "cost": 5, "atk": 0, "def": 0, "m_draw": 0, "m_burn": 0, "m_heal": 0, "m_wipe": 2},
            {"id": "06", "name": "Draw Spell", "cost": 2, "atk": 0, "def": 0, "m_draw": 2, "m_burn": 0, "m_heal": 0, "m_wipe": 0},
        ]
        self.card_pool = {data["id"]: Card(data) for data in self.mock_card_data}

    def test_card_vectorization(self):
        """
        Verifies that the Card class correctly parses data into the dense float32 vector.
        """
        card = self.card_pool["01"]
        vec = card.to_vector()
        
        self.assertEqual(vec.shape, (7,))
        self.assertEqual(vec.dtype, np.float32)
        np.testing.assert_array_equal(vec, np.array([2, 4, 2, 0, 0, 0, 0], dtype=np.float32))

    def test_fatigue_mechanic(self):
        """
        Verifies that drawing from an empty deck correctly applies cumulative unblockable damage.
        """
        # Deck with only 1 card
        deck = Deck(self.card_pool, ["01"])
        player = Player(deck, max_hp=30)
        
        player.draw_card() # Draws the only card. Deck is now empty. HP remains 30.
        self.assertEqual(player.hp, 30)
        
        player.draw_card() # Fatigue 1 triggered
        self.assertEqual(player.hp, 29)
        self.assertEqual(player.fatigue_counter, 1)
        
        player.draw_card() # Fatigue 2 triggered (Cumulative: 29 - 2 = 27)
        self.assertEqual(player.hp, 27)
        self.assertEqual(player.fatigue_counter, 2)

    def test_combat_resolution_and_spillover(self):
        """
        Verifies the 'Highest DEF First' sorting rule and sequential damage subtraction.
        """
        p1 = Player(Deck(self.card_pool, []), max_hp=30)
        p2 = Player(Deck(self.card_pool, []), max_hp=30)
        board = GameState(p1, p2)
        
        # Manually set up the board state
        p2.board = [self.card_pool["01"], self.card_pool["02"]] # Defender has DEF 2 and DEF 5
        deployed = [self.card_pool["01"], self.card_pool["01"]] # Attacker deploys two ATK 4 cards (Total ATK = 8)
        
        # Execute combat phase
        board._resolve_combat(p1, p2, deployed)
        
        # 8 ATK hits DEF 5 first (destroyed, 3 ATK remaining).
        # 3 ATK hits DEF 2 next (destroyed, 1 ATK remaining).
        # 1 ATK spills over to HP.
        self.assertEqual(len(p2.board), 0)
        self.assertEqual(p2.hp, 29) # 30 - 1 spillover
        self.assertEqual(len(p2.history), 2) # Both defenders sent to graveyard
        
        # Attacker's deployed cards should now be on their board
        self.assertEqual(len(p1.board), 2)

    def test_special_effects(self):
        """
        Verifies Burn, capped Heal, and randomized Board Wipe mechanics.
        """
        p1 = Player(Deck(self.card_pool, []), max_hp=30)
        p2 = Player(Deck(self.card_pool, []), max_hp=30)
        board = GameState(p1, p2)
        
        # Set up scenario
        p1.hp = 28 # Injured attacker to test healing cap
        p2.board = [self.card_pool["01"], self.card_pool["02"], self.card_pool["01"]] # 3 defenders
        
        # Deploy: 1 Burn (3 dmg), 1 Heal (5 hp), 1 Wipe (destroys 2 cards)
        deployed = [self.card_pool["03"], self.card_pool["04"], self.card_pool["05"]]
        
        board._resolve_combat(p1, p2, deployed)
        
        # Verify Burn (p2 takes 3 direct damage)
        self.assertEqual(p2.hp, 27)
        
        # Verify Capped Heal (p1 heals 5, but is capped at 30, not 33)
        self.assertEqual(p1.hp, 30)
        
        # Verify Wipe (2 out of 3 defenders destroyed before any physical damage calculation)
        self.assertEqual(len(p2.board), 1)

    def test_play_turn_action_masking(self):
        """
        Verifies that the engine correctly deducts mana and ignores cards the player cannot afford.
        """
        p1 = Player(Deck(self.card_pool, []), max_hp=30)
        p2 = Player(Deck(self.card_pool, []), max_hp=30)
        board = GameState(p1, p2)
        
        # Give Player 1 exactly 3 Mana and a hand of 3 cards (Cost 2, 3, 1)
        p1.mana = 3
        p1.hand = [self.card_pool["01"], self.card_pool["02"], self.card_pool["03"]]
        
        # Attempt to play all 3 cards (Action vector: [1, 1, 1])
        # Total cost is 6, which is > 3. The engine should fallback and reject the action entirely.
        board.play_turn([1, 1, 1])
        
        # Verify action was blocked
        self.assertEqual(p1.mana, 3) 
        self.assertEqual(len(p1.hand), 3)

if __name__ == '__main__':
    unittest.main(verbosity=2)
