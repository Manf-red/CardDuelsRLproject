from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.cards import Card, Deck
from env.engine import GameState, Player


class CardDuelsEnv(gym.Env):
    """
    Gymnasium environment for the Card Duels project.
    """
    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["human"]}
    np_random: np.random.Generator

    def __init__(self, card_pool: dict[str, Card], deck_list_p1: list[str], deck_list_p2: list[str], max_hand: int = 5, max_board: int = 7, max_hp: int = 30, mana_limit: int = 10, total_unique_cards: int = 40) -> None:
        super().__init__()
        self.card_pool = card_pool
        self.deck_list_p1 = deck_list_p1
        self.deck_list_p2 = deck_list_p2
        self.max_hand = max_hand
        self.max_board = max_board
        self.max_hp = max_hp
        self.mana_limit = mana_limit
        self.total_unique_cards = total_unique_cards
        self.card_dim = 7 # [Cost, atk, def, m_draw, m_burn, m_heal, m_wipe]
        self.max_deck = max(len(deck_list_p1), len(deck_list_p2))

        self.action_space = spaces.Discrete(2 ** self.max_hand)

        self.observation_space = spaces.Dict({
            "self_stats": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            "opp_stats": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "self_hand": spaces.Box(low=-1.0, high=1.0, shape=(self.max_hand, self.card_dim), dtype=np.float32),
            "self_board": spaces.Box(low=-1.0, high=1.0, shape=(self.max_board, self.card_dim), dtype=np.float32),
            "opp_hand": spaces.Box(low=-1.0, high=1.0, shape=(self.max_hand, self.card_dim), dtype=np.float32),
            "opp_board": spaces.Box(low=-1.0, high=1.0, shape=(self.max_board, self.card_dim), dtype=np.float32),
            "history_opp": spaces.Box(low=-1.0, high=1.0, shape=(self.max_deck, self.card_dim), dtype=np.float32),
            "history_self": spaces.Box(low=-1.0, high=1.0, shape=(self.max_deck, self.card_dim), dtype=np.float32),
            "deck_stats_self": spaces.Box(low=-1.0, high=1.0, shape=(self.max_deck, self.card_dim), dtype=np.float32)
        })

        self.board: GameState | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reinitializes the MDP to state t=0.
        """
        super().reset(seed=seed, options=options)

        deck1 = Deck(self.card_pool, self.deck_list_p1.copy(), self.np_random)
        deck2 = Deck(self.card_pool, self.deck_list_p2.copy(), self.np_random)

        p1 = Player(deck1, max_hp=self.max_hp, max_hand=self.max_hand, max_board=self.max_board, mana_limit=self.mana_limit)
        p2 = Player(deck2, max_hp=self.max_hp, max_hand=self.max_hand, max_board=self.max_board, mana_limit=self.mana_limit)
        
        # Draw H-1 cards so the active player has H cards starting his turn.
        for _ in range(self.max_hand - 1):
            p1.draw_card()
            p2.draw_card()

        self.board = GameState(p1, p2, self.np_random)
        self.board.start_turn()

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.board is None:
            raise RuntimeError("Environment must be reset before calling step().")

        current_agent_idx = self.board.active_p_idx
        binary_string = format(action, f'0{self.max_hand}b')
        action_vector = [int(bit) for bit in binary_string]

        is_game_over, winner_idx = self.board.play_turn(action=action_vector) # After this the active player in the board is changed

        if not is_game_over:
            is_game_over, winner_idx = self.board.start_turn()

        reward = 0.0

        if is_game_over:
            if winner_idx == current_agent_idx:
                reward = 1.0
            elif winner_idx == -1:
                reward = 0.0 # Mutual destruction
            else:
                reward = -1.0

        terminated = is_game_over
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> dict[str, np.ndarray]:
        if self.board is None:
            raise RuntimeError("Board is uninitialized")

        p_self = self.board.get_active_player()
        p_opp = self.board.get_opponent()

        # Sort statistics and informations
        p_self.hand.sort(key=lambda c: (c.cost, c.atk, c.defe, c.id))
        p_self.board.sort(key=lambda c: (c.defe, c.atk, c.cost, c.id), reverse=True)
        p_opp.board.sort(key=lambda c: (c.defe, c.atk, c.cost, c.id), reverse=True)
        p_opp.hand.sort(key=lambda c: (c.cost, c.atk, c.defe, c.id))

        #safe_obs_deck = sorted(p_self.deck.cards, key=lambda c: c.id)
        safe_obs_deck = p_self.deck.cards

        # Normalizations
        CARD_MAX = np.array([10.0, 10.0, 12.0, 3.0, 14.0, 10.0, 7.0], dtype=np.float32) # Cards: [Cost (10), ATK (10), DEF (12), Draw (3), Burn (14), Heal (10), Wipe (7)]
        STATS_MAX_SELF = np.array([self.max_hp, self.mana_limit, self.mana_limit, self.max_deck], dtype=np.float32)
        STATS_MAX_OPP = np.array([self.max_hp, self.max_hand, self.max_deck], dtype=np.float32)

        obs = {
            "self_stats": np.array([p_self.hp, p_self.mana, p_self.max_mana, len(p_self.deck.cards)], dtype=np.float32) / STATS_MAX_SELF,
            "opp_stats": np.array([p_opp.hp, len(p_opp.hand), len(p_opp.deck.cards)], dtype=np.float32) / STATS_MAX_OPP,
            "self_hand": build_card_matrix(p_self.hand, p_self.max_hand, self.card_dim) / CARD_MAX,
            "self_board": build_card_matrix(p_self.board, p_self.max_board, self.card_dim) / CARD_MAX,
            "opp_hand": build_card_matrix(p_opp.hand, p_opp.max_hand, self.card_dim) / CARD_MAX,
            "opp_board": build_card_matrix(p_opp.board, p_opp.max_board, self.card_dim) / CARD_MAX,
            "history_opp": build_card_matrix(p_opp.history, self.max_deck, self.card_dim) / CARD_MAX,
            "history_self": build_card_matrix(p_self.history, self.max_deck, self.card_dim) / CARD_MAX,
            "deck_stats_self": build_card_matrix(safe_obs_deck, self.max_deck, self.card_dim) / CARD_MAX
        }

        return obs

    def _get_info(self) -> dict[str, Any]:
        """
        Provides useful metadata for action masking or debugging.
        """
        if self.board is None:
            return{}

        p_active = self.board.get_active_player()
        total_actions = 2 ** self.max_hand

        valid_actions_mask = np.zeros(total_actions, dtype=np.int8)
        for i in range(total_actions):
            binary_string = format(i, f'0{self.max_hand}b')
            cost = 0
            is_valid = True

            for bit_idx, bit_str in enumerate(binary_string):
                if bit_str == '1':
                    if bit_idx < len(p_active.hand):
                        cost += p_active.hand[bit_idx].cost
                    else:
                        is_valid = False
                        break

            if is_valid and cost <= p_active.mana:
                valid_actions_mask[i] = 1 # The action is valid

        return {"action_mask": valid_actions_mask}

def build_card_matrix(card_list: list[Card], max_num_cards: int, card_dim: int) -> np.ndarray:
    card_matrix = np.zeros((max_num_cards, card_dim), dtype=np.float32)
    for i, card in enumerate(card_list):
        if i < max_num_cards:
            card_matrix[i] = card.to_vector()
    return card_matrix
