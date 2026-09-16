import numpy as np


class HeuristicOpponent:
    """
    State-Conditioned Utility Agent with Temperature scaling.
    """
    def __init__(self, max_hp: int = 30, max_hand: int = 5, max_board: int = 7, temperature: float = 1.0):
        self.max_hand = max_hand
        self.max_board = max_board
        self.max_hp = max_hp
        self.n_actions = 2 ** max_hand
        self.temperature = temperature

    def _evaluate_card(self, card_vector: np.ndarray, obs: dict[str, np.ndarray], available_slots: int) -> float:
        # Card: [Cost, ATK, DEF, m_draw, m_burn, m_heal, m_wipe]
        atk = card_vector[1] * 10.0
        defe = card_vector[2] * 12.0
        m_draw = round(card_vector[3] * 3.0)
        m_burn = card_vector[4] * 14.0
        m_heal = card_vector[5] * 10.0
        m_wipe = round(card_vector[6] * 7.0)
        
        opp_hp = obs["opp_stats"][0] * 30.0
        self_hp = obs["self_stats"][0] * 30.0
        self_deck_size = obs["self_stats"][3] * 20.0
        opp_board_raw = obs["opp_board"] * np.array([10., 10., 12., 3., 14., 10., 7.], dtype=np.float32)
        
        # Board Space Constraints
        if defe > 0 and available_slots <= 0:
            # The unit goes straight to the graveyard. We severely penalize wasting the stats.
            utility = -(atk * 1.5 + defe)
        else:
            utility = (atk * 1.5) + defe
        
        # Burn & Self-Burn Logic
        if m_burn > 0:
            if m_burn >= opp_hp:
                return 9999.0  # Instant lethal
            utility += m_burn * 10
        elif m_burn < 0:
            utility += m_burn * 10  # Penalize self-inflicted damage
            
        # Heal & Self-Damage Logic
        if m_heal > 0:
            deficit = 30.0 - self_hp # Assuming Max HP is 30
            actual_heal = min(m_heal, deficit)
            utility += actual_heal * 5
        elif m_heal < 0:
            utility += m_heal * 10  # Penalize self-inflicted damage
            if self_hp + m_heal <= 0:
                return -9999.0  # Prevent accidental suicide
            
        # Draw Logic (Fatigue Protection)
        if m_draw > 0:
            if self_deck_size <= m_draw:
                utility -= 1000.0
            else:
                utility += m_draw * 15
                
        # Wipe Logic
        if m_wipe > 0:
            opp_active_cards = np.sum(opp_board_raw[:, 2] > 0)
            if opp_active_cards >= m_wipe:
                utility += 100.0
            else:
                utility -= 500.0
                
        return utility

    def select_action(self, obs: dict, action_mask: np.ndarray) -> int:
        valid_actions = np.flatnonzero(action_mask == 1)
        
        if len(valid_actions) == 1 and valid_actions[0] == 0:
            return 0
            
        utilities = np.zeros(len(valid_actions), dtype=np.float32)
        hand_matrix = obs["self_hand"]
        
        # Calculate available board slots dynamically
        current_board_size = np.sum(obs["self_board"][:, 2] > 0)
        base_available_slots = self.max_board - current_board_size
        
        for idx, action_idx in enumerate(valid_actions):
            binary_string = format(action_idx, f'0{self.max_hand}b')
            action_utility = 0.0
            slots_used = 0
            
            for bit_idx, bitStr in enumerate(binary_string):
                if bitStr == '1':
                    card_vec = hand_matrix[bit_idx]
                    
                    # Project remaining slots to avoid evaluating multi-unit plays on a nearly full board
                    slots_remaining = base_available_slots - slots_used
                    action_utility += self._evaluate_card(card_vec, obs, slots_remaining)
                    
                    if card_vec[2] > 0:  # If DEF > 0, it occupies a board slot
                        slots_used += 1
                        
            utilities[idx] = action_utility

        if self.temperature <= 0.01:
            best_idx = np.argmax(utilities)
            return int(valid_actions[best_idx])
            
        # Restored Softmax Normalization Block
        scaled_u = utilities / self.temperature
        scaled_u -= np.max(scaled_u)
        exp_u = np.exp(scaled_u)
        probs = exp_u / np.sum(exp_u)
        
        chosen_idx = np.random.choice(len(valid_actions), p=probs)
        return int(valid_actions[chosen_idx])

    def end_episode(self, ending_rate: float) -> None:
        """
        Gradually reduce randomness as training progresses.
        """
        self.temperature = max(0.01, 2.0 * (1.0 - ending_rate))  # Curriculum Learning: Gradually reduce randomness as training progresses
