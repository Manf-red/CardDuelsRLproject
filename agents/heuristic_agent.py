import numpy as np


class HeuristicOpponent:
    """
    State-Conditioned Utility Agent with Temperature scaling.
    """
    def __init__(self, max_hp: int = 40, max_hand: int = 5, max_board: int = 7, temperature: float = 1.0, seed: int | None = None):
        self.max_hand = max_hand
        self.max_board = max_board
        self.max_hp = max_hp
        self.n_actions = 2 ** max_hand
        self.initial_temperature = temperature
        self.temperature = temperature

        self.rng = np.random.default_rng(seed)

    def _evaluate_card_base(self, card_vector: np.ndarray, obs: dict[str, np.ndarray], available_slots: int) -> float:
        """Evaluates non-terminal board control, draw, and wipe mechanics."""
        atk = card_vector[1]
        defe = card_vector[2]
        m_draw = card_vector[3]
        m_wipe = card_vector[6]
        m_burn = card_vector[4]
        m_heal = card_vector[5]
        
        self_deck_size = obs.get("self_stats", np.zeros(4, dtype=np.float32))[3] * 20.0
        opp_board_raw = obs.get("opp_board", np.zeros((self.max_board, 7), dtype=np.float32)) * np.array([10., 10., 12., 3., 14., 10., 7.], dtype=np.float32)
        
        utility = 0.0
        
        # Board Space Constraints
        if defe > 0 and available_slots <= 0:   # If the card has defense but no available slots, it cannot be played effectively
            utility -= (atk * 1.5 + defe)
        else:
            utility += (atk * 1.5) + defe   # Else the card contributes positively to utility based on its attack and defense values
            
        # Draw (Fatigue Protection)
        if m_draw > 0:
            if self_deck_size <= m_draw:
                utility -= 1000.0
            else:
                utility += m_draw * 15

        # Wipe
        if m_wipe > 0:
            opp_active_cards = np.sum(opp_board_raw[:, 2] > 0)
            if opp_active_cards >= m_wipe:
                utility += 100.0
            else:
                utility -= 200.0

        # Damage
        utility += m_burn * 10

        # Healing
        if m_heal > 0:
            utility += m_heal * 5
        elif m_heal < 0:
            utility -= m_heal * 10
        
        return utility

    def select_action(self, obs: dict, action_mask: np.ndarray) -> int:
        valid_actions = np.flatnonzero(action_mask == 1)
        
        if len(valid_actions) == 1 and valid_actions[0] == 0:
            return 0
            
        utilities = np.zeros(len(valid_actions), dtype=np.float32)
        hand_matrix = obs.get("self_hand", np.zeros((self.max_hand, 7), dtype=np.float32))
        
        # Un-normalize using the correct self.max_hp dynamic parameter
        opp_hp_base = obs.get("opp_stats", np.zeros(3, dtype=np.float32))[0] * self.max_hp
        self_hp_base = obs.get("self_stats", np.zeros(4, dtype=np.float32))[0] * self.max_hp
        
        # Calculate available board slots dynamically
        self_board = obs.get("self_board", np.zeros((self.max_board, 7), dtype=np.float32))
        current_board_size = np.sum(self_board[:, 2] > 0)
        base_available_slots = self.max_board - current_board_size # Slots available before selecting any actions
        
        for idx, action_idx in enumerate(valid_actions):
            binary_string = format(action_idx, f'0{self.max_hand}b')
            action_utility = 0.0
            slots_used = 0
            tot_atk = np.sum(self_board[:, 1] * 10.0)  # Un-normalize current board attack
            tot_m_burn = 0.0
            tot_m_heal = 0.0

            # For each card in the action:
            for bit_idx, bitStr in enumerate(binary_string):
                if bitStr == '1':
                    card_vec = hand_matrix[bit_idx] * np.array([10., 10., 12., 3., 14., 10., 7.], dtype=np.float32) # Unnormalize card vector
                    slots_remaining = base_available_slots - slots_used # Dynamic slots remaining after accounting for previously selected cards
                    
                    # Add base utility
                    action_utility += self._evaluate_card_base(card_vec, obs, slots_remaining)
                    
                    # Accumulate HP impacts
                    tot_m_burn += card_vec[4]
                    tot_m_heal += card_vec[5]
                    tot_atk += card_vec[1]
                    
                    if card_vec[2] > 0:  
                        slots_used += 1

            # --- Evaluate Aggregate Combo States ---
            tot_opp_board_defe = np.sum(obs.get("opp_board", np.zeros((self.max_board, 7), dtype=np.float32))[:, 2] * 12.0)

            sim_self_hp = self_hp_base + tot_m_heal
            sim_opp_hp = opp_hp_base - tot_m_burn

            if tot_opp_board_defe < tot_atk:
                atk_damage = tot_atk - tot_opp_board_defe
                sim_opp_hp -= atk_damage

            # Combo Lethal Check
            if sim_opp_hp <= 0 and opp_hp_base > 0:
                action_utility += 9999.0
                
            # Combo Suicide Check
            if sim_self_hp <= 0 and self_hp_base > 0:
                action_utility -= 9999.0
                
            # Remove utility awarded for healing past Max HP
            if sim_self_hp > self.max_hp:
                excess_heal = sim_self_hp - self.max_hp
                action_utility -= (excess_heal * 5)

            utilities[idx] = action_utility

        if self.temperature <= 0.01:
            best_indices = np.flatnonzero(utilities == np.max(utilities))
            best_idx = self.rng.choice(best_indices)
            return int(valid_actions[best_idx])

        # Softmax with temperature scaling
        scaled_u = utilities / self.temperature
        scaled_u -= np.max(scaled_u)
        exp_u = np.exp(scaled_u)
        probs = exp_u / np.sum(exp_u)
        
        chosen_idx = self.rng.choice(len(valid_actions), p=probs)
        return int(valid_actions[chosen_idx])

    def end_episode(self, ending_rate: float) -> None:
        """
        Hyperbolic decay: Fast initial randomness drop to quickly provide a stationary benchmark.
        ending_rate: Normalized training progress from 0.0 to 1.0
        """
        if self.initial_temperature <= 0.01:
            return

        t_min = 0.01
        
        # Dynamically calculate alpha using the actual initialization temperature
        alpha = (self.initial_temperature / t_min) - 1.0
        
        self.temperature = max(t_min, self.initial_temperature / (1.0 + alpha * ending_rate))