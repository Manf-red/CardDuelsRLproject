import random

from env.cards import Card, Deck


class EngineError(Exception):
    """Base exception class for Card Duels engine violations."""

class ManaLimitExceededError(EngineError):
    """Raised when an action attempts to spend more mana than available."""

class InvalidHandIndexError(EngineError):
    """Raised when an action bit attempts to play a card index that does not exist."""

class InvalidBoardSpaceError(EngineError):
    """Raised when attempting to deploy a unit without available board slots."""

class Player:
    """
    Mantains the state variables and memory buffers for a single player.
    """
    def __init__(self, deck: Deck, max_hp: int = 30, max_hand: int = 5, max_board: int = 7, mana_limit: int = 10):
        self.max_hp = max_hp
        self.hp = max_hp
        self.max_mana = 0
        self.mana_limit = mana_limit
        self.mana = 0
        self.fatigue_counter = 0
        
        self.deck: Deck = deck
        self.hand: list[Card] = []
        self.board: list[Card] = []
        self.history: list[Card] = [] # The graveyard for destroyed/played cards
        
        self.max_hand = max_hand
        self.max_board = max_board

    def draw_card(self):
        """
        Executees the draw mechanic. If the deck is empty, applies the Fatigue Damage rule.
        """
        card = self.deck.draw()
        if card is None:
            self.fatigue_counter += 1
            self.hp -= self.fatigue_counter
        elif len(self.hand) < self.max_hand:
            self.hand.append(card)
        else:
            # Hand is full, the card is sent to graveyard
            self.history.append(card)

class GameState:
    """
    The core rules engine for the Turn and the Combat resolution mechanic.
    """
    def __init__(self, p1: Player, p2: Player):
        self.players = [p1, p2]
        self.active_p_idx = 0

    def get_active_player(self) -> Player:
        return self.players[self.active_p_idx]

    def get_opponent(self) -> Player:
        return self.players[1 - self.active_p_idx]

    def start_turn(self) -> tuple[bool, int]:
        """
        Restore phase
        """
        p = self.get_active_player()
        if p.max_mana < p.mana_limit:
            p.max_mana += 1

        p.mana = p.max_mana

        p.draw_card()

        return self._game_over_check()

    def play_turn(self, action: list[int]) -> tuple[bool, int]:
        """
        Executes the action phase and resolution phase.
        Return a tuple: (is_game_over, winner_index)
        """
        attacker = self.get_active_player()
        defender = self.get_opponent()
        deployed_cards: list[Card] = []
        total_cost = 0

        # Action Phase
        for i in range(len(action) - 1, -1, -1):
            if action[i] == 1:
                if i >= len(attacker.hand):
                    raise InvalidHandIndexError(f"Action tried to play card at index {i}, but hand size is {len(attacker.hand)}.")
                total_cost += attacker.hand[i].cost

        if total_cost > attacker.mana:
            raise ManaLimitExceededError(f"Action requires {total_cost} mana, but player only has {attacker.mana} mana.")

        attacker.mana -= total_cost

        for i in range(len(action) - 1, -1, -1):
            if action[i] == 1:
                deployed_cards.append(attacker.hand.pop(i))

        # Resolution Phase
        self._resolve_combat(attacker, defender, deployed_cards)

        # Terminal state check
        is_game_over, winner_index = self._game_over_check()

        # Swap active player
        self.active_p_idx = 1 - self.active_p_idx

        return is_game_over, winner_index
        

    def _game_over_check(self) -> tuple[bool, int]:
        active_p = self.get_active_player()
        opponent_p = self.get_opponent()

        if opponent_p.hp <= 0 and active_p.hp <= 0:
            return True, -1
        elif opponent_p.hp <= 0:
            return True, self.active_p_idx
        elif active_p.hp <= 0:
            return True, 1 - self.active_p_idx

        return False, -1

    def _resolve_combat(self,  attacker: Player, defender: Player, deployed_cards: list[Card]):
        """
        Calculates effects, ATKvs DEF, and state updates.
        """
        atk_tot = 0

        for card in deployed_cards:
            atk_tot += card.atk

            # Board wipe
            if card.m_wipe > 0:
                num_to_destroy = min(card.m_wipe, len(defender.board))
                if num_to_destroy > 0:
                    indices_to_destroy = sorted(random.sample(range(len(defender.board)), num_to_destroy), reverse=True)
                    for idx in indices_to_destroy:
                        destroyed = defender.board.pop(idx)
                        defender.history.append(destroyed)
            
            # Burn 
            if card.m_burn != 0:
                defender.hp -= card.m_burn

            # Heal
            if card.m_heal != 0:
                attacker.hp = min(attacker.max_hp, attacker.hp + card.m_heal)

            # Draw
            if card.m_draw > 0:
                for _ in range(card.m_draw):
                    attacker.draw_card()

        for b_card in attacker.board:
            atk_tot += b_card.atk

        # Sort the defenders cards ("Highest DEF First" rule)
        defender.board.sort(key=lambda c: (c.defe, c.atk, c.cost, c.id), reverse=True)

        # Sequential subtraction
        surviving_defenders = []
        for def_card in defender.board:
            if atk_tot >= def_card.defe:
                atk_tot -= def_card.defe
                defender.history.append(def_card)
            else:
                atk_tot = 0
                surviving_defenders.append(def_card)

        defender.board = surviving_defenders

        if atk_tot > 0:
            defender.hp -= atk_tot

        # Board deployment
        deployed_cards.sort(key=lambda c: c.defe, reverse=True)
        for card in deployed_cards:
            if card.defe > 0: # Persistent card (creature)
                if len(attacker.board) < attacker.max_board:
                    attacker.board.append(card)
                else:
                    attacker.history.append(card)
            else:
                attacker.history.append(card)


