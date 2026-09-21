import json

import numpy as np


class Card:
    """
    Represents a single card in the game.
    """
    def __init__(self, data: dict):
        self.id: str = data["id"]
        self.name: str = data["name"]
        self.cost: int = data["cost"]
        self.atk: int = data["atk"]
        self.defe: int = data["def"]
        self.m_draw: int = data["m_draw"]
        self.m_burn: int = data["m_burn"]
        self.m_heal: int = data["m_heal"]
        self.m_wipe: int = data["m_wipe"]

        # Pre-compute the vector once in memory
        self._cached_vector = np.array([
            self.cost, self.atk, self.defe, self.m_draw, 
            self.m_burn, self.m_heal, self.m_wipe
        ], dtype=np.float32)

    def to_vector(self) -> np.ndarray:
        """
        Returns the dense vector representation of the card for the neural network.
        Shape: (7,)
        Format: [Cost, ATK, DEF, m_draw, m_burn, m_heal, m_wipe]
        """
        return self._cached_vector

    def __repr__(self):
        return f"<Card {self.id}: {self.name}>"

class Deck:
    """
    Manages the deck for a player, handling drawing, shuffling, and deck statistics.
    """
    def __init__(self, card_pool: dict[str, Card], card_in_deck_ids: list[str], rng: np.random.Generator) -> None:
        self.card_pool = card_pool
        self.card_in_deck_ids = card_in_deck_ids
        self.rng = rng
        self.cards: list[Card] = []
        self.reset()

    def reset(self):
        """
        Rebuilds the deck and shuffles it. Called at the start of every episode (env.reset()).
        """
        self.cards = [self.card_pool[card_id] for card_id in self.card_in_deck_ids]
        self.rng.shuffle(self.cards)
        #self.cards.sort(key=lambda card: card.id)

    def draw(self) -> Card | None:
        """
        Pops a card from the deck without replacement.
        Returns None if the deck is empty.
        """
        if not self.cards:
            return None
        return self.cards.pop()

    def is_empty(self) -> bool:
        """
        Checks if the deck is empty.
        """
        return len(self.cards) == 0

def load_card_pool(filepath: str) -> dict[str, Card]:
    """
    Utility function to parse the JSON file and create the lookup dictionary.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    card_pool: dict[str, Card] = {}
    for card_data in data:
        card_pool[card_data["id"]] = Card(card_data)

    return card_pool
