from dataclasses import dataclass


@dataclass
class Card:
    suit: str
    rank: int


ace_spades = Card(suit="S", rank=1)
