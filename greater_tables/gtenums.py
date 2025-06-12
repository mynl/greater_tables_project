"""
Defines greater_table enumerations.

Defines Breakability enumeration.

Lines should never be broken, be flagged as a date, may be broken, or are acceptable
for breaks (text).

"""
from enum import IntEnum, Enum


class Breakability(IntEnum):
    """To track if a column should or should not be broken (wrapped)."""

    NEVER = 0
    DATE = 3
    MAYBE = 5
    ACCEPTABLE = 10


class  Alignment(Enum):
    """Left, right, center horizontal alignment."""

    LEFT = 'l'
    CENTER = 'c'
    RIGHT = 'r'
