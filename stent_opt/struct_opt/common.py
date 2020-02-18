# Stuff that, for the life of me, I can't avoid circular importing...
import enum


class RegionReducer(enum.Enum):
    max_val = enum.auto()
    mean_val = enum.auto()

