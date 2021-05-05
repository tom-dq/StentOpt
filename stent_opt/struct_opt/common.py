# Stuff that, for the life of me, I can't avoid circular importing...
import enum
import pydantic


class RegionReducer(enum.Enum):
    max_val = enum.auto()
    mean_val = enum.auto()


class PrimaryRankingComponentFitnessFilter(enum.Enum):
    high_value = enum.auto()
    close_to_mean = enum.auto()
    close_to_median = enum.auto()


class BaseModelForDB(pydantic.BaseModel):

    class Config:
        validate_assignment = True
        frozen = True

    def copy_with_updates(self, **kwargs):
        working_dict = self.dict()
        working_dict.update(kwargs)
        return self.validate(working_dict)
