# Stuff that, for the life of me, I can't avoid circular importing...
import enum
import pydantic
import statistics


class RegionReducer(enum.Enum):
    max_val = enum.auto()
    mean_val = enum.auto()


class PrimaryRankingComponentFitnessFilter(enum.Enum):
    high_value = enum.auto()
    close_to_mean = enum.auto()
    close_to_median = enum.auto()
    p_norm_4 = enum.auto()

    @property
    def is_deviation_from_central_value(self) -> bool:
        if self in (PrimaryRankingComponentFitnessFilter.high_value, PrimaryRankingComponentFitnessFilter.p_norm_4):
            return False

        elif self in (PrimaryRankingComponentFitnessFilter.close_to_mean, PrimaryRankingComponentFitnessFilter.close_to_median):
            return True

        else:
            raise ValueError(self)

    def get_central_value_function(self):
        if self == PrimaryRankingComponentFitnessFilter.close_to_mean:
            return statistics.mean

        elif self == PrimaryRankingComponentFitnessFilter.close_to_median:
            return statistics.median

        else:
            raise ValueError(self)


class BaseModelForDB(pydantic.BaseModel):

    class Config:
        validate_assignment = True
        frozen = True

    def copy_with_updates(self, **kwargs):
        working_dict = self.dict()
        working_dict.update(kwargs)
        return self.validate(working_dict)
