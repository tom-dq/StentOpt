# Stuff that, for the life of me, I can't avoid circular importing...
import enum
import pydantic
import statistics
import typing

class RegionReducer(enum.Enum):
    max_val = enum.auto()
    mean_val = enum.auto()


class PrimaryRankingComponentFitnessFilter(enum.Enum):
    high_value = enum.auto()
    close_to_mean = enum.auto()
    close_to_median = enum.auto()

    @staticmethod
    def get_components_to_even_consider() -> typing.Iterable["PrimaryRankingComponentFitnessFilter"]:
        # Don't worry about close_to_mean or close_to_median - seem like a dead end.
        return (PrimaryRankingComponentFitnessFilter.high_value,)

    @property
    def is_deviation_from_central_value(self) -> bool:
        if self in (PrimaryRankingComponentFitnessFilter.high_value,):
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


class GlobalStatusType(enum.Enum):
    abaqus_history_result = enum.auto()
    current_volume_ratio = enum.auto()
    target_volume_ratio = enum.auto()
    aggregate_min = enum.auto()
    aggregate_mean = enum.auto()
    aggregate_median = enum.auto()
    aggregate_max = enum.auto()
    aggregate_p_norm_4 = enum.auto()
    aggregate_p_norm_8 = enum.auto()
    aggregate_p_norm_12 = enum.auto()
    aggregate_sum = enum.auto()

    @classmethod
    def get_elemental_aggregate_values(cls):
        for name, enum_obj in cls.__members__.items():
            if name.startswith("aggregate_"):
                yield enum_obj

    def get_nice_name(self) -> str:
        agg_start = "aggregate_"
        if not self.name.startswith(agg_start):
            raise ValueError

        is_p_norm, maybe_p_norm_val = self.get_p_norm_status()
        if is_p_norm:
            return f'Norm{maybe_p_norm_val}'

        return self.name[len(agg_start):].title()

    def compute_aggregate(self, elemental_vals: typing.List[float]) -> float:

        is_p_norm, maybe_p_norm_val = self.get_p_norm_status()

        if is_p_norm:
            # Note - we use average here so the result isn't number-of-element dependent.
            vals_powered = [x**maybe_p_norm_val for x in elemental_vals]
            ave = statistics.mean(vals_powered)
            return ave ** (1/maybe_p_norm_val)

        elif self == GlobalStatusType.aggregate_min:
            return min(elemental_vals)

        elif self == GlobalStatusType.aggregate_max:
            return max(elemental_vals)

        elif self == GlobalStatusType.aggregate_mean:
            return statistics.mean(elemental_vals)

        elif self == GlobalStatusType.aggregate_median:
            return statistics.median(elemental_vals)

        elif self == GlobalStatusType.aggregate_sum:
            return sum(elemental_vals)

        raise ValueError(f"Did not know what to return for {self}")


    def get_p_norm_status(self) -> typing.Union[
                typing.Tuple[typing.Literal[False], typing.Literal[None]],
                typing.Tuple[typing.Literal[True], int]]:

        START_TEXT = "aggregate_p_norm_"
        if self.name.startswith(START_TEXT):
            p_val = int(self.name[len(START_TEXT):])
            return True, p_val

        else:
            return False, None


class PatchedElements(enum.Enum):
    boundary = enum.auto()
    all = enum.auto()
