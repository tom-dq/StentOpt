import typing

from stent_opt.odb_interface import db_defs
from stent_opt.struct_opt import common, design, history


class VolumeTargetOpts(typing.NamedTuple):
    """Sets a target volume ratio, at a given iteration."""
    initial_ratio: float
    floor_ratio: float
    reduction_iters: int

    def to_db_strings(self):
        yield from history.nt_to_db_strings(self)

    @classmethod
    def from_db_strings(cls, data):
        return history.nt_from_db_strings(cls, data)

T_elem_result = typing.Union[db_defs.ElementStress, db_defs.ElementPEEQ]


class RegionGradient(typing.NamedTuple):
    """Parameters for the positive/negative influence of element activation or inactivation."""
    component: T_elem_result
    reduce_type: common.RegionReducer
    n_past_increments: int

    def to_db_strings(self):
        yield from history.nt_to_db_strings(self)

    @classmethod
    def from_db_strings(cls, data):
        return history.nt_from_db_strings(cls, data)


T_vol_func = typing.Callable[[VolumeTargetOpts, int], float]


class OptimParams(typing.NamedTuple):
    """Parameters which control the optimisation."""
    max_change_in_vol_ratio: float
    volume_target_opts: VolumeTargetOpts
    volume_target_func: T_vol_func
    region_gradient: typing.Optional[RegionGradient]  # None to not have the region gradient included.
    element_components: typing.List[T_elem_result]
    gaussian_sigma: float

    def _target_volume_ratio_clamped(self, stent_design: "design.StentDesign", iter_num: int) -> float:
        existing_volume_ratio = stent_design.volume_ratio()
        target_ratio_unclamped = self.volume_target_func(self.volume_target_opts, iter_num)
        clamped_target = _clamp_update(existing_volume_ratio, target_ratio_unclamped, self.max_change_in_vol_ratio)
        return clamped_target

    def target_num_elems(self, stent_design: "design.StentDesign", iter_num: int) -> int:
        fully_populated_elems = stent_design.stent_params.divs.fully_populated_elem_count()
        volume_ratio = self._target_volume_ratio_clamped(stent_design, iter_num)
        return int(fully_populated_elems * volume_ratio)

    def is_converged(self, previous_design: "design.StentDesign", this_design: "design.StentDesign", iter_num: int) -> bool:
        """Have we converged?"""
        target_stabilised = iter_num > self.volume_target_opts.reduction_iters + 1
        if not target_stabilised:
            return False

        same_design = previous_design == this_design
        if not same_design:
            return False

        vol_ratio_mismatch = this_design.volume_ratio() - self.volume_target_func(self.volume_target_opts, iter_num)
        within_range = abs(vol_ratio_mismatch) <= self.max_change_in_vol_ratio
        if not within_range:
            return False

        return True

    def to_db_strings(self):
        yield from history.nt_to_db_strings(self)

    @classmethod
    def from_db_strings(cls, data):
        return history.nt_from_db_strings(cls, data)

    def __eq__(self, other) -> bool:
        """Have to override this because "volume_target_func" can get a false negative (since the function can be declared
        and imported different times, I guess)."""
        if type(other) != type(self):
            return False

        return list(self.to_db_strings()) == list(other.to_db_strings())

# Functions which set a target volume ratio. These are idealised.
def vol_reduce_then_flat(vol_target_opts: VolumeTargetOpts, iter_num: int) -> float:

    delta = vol_target_opts.initial_ratio - vol_target_opts.floor_ratio
    reducing = delta * (vol_target_opts.reduction_iters - iter_num)/vol_target_opts.reduction_iters + vol_target_opts.floor_ratio
    target_ratio = max(vol_target_opts.floor_ratio, reducing)

    return target_ratio

_for_db_funcs = [
    vol_reduce_then_flat,
]

def _clamp_update(old, new, max_delta):
    """Go from old towards new, but by no more than max_delta"""

    if max_delta <= 0.0:
        raise ValueError(max_delta)

    full_diff = new-old

    if abs(full_diff) > max_delta:
        lower_bound = old - max_delta
        upper_bound = old + max_delta
        working_a = max(new, lower_bound)
        return min(working_a, upper_bound)

    else:
        return new


active = OptimParams(
    max_change_in_vol_ratio=0.0025,
    volume_target_opts=VolumeTargetOpts(
        initial_ratio=0.12,
        floor_ratio=0.05,
        reduction_iters=100,
    ),
    volume_target_func=vol_reduce_then_flat,
    region_gradient=RegionGradient(
        component=db_defs.ElementStress,  # TODO - make the code respect all these settings rather than whatever's littered around the place.
        reduce_type=common.RegionReducer.mean_val,
        n_past_increments=5,
    ),
    element_components=[
        db_defs.ElementPEEQ,
        db_defs.ElementStress,
    ],
    gaussian_sigma=2.0,
)


if __name__ == "__main__":
    #  check we can serialise and deserialse the optimisation parameters
    data = list(active.to_db_strings())
    for x in data:
        print(*x, sep='\t')

    again = OptimParams.from_db_strings(data)

    print(again)
    print(active)
    assert active == again



