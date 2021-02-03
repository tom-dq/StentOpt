import typing

from stent_opt.odb_interface import db_defs
from stent_opt.struct_opt import common, design, history, score

from stent_opt.abaqus_model import step

class VolumeTargetOpts(typing.NamedTuple):
    """Sets a target volume ratio, at a given iteration."""
    initial_ratio: float
    final_ratio: float
    num_iters: int

    def to_db_strings(self):
        yield from history.nt_to_db_strings(self)

    @classmethod
    def from_db_strings(cls, data):
        return history.nt_from_db_strings(cls, data)


T_elem_result = typing.Union[db_defs.ElementStress, db_defs.ElementPEEQ, db_defs.ElementEnergyElastic, db_defs.ElementEnergyPlastic]


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
T_nodal_pos_func = typing.Callable[["design.StentDesign", typing.Iterable[db_defs.NodePos]], typing.Iterable[score.PrimaryRankingComponent]]   # Accepts a design and some node positions, and generates ranking components.


class OptimParams(typing.NamedTuple):
    """Parameters which control the optimisation."""
    max_change_in_vol_ratio: float
    volume_target_opts: VolumeTargetOpts
    volume_target_func: T_vol_func
    region_gradient: typing.Optional[RegionGradient]  # None to not have the region gradient included.
    element_components: typing.List[T_elem_result]
    nodal_position_components: typing.List[T_nodal_pos_func]
    gaussian_sigma: float
    working_dir: str
    use_double_precision: bool
    abaqus_output_time_interval: float
    abaqus_target_increment: float
    release_stent_after_expansion: bool
    analysis_step_type: typing.Type[step.StepBase]
    nodes_shared_with_old_design_to_expand: int    # Only let new elements come in which are attached to existing elements with at least this many nodes. Zero to allow all elements.
    nodes_shared_with_old_design_to_contract: int  # Only let new elements go out which are attached to existing elements with at least this many nodes. Zero to allow all elements.

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
        target_stabilised = iter_num > self.volume_target_opts.num_iters + 1
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

    @property
    def is_explicit(self) -> bool:
        return step.is_explicit(self.analysis_step_type)


# Functions which set a target volume ratio. These are idealised.
def vol_reduce_then_flat(vol_target_opts: VolumeTargetOpts, iter_num: int) -> float:

    delta = vol_target_opts.initial_ratio - vol_target_opts.final_ratio
    reducing = delta * (vol_target_opts.num_iters - iter_num) / vol_target_opts.num_iters + vol_target_opts.final_ratio
    target_ratio = max(vol_target_opts.final_ratio, reducing)

    return target_ratio


# Add all the functions from score which might be referenced
def _get_for_db_funcs():
    """These are the functions which can be serialised and deserialised when going into and out of History.db"""
    working_list = [
        vol_reduce_then_flat,
    ]

    all_names = dir(score)
    good_names = [n for n in all_names if n.startswith("get_primary_ranking")]
    for maybe_f_name in good_names:
        maybe_f = getattr(score, maybe_f_name)
        if not callable(maybe_f):
            raise ValueError(f"Got shouldn't score.{maybe_f_name} be callable?")

        working_list.append(maybe_f)

    return working_list

_for_db_funcs = _get_for_db_funcs()

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
    max_change_in_vol_ratio=0.0025,  # Was 0.0025
    volume_target_opts=VolumeTargetOpts(
        initial_ratio=0.004,
        final_ratio=0.01,
        num_iters=50,
    ),
    volume_target_func=vol_reduce_then_flat,
    region_gradient=RegionGradient(
        component=db_defs.ElementEnergyElastic,  # TODO - make the code respect all these settings rather than whatever's littered around the place.
        reduce_type=common.RegionReducer.mean_val,
        n_past_increments=5,
    ),
    element_components=[
        # db_defs.ElementPEEQ,
        # db_defs.ElementStress,
        db_defs.ElementEnergyElastic,
        # db_defs.ElementEnergyPlastic,
    ],
    nodal_position_components=[
        # score.get_primary_ranking_element_distortion,
        # score.get_primary_ranking_macro_deformation,
    ],
    gaussian_sigma=0.15,  # TODO - make this proporional to the element length or something? Was 0.75
    working_dir=r"c:\temp\ABCDE",
    use_double_precision=False,
    abaqus_output_time_interval=0.1,
    abaqus_target_increment=1e-6,
    release_stent_after_expansion=False,
    analysis_step_type=step.StepDynamicExplicit,
    nodes_shared_with_old_design_to_expand=2,
    nodes_shared_with_old_design_to_contract=2,
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



