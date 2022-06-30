import enum
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


T_elem_result = typing.Union[
    db_defs.ElementStress,
    db_defs.ElementPEEQ,
    db_defs.ElementEnergyElastic,
    db_defs.ElementEnergyPlastic,
    db_defs.ElementFatigueResult,
    db_defs.ElementGlobalPatchSensitivity,
    db_defs.ElementCustomCompositeOne,
    db_defs.ElementCustomCompositeTwo,
    db_defs.ElementNodeForces,
]


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


class PostExpansionBehaviour(enum.Enum):
    none = enum.auto()
    release = enum.auto()
    oscillate = enum.auto()

    @property
    def requires_second_step(self) -> bool:
        return self in (PostExpansionBehaviour.release, PostExpansionBehaviour.oscillate)

T_vol_func = typing.Callable[[VolumeTargetOpts, int], float]
T_nodal_pos_func = typing.Callable[["OptimParams", bool, "design.StentDesign", typing.Iterable[db_defs.NodePos]], typing.Iterable[score.PrimaryRankingComponent]]   # Accepts a design and some node positions, and generates ranking components.
T_filter_component = typing.Callable[[bool, typing.Iterable[score.PrimaryRankingComponent]], typing.Iterable[score.FilterRankingComponent]]

# TODO - figure out this signature...
T_composite_primary_calculator = typing.Callable[[typing.Any,], typing.Iterable[score.PrimaryRankingComponent]]

class OptimParams(typing.NamedTuple):
    """Parameters which control the optimisation."""
    max_change_in_vol_ratio: float
    volume_target_opts: VolumeTargetOpts
    volume_target_func: T_vol_func
    region_gradient: typing.Optional[RegionGradient]  # None to not have the region gradient included.
    filter_components: typing.List[T_filter_component]
    primary_ranking_fitness_filters: typing.List[common.PrimaryRankingComponentFitnessFilter]
    element_components: typing.List[T_elem_result]
    nodal_position_components: typing.List[T_nodal_pos_func]
    primary_composite_calculator_one: T_composite_primary_calculator
    primary_composite_calculator_two: T_composite_primary_calculator
    final_target_measure_one: common.GlobalStatusType
    final_target_measure_two: typing.Optional[common.GlobalStatusType]
    gaussian_sigma: typing.Optional[float]
    local_deformation_stencil_length: float
    working_dir: str
    use_double_precision: bool
    abaqus_output_time_interval: float
    abaqus_target_increment: float
    time_expansion: float
    time_released: typing.Optional[float]  # Make this None to not have a "release" step.
    post_expansion_behaviour: PostExpansionBehaviour
    analysis_step_type: typing.Type[step.StepBase]
    nodes_shared_with_old_design_to_expand: int    # Only let new elements come in which are attached to existing elements with at least this many nodes. Zero to allow all elements.
    nodes_shared_with_old_design_to_contract: int  # Only let new elements go out which are attached to existing elements with at least this many nodes. Zero to allow all elements.
    patch_hops: typing.Optional[int]  # 0 means patch is just the element in question. 1 means a maximum 3x3 patch, 2 means maximum 5x5, etc. None means no patches
    nonlinear_geometry: bool
    nonlinear_material: bool
    patched_elements: common.PatchedElements
    one_elem_per_patch: bool
    filter_singular_patches: bool

    def get_multi_level_aggregator(self) -> typing.Dict[common.GlobalStatusType, T_elem_result]:
        working_dict = {self.final_target_measure_one: self.primary_composite_calculator_one}
        if self.final_target_measure_two:
            working_dict[self.final_target_measure_two] = self.primary_composite_calculator_two

        return working_dict

    @property
    def add_initial_node_pos(self) -> bool:
        return True

    @property
    def release_stent_after_expansion(self) -> bool:
        return bool(self.time_released)

    @property
    def do_patch_analysis(self) -> bool:
        if self.patch_hops == None:
            return False

        return True

    @property
    def elem_span_for_patch_buffered(self) -> int:
        if self.patch_hops:
            return 4 * self.patch_hops

        else:
            return 1

    @property
    def nominal_number_of_patch_elements(self) -> int:
        if self.patch_hops is None:
            return 1

        half_a_square = (2*self.patch_hops + 1)**2 // 2
        return max(1, int(half_a_square))

    @property
    def simulation_has_second_step(self) -> bool:
        return bool(self.time_released) and self.post_expansion_behaviour.requires_second_step

    def should_override_with_only_ElementGlobalPatchSensitivity(self, patch_model_context: bool) -> bool:
        """
            Doing Patch Model?        Yes                                 No
            Full Model Context:       [ElementGlobalPatchSensitivity]     element_components
            Patch Model Context:      element_components                  N/A
        """

        patch_analysis_for_ec = self.do_patch_analysis and patch_model_context
        non_patch_for_ec = not self.do_patch_analysis and not patch_model_context
        patch_analysis_just_patch_sens = self.do_patch_analysis and not patch_model_context

        if patch_analysis_for_ec or non_patch_for_ec:
            return False

        elif patch_analysis_just_patch_sens:
            return True

        else:
            raise ValueError()

    def get_all_elem_components(self, patch_model_context: bool) -> typing.Iterable[typing.Tuple[bool, T_elem_result]]:
        """Step through the results, selecting the ones which are contribution to the optimisation"""

        if self.should_override_with_only_ElementGlobalPatchSensitivity(patch_model_context):
            include_in_this = [db_defs.ElementGlobalPatchSensitivity]

        else:
            include_in_this = self.element_components

        for elem_component in T_elem_result.__args__:
            include_in_opt = elem_component in include_in_this
            yield include_in_opt, elem_component


    def get_all_node_position_components(self, patch_model_context: bool) -> typing.Iterable[typing.Tuple[bool, T_nodal_pos_func]]:

        if self.should_override_with_only_ElementGlobalPatchSensitivity(patch_model_context):
            include_in_this = set()  # Nothing from here!

        else:
            include_in_this = [
                score.get_primary_ranking_element_distortion,
                score.get_primary_ranking_macro_deformation,
            ]

            # TEMP - don't include anything!
            include_in_this = set()

        for node_component in include_in_this:
            include_in_opt = node_component in self.nodal_position_components
            yield include_in_opt, node_component

    def get_all_filter_components(self) -> typing.Iterable[typing.Tuple[bool, T_filter_component]]:
        all_defined_funcs = [
            # score.constraint_filter_within_fatigue_life,
            # score.constraint_filter_not_yielded_out,
        ]

        for filter_component in all_defined_funcs:
            include_in_opt = filter_component in self.filter_components
            yield include_in_opt, filter_component

    def get_all_primary_ranking_fitness_filters(self) -> typing.Iterable[typing.Tuple[bool, common.PrimaryRankingComponentFitnessFilter]]:
        for one_filter in common.PrimaryRankingComponentFitnessFilter.get_components_to_even_consider():
            include_in_opt = one_filter in self.primary_ranking_fitness_filters
            yield include_in_opt, one_filter

    @property
    def single_primary_ranking_fitness_filter(self) -> common.PrimaryRankingComponentFitnessFilter:
        if len(self.primary_ranking_fitness_filters) != 1:
            raise ValueError(self.primary_ranking_fitness_filters)

        return self.primary_ranking_fitness_filters[0]

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

        same_design = previous_design.active_elements == this_design.active_elements
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

    def get_abaqus_history_time_interval(self) -> float:
        if self.analysis_step_type == step.StepStatic:
            # Don't record the full history it's too slow!
            return self.time_expansion

        else:
            return self.abaqus_output_time_interval


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
    good_names = [n for n in all_names if n.startswith("get_primary_ranking") or n.startswith("constraint_filter_") or n.startswith('primary_composite_')]
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


volume_ratio = VolumeTargetOpts(
    initial_ratio=0.5,
    final_ratio=0.195,  # Simon target
    num_iters=25,
)

volume_ratio_v2 = VolumeTargetOpts(
    initial_ratio=0.1,
    final_ratio=0.3,
    num_iters=50,
)


active = OptimParams(
    # TODO - next time I make changes to this, migrate it over to pydantic first.
    max_change_in_vol_ratio=0.03,  # Was 0.0025
    volume_target_opts=volume_ratio,
    volume_target_func=vol_reduce_then_flat,
    region_gradient=RegionGradient(
        component=db_defs.ElementEnergyElastic,  # TODO - make the code respect all these settings rather than whatever's littered around the place.
        reduce_type=common.RegionReducer.mean_val,
        n_past_increments=5,
    ),
    filter_components=[
        # score.constraint_filter_not_yielded_out,
        # score.constraint_filter_within_fatigue_life,
    ],
    primary_ranking_fitness_filters=[common.PrimaryRankingComponentFitnessFilter.high_value],
    element_components=[
        # db_defs.ElementPEEQ,
        # db_defs.ElementStress,
        # db_defs.ElementEnergyElastic,
        # db_defs.ElementEnergyPlastic,
        # db_defs.ElementFatigueResult,
        db_defs.ElementCustomCompositeOne,
        db_defs.ElementCustomCompositeTwo,
    ],
    primary_composite_calculator_one=score.primary_composite_energy_neg,
    primary_composite_calculator_two=score.primary_composite_stress_over_crit,
    nodal_position_components=[
        # score.get_primary_ranking_element_distortion,
        # score.get_primary_ranking_macro_deformation,
    ],
    final_target_measure_one=history.GlobalStatusType.aggregate_mean,  # Since we're working with the energy density, the mean is the average density of the part.
    final_target_measure_two=None, #history.GlobalStatusType.aggregate_p_norm_8,
    gaussian_sigma=0.15,  # Was 0.3
    local_deformation_stencil_length=0.1,
    working_dir=r"c:\temp\ABCDE",
    use_double_precision=False,
    abaqus_output_time_interval=0.02,  # Was 0.1
    abaqus_target_increment=1e-6,  # 1e-6
    time_expansion=0.5,  # Was 0.5 then 2.0, 0.2 seems OK as well
    time_released=None,
    post_expansion_behaviour=PostExpansionBehaviour.oscillate,
    analysis_step_type=step.StepDynamicExplicit,
    nodes_shared_with_old_design_to_expand=2,
    nodes_shared_with_old_design_to_contract=2,
    patch_hops=2,
    nonlinear_geometry=True,
    nonlinear_material=True,
    patched_elements=common.PatchedElements.boundary,
    one_elem_per_patch=False,
    filter_singular_patches=False,
)

active = active._replace(region_gradient=None)


if __name__ == "__main__":
    #  check we can serialise and deserialse the optimisation parameters
    data = list(active.to_db_strings())
    for x in data:
        print(*x, sep='\t')

    again = OptimParams.from_db_strings(data)

    print(again)
    print(active)
    assert active == again



