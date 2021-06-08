import collections
import functools
import itertools
import math
import multiprocessing
import pathlib
import statistics
import time
import typing
import enum
import operator

import networkx
import numpy
import scipy.ndimage
import matplotlib.pyplot as plt

from stent_opt.odb_interface import datastore, db_defs
from stent_opt.abaqus_model import base
from stent_opt.struct_opt import design, display, score, history, optimisation_parameters

from stent_opt.struct_opt import patch_manager
from stent_opt.struct_opt import element_bucket
from stent_opt.struct_opt import construct_model
from stent_opt.struct_opt import computer
from stent_opt.struct_opt import common
from stent_opt.struct_opt import singular_filter


class Tail(enum.Enum):
    bottom = enum.auto()
    top = enum.auto()

    @property
    def action_is_adding_element(self) -> bool:
        if self == Tail.top:
            return True

        elif self == Tail.bottom:
            return False

        else:
            raise ValueError


MAKE_PLOTS = False

# TODO - hinge behaviour
#   - Stop it disconnecting bits


# Zero based node indexes from Figure 28.1.1–1 in the Abaqus manual


class RunOneArgs(typing.NamedTuple):
    working_dir: pathlib.Path
    optim_params: optimisation_parameters.OptimParams
    iter_this: int
    nodal_z_override_in_odb: float
    working_dir_extract: str
    model_infos: typing.List[patch_manager.SubModelInfoBase]
    patch_suffix: str
    child_patch_run_one_args: typing.Tuple["RunOneArgs"]
    executed_feedback_text: str
    do_run_model_TESTING: bool
    do_run_extraction_TESTING: bool

    @property
    def fn_inp(self) -> pathlib.Path:
        return history.make_fn_in_dir(self.working_dir, ".inp", self.iter_this, self.patch_suffix)

    @property
    def is_full_model(self) -> bool:
        return not bool(self.patch_suffix)

    def get_real_model_patch_to_elem(self) -> typing.Dict[int, typing.Set[int]]:
        """Get a dictionary of elem_id -> set(elem_ids), where elem_id is some element and the value in the dictionary
        is the set of all the elements which are mutually in each other's patches. All the element ids are in the
        full model form, not the sub-model +offset ones"""

        elem_to_set = collections.defaultdict(set)

        for patch_run_one_args in self.child_patch_run_one_args:
            for sub_model_info in patch_run_one_args.model_infos:

                for source_elem_num in sub_model_info.elem_nums: # this is always a patch_manager.SubModelInfo so elem_nums will be there.
                    attached_elems = [elem_num for elem_num in sub_model_info.elem_nums if source_elem_num != elem_num]
                    for target_elem_num in attached_elems:
                        elem_to_set[source_elem_num].add(target_elem_num)

        return elem_to_set


def get_gradient_input_data(
        optim_params: optimisation_parameters.OptimParams,
        iteration_nums: typing.Iterable[int],
) -> typing.Iterable[score.GradientInputData]:
    """raw_component is db_defs.ElementStress or db_defs.ElementPEEQ"""

    working_dir = pathlib.Path(optim_params.working_dir)
    history_db_fn = history.make_history_db(working_dir)

    with history.History(history_db_fn) as hist:
        stent_params = hist.get_stent_params()

        for iter_num in iteration_nums:
            output_db_fn = history.make_fn_in_dir(working_dir, ".db", iter_num)

            snapshot = hist.get_snapshot(iter_num)

            with datastore.Datastore(output_db_fn) as data:
                print(output_db_fn)
                one_frame = data.get_maybe_last_frame_of_instance("STENT-1")

                raw_rows = list(data.get_all_rows_at_frame(optim_params.region_gradient.component, one_frame))
                raw_ranking_components = list(score.get_primary_ranking_components(True, optim_params.single_primary_ranking_fitness_filter, optim_params, raw_rows))
                element_ranking_components = {one_comp.elem_id: one_comp for one_comp in raw_ranking_components}
                yield score.GradientInputData(
                    iteration_num=iter_num,
                    stent_design=design.make_design_from_snapshot(stent_params, snapshot),
                    element_ranking_components=element_ranking_components,
                )


T_index_to_val = typing.Dict[design.PolarIndex, float]
T_index_to_two_val = typing.Dict[design.PolarIndex, typing.Tuple[int, float, float]]
def gaussian_smooth(optim_params: optimisation_parameters.OptimParams, stent_params: design.StentParams, unsmoothed: T_index_to_val) -> T_index_to_val:

    if not optim_params.gaussian_sigma:
        # If it's None or 0.0, don't smooth anything.
        return unsmoothed.copy()

    # Make a 3D array

    design_space = stent_params.divs

    design_space_elements = design.node_to_elem_design_space(design_space)
    raw = numpy.zeros(shape=design_space_elements.to_tuple())
    for polar_index, val in unsmoothed.items():
        raw[polar_index.R, polar_index.Th, polar_index.Z] = val

    def normalised_sigma():
        """Since the sigma depends on the mesh size, we have to normalise it."""
        nominal_element_length = 0.05  # mm
        elem_lengths = [stent_params.single_element_r_span, stent_params.single_element_theta_span, stent_params.single_element_z_span]

        def make_sigmas():
            for l in elem_lengths:
                try:
                    yield optim_params.gaussian_sigma * nominal_element_length / l

                except ZeroDivisionError:
                    yield 0.0

        sigmas = list(make_sigmas())
        return sigmas

    # Gaussian smooth with wraparound in the Theta direction if required... 'nearest' just means use the closest
    theta_mode = 'wrap' if stent_params.wrap_around_theta else 'nearest'

    modes = ('nearest', theta_mode, 'nearest')
    sigmas = normalised_sigma()

    out_nd_array = scipy.ndimage.gaussian_filter(raw, sigma=sigmas, mode=modes)

    #step1 = scipy.ndimage.gaussian_filter1d(raw, sigma=optim_params.gaussian_sigma, axis=0, mode='nearest')
    #step2 = scipy.ndimage.gaussian_filter1d(step1, sigma=optim_params.gaussian_sigma, axis=1, mode=theta_mode)
    #step3 = scipy.ndimage.gaussian_filter1d(step2, sigma=optim_params.gaussian_sigma, axis=2, mode='nearest')

    # Put it back and return
    #out_nd_array = step3

    non_zero_inds = numpy.nonzero(out_nd_array)
    out_dict = {}
    for r, th, z in zip(*non_zero_inds):
        out_idx = design.PolarIndex(R=r, Th=th, Z=z)
        smoothed_val = float(out_nd_array[(r, th, z)])
        was_in_old_design = out_idx in unsmoothed
        is_nonzero = bool(smoothed_val)
        if was_in_old_design or is_nonzero:
            out_dict[out_idx] = smoothed_val

    return out_dict


def get_keys_of_tail(data: T_index_to_val, tail: Tail, percentile: float) -> typing.Set[design.PolarIndex]:
    """Gets the keys based on the percentiles in the dictionary."""

    if tail == Tail.bottom:
        true_percentile = percentile
        comp_op = operator.lt

    elif tail == Tail.top:
        true_percentile = 100.0 - percentile
        comp_op = operator.gt

    else:
        raise ValueError(tail)

    cutoff = numpy.percentile(sorted(data.values()), q=true_percentile)

    return {key for key, val in data.items() if comp_op(val, cutoff)}


def _evolve_decider_sorted_data(data: T_index_to_val, tail: Tail, n_elems: int):
    def sort_key(index_val):
        return index_val[1]

    if n_elems < 0:
        raise ValueError(n_elems)

    rev = True if tail == Tail.top else False
    sorted_data = sorted(data.items(), key=sort_key, reverse=rev)

    return sorted_data

def get_top_n_elements(data: T_index_to_val, tail: Tail, n_elems: int) -> typing.Set[design.PolarIndex]:
    """Gets the top however many elements"""

    sorted_data = _evolve_decider_sorted_data(data, tail, n_elems)
    top_n = {idx for idx, val in sorted_data[0:n_elems]}

    return top_n


def get_top_n_elements_maintaining_edge_connectivity(
        optim_params: optimisation_parameters.OptimParams,
        stent_params: design.StentParams,
        initial_active_elems: typing.Set[design.PolarIndex],
        data: T_index_to_val,
        tail: Tail,
        n_elems: int,
        elems_to_patch_buddies_idx: typing.Dict[design.PolarIndex, typing.Set[design.PolarIndex]],
        dirty_elems: typing.Set[design.PolarIndex]
) -> typing.Tuple[typing.Set[design.PolarIndex], typing.Set[design.PolarIndex]]:

    # Graph of connectivity if all elements are enabled, for stack connectivity stuff.
    all_in_elements = initial_active_elems.copy() | set(data.keys())
    graph_element_pool_all = {
        elemIdx: element_bucket.Element(num=elemIdx, conn=design.get_single_element_connection(stent_params, elemIdx))
        for elemIdx in all_in_elements}
    graph_all_in = element_bucket.build_graph(element_bucket.GraphEdgeEntity.fe_elem_edge, graph_element_pool_all.values())

    sorted_data = _evolve_decider_sorted_data(data, tail, n_elems)

    top_n = set()

    # Reverse the list so we can pop off the end and treat it as a stack...
    sorted_data.reverse()

    def element_pool_is_connected(elem_working_set: typing.Set[design.PolarIndex]):
        """Returns False if this bunch of elements would introduce a disconnection"""
        graph_element_pool = {
            element_bucket.Element(num=elemIdx, conn=design.get_single_element_connection(stent_params, elemIdx))
            for elemIdx in elem_working_set}

        graph = element_bucket.build_graph(element_bucket.GraphEdgeEntity.fe_elem_edge, graph_element_pool)
        num_components = networkx.number_connected_components(graph)

        return num_components == 1

    def element_pool_has_boundaries_attached(elem_working_set: typing.Set[design.PolarIndex]):
        """Returns False if this bunch of elements is not attached to the boundary (or in the interior)"""
        elems_at_each_slice = collections.Counter()
        left_side = set()
        right_side = set()
        for elem_idx in elem_working_set:
            elems_at_each_slice[elem_idx.Th] += 1

            # TODO - make this use the new stuff in stent_params
            bottom_bound_cutoff = stent_params.length * (0.5 - 0.5 * stent_params.end_connection_length_ratio)
            top_bound_cutoff = stent_params.length * (0.5 + 0.5 * stent_params.end_connection_length_ratio)
            is_in_middle = bottom_bound_cutoff <= elem_idx.Z * stent_params.single_element_z_span <= top_bound_cutoff
            if is_in_middle:
                if elem_idx.Th == 0:
                    left_side.add(elem_idx)

                elif elem_idx.Th+2 == stent_params.divs.Th:
                    right_side.add(elem_idx)

        has_all_thetas = len(elems_at_each_slice)+1 == stent_params.divs.Th
        return has_all_thetas and bool(left_side) and bool(right_side)

    def element_pool_is_OK(elem_working_set: typing.Set[design.PolarIndex]):
        if not element_pool_has_boundaries_attached(elem_working_set):
            return False

        is_connected = element_pool_is_connected(elem_working_set)
        if not is_connected:
            return False

        return True

    if not element_pool_is_OK(initial_active_elems):
        raise ValueError("Disconnection from the get go?")

    active_elems_working_set = set(initial_active_elems)

    def num_changed_elements():
        if tail.action_is_adding_element:
            return len(top_n - initial_active_elems)

        else:
            return len(top_n & initial_active_elems)

    # If we make a change to an element and it was attached to some other elements which were skipped, they can go back on the stack to be checked.
    # If they were skipped in the first place, they must be a higher priority so they can go on top.
    back_on_stack_after_change = collections.defaultdict(dict)

    halfway = len(sorted_data) // 2
    while num_changed_elements() < n_elems and len(sorted_data) > halfway:
        elemIdxCandidate, obj_fun_val = sorted_data.pop()

        elem_idx_is_clean = elemIdxCandidate not in dirty_elems
        if optim_params.one_elem_per_patch and not elem_idx_is_clean:
            print(f"  [Conn] {elemIdxCandidate} = {obj_fun_val} skipping... invalidated by a prior patch buddy being changed.")
            continue

        # Does this element change break some connectivity?
        trial_working_set = set(active_elems_working_set)
        if tail.action_is_adding_element:
            trial_working_set.add(elemIdxCandidate)

        else:
            trial_working_set.discard(elemIdxCandidate)

        DEBUG_verb = "add" if tail.action_is_adding_element else "remove"
        connected_to_candidate = list(graph_all_in.neighbors(graph_element_pool_all[elemIdxCandidate]))
        if element_pool_is_OK(trial_working_set):
            # Was OK - add to the real mesh
            top_n.add(elemIdxCandidate)
            if tail.action_is_adding_element:
                suffix_text = " (but was already in mesh)" if elemIdxCandidate in active_elems_working_set else ''
                active_elems_working_set.add(elemIdxCandidate)

            else:
                suffix_text = ' (but was not in the mesh???)' if elemIdxCandidate not in active_elems_working_set else ''
                active_elems_working_set.discard(elemIdxCandidate)

            # Set any patches this element was in to dirty. On submodels we won't get anything here...
            dirty_elem_idxs = elems_to_patch_buddies_idx.get(elemIdxCandidate, tuple())
            dirty_elems.add(elemIdxCandidate)
            for other_elem_idx in dirty_elem_idxs:
                dirty_elems.add(other_elem_idx)

            print(f"  [Conn] {elemIdxCandidate} = {obj_fun_val} OK to {DEBUG_verb}{suffix_text}")

            back_on_stack_elems = back_on_stack_after_change[elemIdxCandidate]
            # Put stuff back on the stack - do it in reverse order so the top ranked one ends up on top.
            for reactivated_idx, reactivated_value in reversed(_evolve_decider_sorted_data(back_on_stack_elems, tail, len(back_on_stack_elems))):
                print(f"  [Conn] reactivating {reactivated_idx}={reactivated_value}.")
                sorted_data.insert(0, reactivated_idx, reactivated_value)

        else:
            print(f"  [Conn] {elemIdxCandidate} = {obj_fun_val} not going to  {DEBUG_verb} {elemIdxCandidate} - would create disconnection.")
            for element_i_am_attached_to in connected_to_candidate:
                print(f"  [Conn] stacking {elemIdxCandidate} to be reactivated if {element_i_am_attached_to} changes state.")
                back_on_stack_after_change[element_i_am_attached_to][elemIdxCandidate] = obj_fun_val

    return top_n, dirty_elems


def display_design_flat(design_space: design.PolarIndex, data: T_index_to_val):
    # Flatten the R axis (through thickness)

    design_space_elements = design.node_to_elem_design_space(design_space)

    flat = numpy.zeros(shape=(design_space_elements.Th, design_space_elements.Z))
    for (_, th, z), val in data.items():
        flat[th, z] += val

    plt.imshow(flat)
    plt.show()


class CandidatesFor(typing.NamedTuple):
    removal_from_boundary: typing.FrozenSet[design.PolarIndex]
    existing_next_round: typing.FrozenSet[design.PolarIndex]  # Allowable design space for the next round (current elems and new ones)
    new_introduction: typing.FrozenSet[design.PolarIndex]  # Elements we could add but which weren't there last round.
    removal_from_everywhere: typing.FrozenSet[design.PolarIndex]  # If we don't limit ourselves to the boundary, these elements can go as well.


@functools.lru_cache()
def _get_all_admissible_elements(stent_params):
    fully_populated = design.generate_elem_indices_admissible(stent_params)
    return frozenset(elemIdx for _, elemIdx in fully_populated)

@functools.lru_cache()
def _get_all_elements(stent_params):
    fully_populated = design.generate_elem_indices(stent_params.divs)
    return frozenset(elemIdx for _, elemIdx in fully_populated)



def get_candidate_elements(
        optim_params: optimisation_parameters.OptimParams,
        design_n_min_1: design.StentDesign,
) -> CandidatesFor:

    """Figure out which elements can come and go for the next iteration"""

    fully_populated_indices = _get_all_admissible_elements(design_n_min_1.stent_params)
    fully_populated_indices_including_inadmissable = _get_all_elements(design_n_min_1.stent_params)

    def get_adj_elements(last_iter_elems, threshold: int):
        # Prepare the set of nodes to which eligible elements are attached.
        connections = (design.get_single_element_connection(design_n_min_1.stent_params, elemIdx) for elemIdx in last_iter_elems)
        nodes = (iNode for connection in connections for iNode in connection)
        nodes_on_last = collections.Counter(nodes)

        # Hacky fix - do not consider nodes which are entirely encased on a boundary interface.
        some_kind_of_limits_here = threshold > 0
        if some_kind_of_limits_here:
            fully_encased_node_connection = 2 if design_n_min_1.stent_params.stent_element_dimensions == 2 else 4
            #  boundary_nodes = design.generate_stent_boundary_nodes(design_n_min_1.stent_params)
            boundary_nodes = set()
            pretend_nodes_werent_there = {iNodeIdx for iNodeIdx, n_attached_elems in nodes_on_last.items() if iNodeIdx in boundary_nodes and n_attached_elems == fully_encased_node_connection}
            for iNodeIdx in pretend_nodes_werent_there:
                if iNodeIdx in nodes_on_last:
                    del nodes_on_last[iNodeIdx]

        working_set = set()
        for elemIdx in fully_populated_indices:
            this_elem_nodes = design.get_single_element_connection(design_n_min_1.stent_params, elemIdx)
            node_with_shared_interface = [iNode for iNode in this_elem_nodes if iNode in nodes_on_last]

            # Say, two or more nodes touching the old stent
            eligible = len(node_with_shared_interface) >= threshold
            if eligible:
                working_set.add(elemIdx)

        return frozenset(working_set)

    indicies_holes = fully_populated_indices_including_inadmissable - design_n_min_1.active_elements

    # Have to hackily add in the elements on the boundaries to the "removable" set...
    boundary_nodes_idxs = list(design.generate_stent_boundary_nodes(design_n_min_1.stent_params))
    boundary_nodes_nums = {design.node_from_index(design_n_min_1.stent_params.divs, node_idx.R, node_idx.Th, node_idx.Z) for node_idx in boundary_nodes_idxs}
    boundary_elements = set()
    for elemIdx in fully_populated_indices:
        this_elem_nodes = design.get_single_element_connection(design_n_min_1.stent_params, elemIdx)
        if any(n in boundary_nodes_nums for n in this_elem_nodes):
            boundary_elements.add(elemIdx)

    removal_partial = get_adj_elements(indicies_holes, optim_params.nodes_shared_with_old_design_to_contract)
    removal_and_bound = removal_partial | frozenset(boundary_elements)
    removal_from_boundary = removal_and_bound & design_n_min_1.active_elements

    existing_next_round = get_adj_elements(design_n_min_1.active_elements, optim_params.nodes_shared_with_old_design_to_expand)

    new_introduction = existing_next_round - design_n_min_1.active_elements

    return CandidatesFor(
        removal_from_boundary=removal_from_boundary,
        existing_next_round=existing_next_round,
        new_introduction=new_introduction,
        removal_from_everywhere=design_n_min_1.active_elements,
    )


def evolve_decider(optim_params: optimisation_parameters.OptimParams, design_n_min_1, sensitivity_result, run_one_args_completed: RunOneArgs) -> typing.Set[design.PolarIndex]:
    """Decides which elements are coming in and going out."""

    iter_this = run_one_args_completed.iter_this + 1
    target_count = optim_params.target_num_elems(design_n_min_1, iter_this)
    delta_n_elems = target_count - len(design_n_min_1.active_elements)

    candidates_for = get_candidate_elements(optim_params, design_n_min_1)

    # If there was no change in volume ratio, what would be the maximum number of elements we could add?
    max_new_num_unconstrained = int(optim_params.max_change_in_vol_ratio * design_n_min_1.stent_params.divs.fully_populated_elem_count())
    max_new_num = min(max_new_num_unconstrained, max_new_num_unconstrained+delta_n_elems)
    if max_new_num < 0:
        max_new_num = 0

    # Keep track of the element indices which haven't been touched by another addition or removal.
    clean_elems_to_patch_buddies_num = run_one_args_completed.get_real_model_patch_to_elem()
    elem_num_to_idx = {elem_id: elem_idx for elem_idx, elem_id, _ in design.generate_plate_elements_all(design_n_min_1.stent_params.divs, design_n_min_1.stent_params.stent_element_type)}

    clean_elems_to_patch_buddies_idx = dict()
    for clean_elem_num, other_clean_elem_nums in clean_elems_to_patch_buddies_num.items():
        clean_idxs = [elem_num_to_idx[other_elem_num] for other_elem_num in other_clean_elem_nums]
        clean_elems_to_patch_buddies_idx[elem_num_to_idx[clean_elem_num]] = clean_idxs

    # top_new_potential_elems_all = get_top_n_elements(sensitivity_result, Tail.top, max_new_num)
    dirty_elems = set()
    top_new_potential_elems_all, dirty_elems = get_top_n_elements_maintaining_edge_connectivity(optim_params, design_n_min_1.stent_params, design_n_min_1.active_elements, sensitivity_result, Tail.top, max_new_num, clean_elems_to_patch_buddies_idx, dirty_elems)
    top_new_potential_elems = {idx for idx in top_new_potential_elems_all if idx in candidates_for.existing_next_round}
    actual_new_elems = top_new_potential_elems - design_n_min_1.active_elements
    num_new = len(actual_new_elems)

    num_with_new_additions = len(design_n_min_1.active_elements) + num_new
    num_to_go = max(num_with_new_additions - target_count, 0)

    # Remove however many we added.
    if optim_params.patched_elements == common.PatchedElements.boundary:
        candidates_for_removal = candidates_for.removal_from_boundary

    elif optim_params.patched_elements == common.PatchedElements.all:
        candidates_for_removal = candidates_for.removal_from_everywhere

    else:
        raise ValueError(optim_params.patched_elements)

    existing_elems_only_ranked = {idx: val for idx,val in sensitivity_result.items() if idx in design_n_min_1.active_elements and idx in candidates_for_removal}
    # to_go = get_top_n_elements(existing_elems_only_ranked, Tail.bottom, num_to_go)
    elem_working_set = design_n_min_1.active_elements | top_new_potential_elems
    to_go, _ = get_top_n_elements_maintaining_edge_connectivity(optim_params, design_n_min_1.stent_params, elem_working_set, existing_elems_only_ranked, Tail.bottom, num_to_go, clean_elems_to_patch_buddies_idx, dirty_elems)

    new_active_elems = (design_n_min_1.active_elements | top_new_potential_elems) - to_go
    return new_active_elems


def _hist_status_stage_from_rank_comp(rank_comp: score.T_AnyRankingComponent) -> history.StatusCheckStage:
    if isinstance(rank_comp, score.PrimaryRankingComponent):
        return history.StatusCheckStage.primary

    elif isinstance(rank_comp, score.SecondaryRankingComponent):
        if "smoothed" in rank_comp.comp_name.lower():
            return history.StatusCheckStage.smoothed

        else:
            return history.StatusCheckStage.secondary

    elif isinstance(rank_comp, score.FilterRankingComponent):
        return history.StatusCheckStage.filtering

    raise TypeError(rank_comp)


class RankingResults(typing.NamedTuple):
    all_ranks: score.T_ListOfComponentLists
    final_ranking_component_unsmoothed: T_index_to_val
    final_ranking_component_smoothed: T_index_to_val
    filter_out_priority: T_index_to_val
    final_ranking_component_to_unsmoothed: typing.Dict[common.GlobalStatusType, typing.Tuple[str, T_index_to_val]]
    pos_rows: typing.List[db_defs.NodePos]

    def get_ordered_ranking_with_filter(self, max_to_filter_out: int) -> T_index_to_two_val:
        """Makes an "overall" ordering in which the elements to be filtered out are put with the lowest priority."""

        LEFT_ALONE, FILTERED_OUT = 1, -1

        overall_ranking: T_index_to_val = dict()

        # Step one - the values to be filtered out go into the overall list first. where we can't put all the values to be filter out in there, pick the highest filter_priority.
        first_ordering = itertools.chain( itertools.repeat(FILTERED_OUT, max_to_filter_out), itertools.repeat(LEFT_ALONE) )
        filter_out_priority_sorted = reversed(sorted((val, key) for key, val in self.filter_out_priority.items()))
        for should_filter, (filter_val, elem_idx) in zip(first_ordering, filter_out_priority_sorted):
            overall_ranking[elem_idx] = (should_filter, self.final_ranking_component_smoothed.get(elem_idx, 0.0))

        # Step two - anything not filtered out already
        for elem_idx, val in self.final_ranking_component_smoothed.items():
            if elem_idx not in overall_ranking:
                overall_ranking[elem_idx] = (LEFT_ALONE, val)

        return overall_ranking


def _combine_constraint_violations_rows(elem_num_to_indices, cons_viol_rows: typing.Iterable[score.FilterRankingComponent]) -> T_index_to_val:
    # Super simple - just add all the constraint violations together.

    all_cons_viols = collections.Counter()
    for filter_row in cons_viol_rows:
        all_cons_viols[filter_row.elem_id] += filter_row.value

    return {elem_num_to_indices[iElem]: val for iElem, val in all_cons_viols.items()}


T_filter_applicable_primary_components = typing.Union[db_defs.ElementFatigueResult, db_defs.ElementPEEQ]
def _get_raw_data_relevant_to_constraint_filter(data: datastore.Datastore, one_frame, maybe_only_these_elem_nums) -> typing.Iterable[T_filter_applicable_primary_components]:
    for elem_component in T_filter_applicable_primary_components.__args__:
        yield from data.get_all_rows_at_frame(elem_component, one_frame, only_these_elem_nums=maybe_only_these_elem_nums)


def _get_ranking_functions(
        run_one_args: RunOneArgs,
        model_info: patch_manager.SubModelInfoBase,
        iter_n: int,
        design_n_min_1: design.StentDesign,
        data: datastore.Datastore,
) -> RankingResults:
    """Gets the abaqus results out of the sqlite DB and returns the optimsation element-wise effort results."""

    all_ranks = []

    optim_params = run_one_args.optim_params

    # Gradient tracking
    if optim_params.region_gradient:
        final_grad_track = iter_n
        first_grad_track = max(0, final_grad_track - optim_params.region_gradient.n_past_increments)
        grad_track_steps = range(first_grad_track, final_grad_track)

        recent_gradient_input_data = get_gradient_input_data(optim_params, grad_track_steps)
        vicinity_ranking = list(score.get_primary_ranking_local_region_gradient(design_n_min_1.stent_params, recent_gradient_input_data, statistics.mean))

        # Sometimes there won't be data here (i.e., on the first few iterations).
        if vicinity_ranking:
            all_ranks.append(vicinity_ranking)

    one_frame = data.get_maybe_last_frame_of_instance("STENT-1")
    raw_elem_rows = []

    # Only extract the element results from this patch.
    maybe_only_these_elem_nums = model_info.all_patch_elem_ids_in_this_model()

    # Element based funtions
    elem_components_data = list(optim_params.get_all_elem_components(patch_model_context=model_info.is_sub_model))
    for include_in_opt_comp, elem_component in elem_components_data:
        for include_in_opt_filter, one_filter in optim_params.get_all_primary_ranking_fitness_filters():
            include_in_opt = include_in_opt_comp and include_in_opt_filter

            elem_results_this_submod = list(data.get_all_rows_at_frame(elem_component, one_frame, only_these_elem_nums=maybe_only_these_elem_nums))
            this_comp_rows = score.get_primary_ranking_components(include_in_opt, one_filter, optim_params, elem_results_this_submod)
            raw_elem_rows.append(list(this_comp_rows))

    # Nodal position based functions
    maybe_only_these_node_nums = model_info.all_patch_node_ids_in_this_model()
    pos_rows_submod = list(data.get_all_rows_at_frame(db_defs.NodePos, one_frame, only_these_node_nums=maybe_only_these_node_nums))
    pos_rows = [pos_row._replace(node_num=model_info.model_to_real_node(pos_row.node_num)) for pos_row in pos_rows_submod]
    for include_in_opt, one_func in optim_params.get_all_node_position_components(patch_model_context=model_info.is_sub_model):
        this_comp_rows = one_func(optim_params, include_in_opt, design_n_min_1, pos_rows)
        raw_elem_rows.append(list(this_comp_rows))

    # Map from submodel element ids to the main model
    for one_submodel_id_prim_result in raw_elem_rows:

        # Sometimes there will be nothing here (for example, a Fatigue result when there is no oscillating step).
        if one_submodel_id_prim_result:
            with_main_model_ids = [elem_row._replace(elem_id=model_info.model_to_real_elem(elem_row.elem_id)) for elem_row in one_submodel_id_prim_result]
            all_ranks.append(with_main_model_ids)

    # Smooth the final result
    elem_num_to_indices = {iElem: idx for iElem, idx in design.generate_elem_indices(design_n_min_1.stent_params.divs)}
    elem_indices_to_num = {idx: iElem for iElem, idx in design.generate_elem_indices(design_n_min_1.stent_params.divs)}

    # Compute the primary and overall ranking components

    # New thing - support sum for one thing and norm for something else
    multi_level_agg = {}
    for global_status_type, f_elem_result in optim_params.get_multi_level_aggregator().items():
        def is_matching_func_output(one_rank):
            return all(prc.comp_name.startswith(f_elem_result.__doc__) and isinstance(prc, score.PrimaryRankingComponent) for prc in one_rank)

        this_level_ranks = [one_rank for one_rank in all_ranks if is_matching_func_output(one_rank)]

        if len(this_level_ranks) != 1:
            raise ValueError("???")

        multi_level_agg[global_status_type] = f_elem_result.__doc__, {elem_num_to_indices[prc.elem_id]: prc.value for prc in this_level_ranks[0]}

    # Compute a secondary rank from all the first ones.
    sec_rank_unsmoothed = list(score.get_secondary_ranking_scaled_sum(all_ranks))
    all_ranks.append(sec_rank_unsmoothed)


    overall_rank = {one.elem_id: one.value for one in sec_rank_unsmoothed}
    unsmoothed = {elem_num_to_indices[iElem]: val for iElem, val in overall_rank.items()}
    smoothed = gaussian_smooth(optim_params, design_n_min_1.stent_params, unsmoothed)

    def append_additional_output(idx_to_val: typing.Dict[design.PolarIndex, float], comp_name: str):
        additional_ranking = [
            score.SecondaryRankingComponent(
                comp_name=comp_name,
                elem_id=elem_indices_to_num[elem_id],
                value=value,
                include_in_opt=True,
            ) for elem_id, value in idx_to_val.items()
        ]

        all_ranks.append(additional_ranking)

    # Save the smoothed form so it can go in the database.
    sec_rank_name = sec_rank_unsmoothed[0].comp_name + " Smoothed"
    append_additional_output(smoothed, sec_rank_name)

    # Constraint violation filters
    constraint_violations = []
    aba_db_rows_submod_ids = list(_get_raw_data_relevant_to_constraint_filter(data, one_frame, maybe_only_these_elem_nums))
    aba_db_rows_full_mod_ids = [elem_row._replace(elem_num=model_info.model_to_real_elem(elem_row.elem_num)) for elem_row in aba_db_rows_submod_ids]
    for include_in_opt_comp, one_filter_func in optim_params.get_all_filter_components():
        this_filter_rows = list(one_filter_func(include_in_opt_comp, aba_db_rows_full_mod_ids))
        constraint_violations.extend(this_filter_rows)
        if this_filter_rows:
            all_ranks.append(this_filter_rows)

    filter_out_priority = _combine_constraint_violations_rows(elem_num_to_indices, constraint_violations)

    return RankingResults(
        all_ranks=all_ranks,
        final_ranking_component_unsmoothed=unsmoothed,
        final_ranking_component_smoothed=smoothed,
        filter_out_priority=filter_out_priority,
        final_ranking_component_to_unsmoothed=multi_level_agg,
        pos_rows=pos_rows,
    )


T_PatchModelPair = typing.Tuple[patch_manager.SubModelInfo, patch_manager.SubModelInfo]
def prepare_patch_models(working_dir: pathlib.Path, optim_params: optimisation_parameters.OptimParams, iter_prev: int) -> typing.Iterable[T_PatchModelPair]:
    """Build the patch submodel info. They come in pairs so the "with" and "without" cases are solved in the same batch"""

    db_fn_prev = history.make_fn_in_dir(working_dir, ".db", iter_prev)
    history_db = history.make_history_db(working_dir)

    with history.History(history_db) as hist:
        stent_params = hist.get_stent_params()
        optim_params = hist.get_opt_params()

    if not optim_params.do_patch_analysis:
        return

    node_num_to_pos = {iNode: xyz for iNode, _, xyz in design.generate_nodes(stent_params)}

    elem_num_to_indices = {elem_num: polar_index for elem_num, polar_index in design.generate_elem_indices(stent_params.divs)}
    elem_indices_to_num = {polar_index: elem_num for elem_num, polar_index in elem_num_to_indices.items()}

    all_node_polar_index_admissible = stent_params.get_all_node_polar_indices_admissible()

    with history.History(history_db) as hist:
        snapshot_n_min_1 = hist.get_snapshot(iter_prev)
        design_n_min_1 = design.StentDesign(
            stent_params=stent_params,
            active_elements=frozenset( (elem_num_to_indices[iElem] for iElem in snapshot_n_min_1.active_elements)),
            label=snapshot_n_min_1.label,
        )

    candidates_for = get_candidate_elements(optim_params, design_n_min_1)

    # Elements to consider for the patch tests
    if optim_params.patched_elements == common.PatchedElements.boundary:
        init_state_and_elems = (
            (True, candidates_for.removal_from_boundary),
            (False, candidates_for.new_introduction),
        )

    elif optim_params.patched_elements == common.PatchedElements.all:
        init_state_and_elems = (
            (True, candidates_for.removal_from_everywhere),
            (False, candidates_for.new_introduction),
        )

    else:
        raise ValueError(optim_params.patched_elements)

    # Build the networkx graph so we can get the connectivity easily
    graph_fe_elems = {
        elem_num: element_bucket.Element(elem_num, design.get_single_element_connection(stent_params, polar_index))
        for elem_num, polar_index in elem_num_to_indices.items()
    }

    # Get the offsets for node and element numbers. If the max nodes are 2345, go in steps of 10000
    delta_offset = 10 ** (len(str(stent_params.divs.fully_populated_node_count())))
    offset_counter = itertools.count(delta_offset, delta_offset)

    graph = element_bucket.build_graph(element_bucket.GraphEdgeEntity.node, graph_fe_elems.values())

    with patch_manager.PatchManager(node_num_to_pos) as this_patch_manager:
        with datastore.Datastore(db_fn_prev) as data:
            this_patch_manager.ingest_node_pos(data.get_all_frames(), data.get_all_rows(db_defs.NodePos))

        for initial_active_state, elem_idxs in init_state_and_elems:
            for polar_index in sorted(elem_idxs):
                # Make a stent design with this element active (so we can support Off->On checks
                design_n_min_1_with_elem = design_n_min_1.with_additional_elements( [polar_index])
                reference_elem_num = elem_indices_to_num[polar_index]
                graph_elem = graph_fe_elems[reference_elem_num]

                boundary_node_nums, interior_fe_elems = element_bucket.get_fe_nodes_on_boundary_interface(graph, optim_params.patch_hops, graph_elem)
                elem_nums = frozenset(fe_elem.num for fe_elem in interior_fe_elems if elem_num_to_indices[fe_elem.num] in design_n_min_1_with_elem.active_elements)

                # Potentially different nodes active in the model if there are nodes only attached to the trial element.
                node_nums_this_trial_active = set()
                node_nums_this_trial_inactive = set()
                for fe_elem in interior_fe_elems:
                    elem_idx = elem_num_to_indices[fe_elem.num]
                    if elem_idx in design_n_min_1_with_elem.active_elements:
                        node_nums_this_trial_active.update(fe_elem.conn)
                        if fe_elem.num != reference_elem_num:
                            node_nums_this_trial_inactive.update(fe_elem.conn)

                node_nums_lookup = {
                    False: frozenset(node_nums_this_trial_inactive),
                    True: frozenset(node_nums_this_trial_active),
                }

                sub_model_pair = [
                    patch_manager.SubModelInfo(
                        boundary_node_nums=boundary_node_nums,
                        patch_manager=this_patch_manager,
                        stent_design=design_n_min_1_with_elem,
                        elem_nums=elem_nums,
                        node_nums=node_nums_lookup[this_trial_active_state],
                        reference_elem_num=reference_elem_num,
                        reference_elem_idx=polar_index,
                        initial_active_state=initial_active_state,
                        this_trial_active_state=this_trial_active_state,
                        node_elem_offset=next(offset_counter),
                        all_node_polar_index_admissible=all_node_polar_index_admissible,
                    )
                    for this_trial_active_state in (False, True)
                ]
                # print(f"reference_elem_num={reference_elem_num}")
                # print(f"   Off: {sub_model_pair[0]}")
                # print(f"   On:  {sub_model_pair[1]}")
                yield sub_model_pair

def _smi_pair_are_good(smi_pair):
    pair_are_ok = all(singular_filter.patch_matrix_OK(smi) for smi in smi_pair)
    return pair_are_ok, smi_pair


def produce_patch_models(working_dir: pathlib.Path, iter_prev: int) -> typing.Dict[
        str, typing.List[patch_manager.SubModelInfoBase]]:

    history_db = history.make_history_db(working_dir)
    with history.History(history_db) as hist:
        optim_params = hist.get_opt_params()

    suffix_to_patch_list = dict()

    sub_model_infos_all = list(prepare_patch_models(working_dir, optim_params, iter_prev))

    def sort_key(smi_pair):
        return smi_pair[0].reference_elem_num, smi_pair[0].this_trial_active_state

    sub_model_infos_all.sort(key=sort_key)

    # Filter out the ones which are heading for a singular matrix...
    t_before_patch = time.time()
    if optim_params.filter_singular_patches:
        if computer.this_computer.n_processes_unlicensed > 1:
            with multiprocessing.Pool(processes=computer.this_computer.n_processes_unlicensed) as pool:
                sub_model_infos = [smi_pair for is_ok, smi_pair in pool.imap_unordered(_smi_pair_are_good, sub_model_infos_all) if is_ok]

        else:
            sub_model_infos = []
            for smi_pair in sub_model_infos_all:
                is_ok, _ = _smi_pair_are_good(smi_pair)
                if is_ok:
                    sub_model_infos.append(smi_pair)
    else:
        sub_model_infos = list(sub_model_infos_all)

    # sub_model_infos = [smi_pair for smi_pair in sub_model_infos_all if all(singular_filter.patch_matrix_OK(smi) for smi in smi_pair)]
    print(f"Time to check singularity of {len(sub_model_infos_all)} patch pairs: {time.time() - t_before_patch} sec")

    # Batch them up into even-ish chunks
    chunk_size_ideal = len(sub_model_infos) / computer.this_computer.n_abaqus_parallel_solves
    chunk_size_round = math.ceil(chunk_size_ideal)

    # Don't need to divide them up too much (no point having four models with one patch in each...
    FLOOR_ELEMS_IN_MODEL = 1000
    floor_chunk_size = math.ceil(FLOOR_ELEMS_IN_MODEL / optim_params.nominal_number_of_patch_elements)
    chunk_size = max(floor_chunk_size, chunk_size_round)  # Don't need to make it too crazy tiny...
    print(f"  Chunk Size [Ideal / Round / Used]: {chunk_size_ideal} / {chunk_size_round} / {chunk_size}")
    #  chunk_size = 1 # TEMP!!!
    # https://stackoverflow.com/a/312464
    def get_prefix_and_smi():
        for sm_idx, i in enumerate(range(0, len(sub_model_infos), chunk_size), start=1):
            yield f'-P{sm_idx}', sub_model_infos[i:i + chunk_size]

    for suffix, sub_model_info_list in get_prefix_and_smi():
        # This will not occur if patch_hops is None.
        if sub_model_info_list:
            sub_fn_inp = history.make_fn_in_dir(working_dir, ".inp", iter_prev, suffix)

            # Flatten the with/without element pair.
            flat_sub_model_infos = [sub_mod_inf for sub_model_info_pair in sub_model_info_list for sub_mod_inf in sub_model_info_pair]

            # Make a dummy stent design with all the newly activated elements in it too.
            # TODO! - fix this! This is adding all the extra elements of all the patches! Need to do it on a sub-model by sub-model basis.
            stent_design_with_extras = flat_sub_model_infos[0].stent_design.with_additional_elements(fsmi.reference_elem_idx for fsmi in flat_sub_model_infos)
            construct_model.make_stent_model(optim_params, stent_design_with_extras, flat_sub_model_infos, sub_fn_inp)
            suffix_to_patch_list[suffix] = flat_sub_model_infos

    return suffix_to_patch_list


def _get_combined_global_objective_function(ranking_results: RankingResults):
    """e.g., Sum[Energy] + Norm8[Stress>350]..."""
    obj_funcs = []
    terms = []
    for comp_agg_func, (func_doc, elem_idx_to_val) in ranking_results.final_ranking_component_to_unsmoothed.items():
        obj_func = comp_agg_func.compute_aggregate(list(elem_idx_to_val.values()))
        obj_funcs.append(obj_func)
        term = f"{comp_agg_func.get_nice_name()}( {func_doc} )"
        terms.append(term)

    return ' + '.join(terms), sum(obj_funcs)


def _get_change_in_overall_objective_from_patches(
        model_info_to_patch_rank: typing.Dict[patch_manager.SubModelInfo, RankingResults],
        ) -> typing.Dict[int, float]:
    """This returns the "gradient" from the patches - basically, the change in objective function we can expect by activating the element."""

    def make_working_data():
        for model_info, ranking_results in model_info_to_patch_rank.items():

            ref_elem_num = model_info.reference_elem_num

            overall_term_name, obj_func_val = _get_combined_global_objective_function(ranking_results)
            yield ref_elem_num, model_info.this_trial_active_state, obj_func_val

    working_data = sorted(make_working_data())

    def get_elem_num(row):
        return row[0]

    elem_patch_deltas = dict()
    for elem_num, patch_rows in itertools.groupby(working_data, get_elem_num):
        activation_data = {active_state: obj_func for _, active_state, obj_func in patch_rows}
        gradient_from_patch = activation_data[False] - activation_data[True]
        elem_patch_deltas[elem_num] = gradient_from_patch

    return elem_patch_deltas


def process_completed_simulation(run_one_args: RunOneArgs
        ) -> typing.Tuple[design.StentDesign, typing.Dict[patch_manager.SubModelInfoBase, RankingResults]]:
    """Processes the results of a completed simulation and returns its design and results to produce the next iteration."""

    iter_prev = run_one_args.iter_this

    history_db = history.make_history_db(run_one_args.working_dir)
    with history.History(history_db) as hist:
        stent_params = hist.get_stent_params()
        optim_params = hist.get_opt_params()

    db_fn_prev = history.make_fn_in_dir(run_one_args.working_dir, ".db", iter_prev, run_one_args.patch_suffix)

    # Add any composite results which need to be added.
    with datastore.Datastore(db_fn_prev) as data:
        data._DEBUG_drop_all_idx()  # TODO - REMOVE!!!

        data.prepare_indices_for_extraction(db_defs.IndexCreateStage.primary_extract)
        one_frame = data.get_maybe_last_frame_of_instance("STENT-1")
        abaqus_created_primary_rows = list(data.get_all_rows_at_all_frames_any_element_type())
        composite_primary_rows_one = score.compute_composite_ranking_component_one(optim_params, abaqus_created_primary_rows)
        composite_primary_rows_two = score.compute_composite_ranking_component_two(optim_params, abaqus_created_primary_rows)
        data.add_results_on_existing_frame(one_frame, composite_primary_rows_one)
        data.add_results_on_existing_frame(one_frame, composite_primary_rows_two)
        data.prepare_indices_for_extraction(db_defs.IndexCreateStage.composite)

    # Go between num (1234) and idx (5, 6, 7)...
    elem_num_to_indices = {iElem: idx for iElem, idx in design.generate_elem_indices(stent_params.divs)}

    # Get any patch results and include them in the mix.
    model_info_to_patch_rank = {}
    for run_one_args_patch in run_one_args.child_patch_run_one_args:
        # The patch with and without a given element are going to be in the same sub-model.
        _, this_model_info_to_patch_rank = process_completed_simulation(run_one_args_patch)
        model_info_to_patch_rank.update(this_model_info_to_patch_rank)

    elem_patch_deltas = _get_change_in_overall_objective_from_patches(model_info_to_patch_rank)
    # Add in the patch results to the DB.
    with datastore.Datastore(db_fn_prev) as data:
        one_frame = data.get_maybe_last_frame_of_instance("STENT-1")

        in_db_form = [
            db_defs.ElementGlobalPatchSensitivity(frame_rowid=None, elem_num=elem_num, gradient_from_patch=gradient_from_patch)
            for elem_num, gradient_from_patch in elem_patch_deltas.items()]

        data.add_results_on_existing_frame(one_frame, in_db_form)
        data.prepare_indices_for_extraction(db_defs.IndexCreateStage.patch)

    with history.History(history_db) as hist:
        snapshot_n_min_1 = hist.get_snapshot(iter_prev)
        design_n_min_1 = design.StentDesign(
            stent_params=stent_params,
            active_elements=frozenset( (elem_num_to_indices[iElem] for iElem in snapshot_n_min_1.active_elements)),
            label=snapshot_n_min_1.label,
        )

    # Get the data from the previously run simulation.
    with datastore.Datastore(db_fn_prev) as data:
        ranking_results = {}
        for model_info in run_one_args.model_infos:
            ranking_result = _get_ranking_functions(run_one_args, model_info, iter_prev, design_n_min_1, data)
            global_status_raw = list(data.get_final_history_result())

            ranking_results[model_info] = ranking_result

            if run_one_args.is_full_model:
                applied_load_results = _get_sum_of_reaction_forces(data, iter_prev)
                _log_completed_in_history_db(history_db, ranking_result, global_status_raw, applied_load_results, optim_params, design_n_min_1, iter_prev)

    return design_n_min_1, ranking_results


def _get_sum_of_reaction_forces(data: datastore.Datastore, iteration_num) -> typing.Iterable[history.GlobalStatus]:
    rows: typing.List[db_defs.NodeReact] = list(data.get_all_rows(db_defs.NodeReact))

    def sort_key(row: db_defs.NodeReact):
        return row.frame_rowid

    rows.sort(key=sort_key)

    if not rows:
        return

    frame_rowids = sorted(set(row.frame_rowid for row in rows))

    last_rowid = frame_rowids[-1]
    second_last_rowid = frame_rowids[-2]

    def get_total_x_components():
        # Overall x reaction
        for row in rows:
            if row.frame_rowid == last_rowid:
                yield row.X

    def get_delta_components():
        # X stiffness
        contribs = collections.defaultdict(list)
        good_rows = [row for row in rows if row.frame_rowid in {last_rowid, second_last_rowid}]
        for row in good_rows:
            contribs[row.node_num].append(row.X)

        for node_num, last_two in contribs.items():
            yield last_two[1] - last_two[0]

    for global_name, global_val in (
            ("Reaction X", sum(get_total_x_components())),
            ("Final Frame Stiffness X", sum(get_delta_components())),
    ):
        yield history.GlobalStatus(iteration_num, history.GlobalStatusType.abaqus_history_result, global_name, global_val)


def _log_completed_in_history_db(history_db, ranking_result, global_status_raw, applied_load_results, optim_params, design_n_min_1, iter_prev):

    pos_lookup = {row.node_num: base.XYZ(x=row.X, y=row.Y, z=row.Z) for row in ranking_result.pos_rows}

    # Log the history
    with history.History(history_db) as hist:
        # Overall objective function.
        overall_term_name, obj_func_val = _get_combined_global_objective_function(ranking_result)
        global_status_check = history.GlobalStatus(
            iteration_num=iter_prev,
            global_status_type=history.GlobalStatusType.abaqus_history_result,
            global_status_sub_type=overall_term_name,
            global_status_value=obj_func_val,
        )
        hist.add_many_global_status_checks([global_status_check])


        status_checks = (history.StatusCheck(
            iteration_num=iter_prev,
            elem_num=rank_comp.elem_id,
            stage=_hist_status_stage_from_rank_comp(rank_comp),
            metric_name=rank_comp.comp_name,
            metric_val=rank_comp.value,
            constraint_violation_priority=rank_comp.constraint_violation_priority
        )
            for rank_comp_list in ranking_result.all_ranks for rank_comp in rank_comp_list
        )

        hist.add_many_status_checks(status_checks)

        # Save the global model results from Abaqus in the history file.
        global_status_history = [history.GlobalStatus(
            iteration_num=iter_prev,
            global_status_type=history.GlobalStatusType.abaqus_history_result,
            global_status_sub_type=stat_raw.history_identifier,
            global_status_value=stat_raw.history_value,
        ) for stat_raw in global_status_raw]
        hist.add_many_global_status_checks(global_status_history)

        hist.update_global_with_elemental(iteration_num=iter_prev)

        # Note the node positions for rendering later on.
        node_pos_for_hist = (history.NodePosition(
            iteration_num=iter_prev,
            node_num=node_num,
            x=pos.x,
            y=pos.y,
            z=pos.z) for node_num, pos in pos_lookup.items())
        hist.add_node_positions(node_pos_for_hist)

        # Reaction and stiffness
        hist.add_many_global_status_checks(applied_load_results)

        # Record the volume ratios
        vol_ratios = [
            history.GlobalStatus(
                iteration_num=iter_prev,
                global_status_type=history.GlobalStatusType.target_volume_ratio,
                global_status_sub_type="TargetVol",
                global_status_value=optim_params.volume_target_func(optim_params.volume_target_opts, iter_prev),
            ),
            history.GlobalStatus(
                iteration_num=iter_prev,
                global_status_type=history.GlobalStatusType.current_volume_ratio,
                global_status_sub_type="CurrentVol",
                global_status_value=design_n_min_1.volume_ratio(),
            ),
        ]
        hist.add_many_global_status_checks(vol_ratios)



T_ProdNewGen = typing.Callable[
    [pathlib.Path, design.StentDesign, RankingResults, int, str],
    design.StentDesign
]

def produce_new_generation(working_dir: pathlib.Path, design_prev: design.StentDesign, ranking_result: RankingResults, run_one_args_completed: RunOneArgs, label: str) -> design.StentDesign:

    iter_this = run_one_args_completed.iter_this + 1

    inp_fn = history.make_fn_in_dir(working_dir, ".inp", iter_this)
    history_db = history.make_history_db(working_dir)
    with history.History(history_db) as hist:
        stent_params = hist.get_stent_params()
        optim_params = hist.get_opt_params()

    # Max number of elements to change in an iteration limits the number of filter-based reorderings.
    delta_n_elems = int(optim_params.max_change_in_vol_ratio * design_prev.stent_params.divs.fully_populated_elem_count())
    sensitivity_ranking = ranking_result.get_ordered_ranking_with_filter(delta_n_elems)

    new_active_elems = evolve_decider(optim_params, design_prev, sensitivity_ranking, run_one_args_completed)

    elem_indices_to_num = {idx: iElem for iElem, idx in design.generate_elem_indices(stent_params.divs)}

    # Save the latest design snapshot
    # Log the history
    new_active_elem_nums = frozenset(elem_indices_to_num[idx] for idx in new_active_elems)
    with history.History(history_db) as hist:
        snapshot = history.Snapshot(
            iteration_num=iter_this,
            label=label,
            filename=str(inp_fn),
            active_elements=new_active_elem_nums)

        hist.add_snapshot(snapshot)

    return design.StentDesign(stent_params=design_prev.stent_params, active_elements=frozenset(new_active_elems), label=snapshot.label)


def make_plot_tests(working_dir: pathlib.Path, iter_n: int):

    history_db = history.make_history_db(working_dir)
    with history.History(history_db) as hist:
        stent_params = hist.get_stent_params()
        optim_params = hist.get_opt_params()


    first_iter = max(0, iter_n - optim_params.region_gradient.n_past_increments)
    history_iters = list(range(first_iter, iter_n+1))
    last_iter = history_iters[-1]

    all_ranks = []
    iter_to_pos = {}

    # Gradient tracking test
    recent_gradient_input_data = list(get_gradient_input_data(optim_params, history_iters))

    vicinity_ranking = list(score.get_primary_ranking_local_region_gradient(stent_params, recent_gradient_input_data, statistics.mean))
    for idx, x in enumerate(vicinity_ranking):
        print(idx, x, sep='\t')

    all_ranks.append(vicinity_ranking)

    for one_iter in history_iters:
        db_fn = history.make_fn_in_dir(working_dir, ".db", one_iter)
        with history.History(history_db) as hist:
            stent_params = hist.get_stent_params()
            old_snapshot = hist.get_snapshot(one_iter)
            old_design = design.make_design_from_snapshot(stent_params, old_snapshot)

        # stent_params = design.basic_stent_params
        # old_design = design.make_initial_design(stent_params)

        with datastore.Datastore(db_fn) as data:
            one_frame = data.get_maybe_last_frame_of_instance("STENT-1")

            peeq_rows = list(data.get_all_rows_at_frame(db_defs.ElementPEEQ, one_frame))
            stress_rows = list(data.get_all_rows_at_frame(db_defs.ElementStress, one_frame))
            pos_rows = list(data.get_all_rows_at_frame(db_defs.NodePos, one_frame))

        pos_lookup = {row.node_num: base.XYZ(x=row.X, y=row.Y, z=row.Z) for row in pos_rows}
        iter_to_pos[one_iter] = pos_lookup
        # Node position pased effort functions
        eff_funcs = [score.get_primary_ranking_macro_deformation, score.get_primary_ranking_element_distortion, ]

        for one_func in eff_funcs:
            # all_ranks.append(list(one_func(old_design, pos_rows)))
            pass

        # Element-based effort functions
        for db_data in [stress_rows]: # [peeq_rows]: #, stress_rows]:
            all_ranks.append(list(score.get_primary_ranking_components(True, optim_params.single_primary_ranking_fitness_filter, optim_params, db_data)))


    sec_rank = list(score.get_secondary_ranking_sum_of_norm(all_ranks))
    #all_ranks.append(sec_rank)

    for one_rank in all_ranks:
        display.render_status(old_design, pos_lookup, one_rank, "Testing")


def plot_history_gradient():
    working_dir = r"E:\Simulations\StentOpt\AA-178"

    quant = db_defs.ElementStress
    elems_of_interest = {30942, 40937}  # {28943, 42936, 30942, 40937}

    old_data = list(get_gradient_input_data(working_dir, quant, [0, 1]))

    for elem_id in elems_of_interest:
        x, y = [], []
        for grad_input_data in old_data:

            try:
                val = grad_input_data.element_ranking_components[elem_id].value
                x.append(grad_input_data.iteration_num)
                y.append(val)

            except KeyError:
                pass

        plt.plot(x, y, label=f"{elem_id}-{quant.__name__}")

    plt.legend()
    plt.show()


def _make_testing_run_one_args(working_dir, stent_design: design.StentDesign, optim_params=None, iter_this=0) -> RunOneArgs:

    from stent_opt.make_stent import run_model, working_dir_extract, make_full_model_list, process_pool_run_and_process

    from stent_opt.struct_opt.design import basic_stent_params as stent_params

    if not optim_params:
        from stent_opt.struct_opt.optimisation_parameters import active as optim_params

    return RunOneArgs(
        working_dir=working_dir,
        optim_params=optim_params,
        iter_this=iter_this,
        nodal_z_override_in_odb=stent_params.nodal_z_override_in_odb,
        working_dir_extract=working_dir_extract,
        model_infos=make_full_model_list(stent_design),
        patch_suffix='',
        child_patch_run_one_args=tuple(),
        executed_feedback_text='',
        do_run_model_TESTING=False,
        do_run_extraction_TESTING=False,
    )


def evolve_decider_test():
    from stent_opt.make_stent import process_pool_run_and_process

    iter_n_min_1 = 1
    iter_n = iter_n_min_1 + 1

    history_db = pathlib.Path(r"C:\Simulations\StentOpt\AA-109\history.db")

    with history.History(history_db) as hist:
        stent_params = hist.get_stent_params()
        optim_params = hist.get_opt_params()

        elem_num_to_indices = {iElem: idx for iElem, idx in design.generate_elem_indices(stent_params.divs)}

        snapshot_n_min_1 = hist.get_snapshot(iter_n_min_1)
        design_n_min_1 = design.StentDesign(
            stent_params=stent_params,
            active_elements=frozenset( (elem_num_to_indices[iElem] for iElem in snapshot_n_min_1.active_elements)),
            label=snapshot_n_min_1.label,
        )

    run_one_args_input = _make_testing_run_one_args(pathlib.Path(r"C:\Simulations\StentOpt\AA-109"), design_n_min_1, optim_params, iter_n_min_1)
    run_one_args_completed = process_pool_run_and_process(run_one_args_input)
    with history.History(history_db) as hist:

        status_checks = hist.get_status_checks(0, 1_000_000_000)
        smoothed_checks = [st for st in status_checks if st.stage == history.StatusCheckStage.smoothed]
        smoothed = {elem_num_to_indices[st.elem_num]: st.metric_val for st in smoothed_checks}


    new_active_elems = evolve_decider(optim_params, design_n_min_1, smoothed, run_one_args_completed)
    n_new = len(new_active_elems - design_n_min_1.active_elements)
    n_gone = len(design_n_min_1.active_elements - new_active_elems)

    print(f"Gone: {n_gone}\tNew: {n_new}")




def run_test_process_completed_simulation():

    from stent_opt.make_stent import process_pool_run_and_process

    # working_dir = pathlib.Path(r"E:\Simulations\StentOpt\AA-49")
    working_dir = pathlib.Path(r"C:\Simulations\StentOpt\AA-135")

    history_db = working_dir / "history.db"


    with history.History(history_db) as hist:
        old_design = hist.get_most_recent_design()

    testing_run_one_args_skeleton = _make_testing_run_one_args(working_dir, old_design, iter_this=0)

    testing_run_one_args_completed = process_pool_run_and_process(testing_run_one_args_skeleton)
    one_design, model_info_to_rank = process_completed_simulation(testing_run_one_args_completed)

    if len(model_info_to_rank) != 1:
        raise ValueError("What?")

    one_ranking = next(iter(model_info_to_rank.values()))

    one_new_design = produce_new_generation(working_dir, one_design, one_ranking, testing_run_one_args_completed, "TESTING")


if __name__ == '__main__':
    run_test_process_completed_simulation()

    # run_test_process_completed_simulation()


if False:


    for sub_inp in extra_inp_pool:
        print(sub_inp)
        run_model(active, sub_inp, force_single_core=True)

    new_design = produce_new_generation(working_dir, one_design, one_ranking, iter_this, "generation.produce_new_generation")
    # print(new_design)


