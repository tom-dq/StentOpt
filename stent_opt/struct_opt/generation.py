import collections
import pathlib
import statistics
import typing
import enum
import operator


import numpy
import scipy.ndimage
import matplotlib.pyplot as plt

from stent_opt.odb_interface import datastore, db_defs
from stent_opt.abaqus_model import base
from stent_opt.struct_opt import design, display, score, history, optimisation_parameters


class Tail(enum.Enum):
    bottom = enum.auto()
    top = enum.auto()


MAKE_PLOTS = False

# TODO - hinge behaviour
#   - Stop it disconnecting bits


# Zero based node indexes from Figure 28.1.1–1 in the Abaqus manual


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
                raw_ranking_components = list(score.get_primary_ranking_components(True, optim_params.single_primary_ranking_fitness_filter, raw_rows))
                element_ranking_components = {one_comp.elem_id: one_comp for one_comp in raw_ranking_components}
                yield score.GradientInputData(
                    iteration_num=iter_num,
                    stent_design=design.make_design_from_snapshot(stent_params, snapshot),
                    element_ranking_components=element_ranking_components,
                )


T_index_to_val = typing.Dict[design.PolarIndex, float]
def gaussian_smooth(optim_params: optimisation_parameters.OptimParams, stent_params: design.StentParams, unsmoothed: T_index_to_val) -> T_index_to_val:
    # Make a 3D array

    design_space = stent_params.divs

    design_space_elements = design.node_to_elem_design_space(design_space)
    raw = numpy.zeros(shape=design_space_elements.to_tuple())
    for (r, th, z), val in unsmoothed.items():
        raw[r, th, z] = val

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


def get_top_n_elements(data: T_index_to_val, tail: Tail, n_elems: int) -> typing.Set[design.PolarIndex]:
    """Gets the top however many elements"""

    def sort_key(index_val):
        return index_val[1]

    if n_elems < 0:
        raise ValueError(n_elems)

    rev = True if tail == Tail.top else False
    sorted_data = sorted(data.items(), key=sort_key, reverse=rev)
    top_n = {idx for idx, val in sorted_data[0:n_elems]}
    return top_n


def display_design_flat(design_space: design.PolarIndex, data: T_index_to_val):
    # Flatten the R axis (through thickness)

    design_space_elements = design.node_to_elem_design_space(design_space)

    flat = numpy.zeros(shape=(design_space_elements.Th, design_space_elements.Z))
    for (_, th, z), val in data.items():
        flat[th, z] += val

    plt.imshow(flat)
    plt.show()


class CandidatesFor(typing.NamedTuple):
    removal: typing.FrozenSet[design.PolarIndex]
    introduction: typing.FrozenSet[design.PolarIndex]


def get_candidate_elements(
        optim_params: optimisation_parameters.OptimParams,
        design_n_min_1: design.StentDesign,
) -> CandidatesFor:

    """Figure out which elements can come and go for the next iteration"""

    fully_populated = design.generate_elem_indices(design_n_min_1.stent_params.divs)
    fully_populated_indices = frozenset(elemIdx for _, elemIdx in fully_populated)

    def get_adj_elements(last_iter_elems, threshold: int):
        # Prepare the set of nodes to which eligible elements are attached.
        connections = (design.get_single_element_connection(design_n_min_1.stent_params, elemIdx) for elemIdx in last_iter_elems)
        nodes = (iNode for connection in connections for iNode in connection)
        nodes_on_last = collections.Counter(nodes)

        # Hacky fix - do not consider nodes which are entirely encased on a boundary interface.
        some_kind_of_limits_here = threshold > 0
        if some_kind_of_limits_here:
            fully_encased_node_connection = 2 if design_n_min_1.stent_params.stent_element_dimensions == 2 else 4
            boundary_nodes = design.generate_stent_boundary_nodes(design_n_min_1.stent_params)
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

    indicies_holes = fully_populated_indices - design_n_min_1.active_elements

    return CandidatesFor(
        removal=get_adj_elements(indicies_holes, optim_params.nodes_shared_with_old_design_to_contract),
        introduction=get_adj_elements(design_n_min_1.active_elements, optim_params.nodes_shared_with_old_design_to_expand),
    )


def evolve_decider(optim_params: optimisation_parameters.OptimParams, design_n_min_1, sensitivity_result, iter_n) -> typing.Set[design.PolarIndex]:
    """Decides which elements are coming in and going out."""

    target_count = optim_params.target_num_elems(design_n_min_1, iter_n)
    delta_n_elems = target_count - len(design_n_min_1.active_elements)

    candidates_for = get_candidate_elements(optim_params, design_n_min_1)

    # If there was no change in volume ratio, what would be the maximum number of elements we could add?
    max_new_num_unconstrained = int(optim_params.max_change_in_vol_ratio * design_n_min_1.stent_params.divs.fully_populated_elem_count())
    max_new_num = min(max_new_num_unconstrained, max_new_num_unconstrained+delta_n_elems)
    if max_new_num < 0:
        max_new_num = 0

    top_new_potential_elems_all = get_top_n_elements(sensitivity_result, Tail.top, max_new_num)
    top_new_potential_elems = {idx for idx in top_new_potential_elems_all if idx in candidates_for.introduction}
    actual_new_elems = top_new_potential_elems - design_n_min_1.active_elements
    num_new = len(actual_new_elems)

    num_with_new_additions = len(design_n_min_1.active_elements) + num_new
    num_to_go = max(num_with_new_additions - target_count, 0)

    # Remove however many we added.
    existing_elems_only_ranked = {idx: val for idx,val in sensitivity_result.items() if idx in design_n_min_1.active_elements and idx in candidates_for.removal}
    to_go = get_top_n_elements(existing_elems_only_ranked, Tail.bottom, num_to_go)

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

    raise TypeError(rank_comp)


class RankingResults(typing.NamedTuple):
    all_ranks: score.T_ListOfComponentLists
    final_ranking_component: T_index_to_val
    pos_rows: typing.List[db_defs.NodePos]


def _get_ranking_functions(
        optim_params: optimisation_parameters.OptimParams,
        iter_n: int,
        design_n_min_1: design.StentDesign,
        data: datastore.Datastore,
) -> RankingResults:
    """Gets the abaqus results out of the sqlite DB and returns the optimsation element-wise effort results."""

    all_ranks = []

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

    # Element based funtions
    for include_in_opt_comp, elem_component in optim_params.get_all_elem_components():
        for include_in_opt_filter, one_filter in optim_params.get_all_primary_ranking_fitness_filters():
            include_in_opt = include_in_opt_comp and include_in_opt_filter
            this_comp_rows = score.get_primary_ranking_components(include_in_opt, one_filter, data.get_all_rows_at_frame(elem_component, one_frame))
            raw_elem_rows.append( list(this_comp_rows) )

    # Nodal position based functions
    pos_rows = list(data.get_all_rows_at_frame(db_defs.NodePos, one_frame))
    for include_in_opt, one_func in optim_params.get_all_node_position_components():
        this_comp_rows = one_func(optim_params, include_in_opt, design_n_min_1, pos_rows)
        raw_elem_rows.append(list(this_comp_rows))

    # Compute the primary and overall ranking components
    all_ranks.extend(raw_elem_rows)

    # Compute a secondary rank from all the first ones.
    sec_rank_unsmoothed = list(score.get_secondary_ranking_sum_of_norm(all_ranks))
    all_ranks.append(sec_rank_unsmoothed)

    # Smooth the final result
    elem_num_to_indices = {iElem: idx for iElem, idx in design.generate_elem_indices(design_n_min_1.stent_params.divs)}
    elem_indices_to_num = {idx: iElem for iElem, idx in design.generate_elem_indices(design_n_min_1.stent_params.divs)}

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

    return RankingResults(
        all_ranks=all_ranks,
        final_ranking_component=smoothed,
        pos_rows=pos_rows,
    )


def process_completed_simulation(working_dir: pathlib.Path, iter_prev: int) -> typing.Tuple[design.StentDesign, RankingResults]:
    """Processes the results of a completed simulation and returns it's design and results to produce the next iteration."""

    db_fn_prev = history.make_fn_in_dir(working_dir, ".db", iter_prev)
    history_db = history.make_history_db(working_dir)

    with history.History(history_db) as hist:
        stent_params = hist.get_stent_params()
        optim_params = hist.get_opt_params()

    # Go between num (1234) and idx (5, 6, 7)...
    elem_num_to_indices = {iElem: idx for iElem, idx in design.generate_elem_indices(stent_params.divs)}

    with history.History(history_db) as hist:
        snapshot_n_min_1 = hist.get_snapshot(iter_prev)
        design_n_min_1 = design.StentDesign(
            stent_params=stent_params,
            active_elements=frozenset( (elem_num_to_indices[iElem] for iElem in snapshot_n_min_1.active_elements)),
            label=snapshot_n_min_1.label,
        )

    # Get the data from the previously run simulation.
    with datastore.Datastore(db_fn_prev) as data:
        ranking_result = _get_ranking_functions(optim_params, iter_prev, design_n_min_1, data)
        global_status_raw = list(data.get_final_history_result())

    pos_lookup = {row.node_num: base.XYZ(x=row.X, y=row.Y, z=row.Z) for row in ranking_result.pos_rows}

    # Log the history
    with history.History(history_db) as hist:
        status_checks = (history.StatusCheck(
            iteration_num=iter_prev,
            elem_num=rank_comp.elem_id,
            stage=_hist_status_stage_from_rank_comp(rank_comp),
            metric_name=rank_comp.comp_name,
            metric_val=rank_comp.value) for rank_comp_list in ranking_result.all_ranks for rank_comp in rank_comp_list)

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

    return design_n_min_1, ranking_result

T_ProdNewGen = typing.Callable[
    [pathlib.Path, design.StentDesign, RankingResults, int, str],
    design.StentDesign
]

def produce_new_generation(working_dir: pathlib.Path, design_prev: design.StentDesign, ranking_result: RankingResults, iter_this: int, label: str) -> design.StentDesign:

    inp_fn = history.make_fn_in_dir(working_dir, ".inp", iter_this)
    history_db = history.make_history_db(working_dir)
    with history.History(history_db) as hist:
        stent_params = hist.get_stent_params()
        optim_params = hist.get_opt_params()

    new_active_elems = evolve_decider(optim_params, design_prev, ranking_result.final_ranking_component, iter_this)

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
            all_ranks.append(list(score.get_primary_ranking_components(True, optim_params.single_primary_ranking_fitness_filter, db_data)))


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


def evolve_decider_test():
    iter_n_min_1 = 0
    iter_n = iter_n_min_1 + 1

    history_db = pathlib.Path(r"E:\Simulations\StentOpt\AA-178")

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

        status_checks = hist.get_status_checks(0, 1_000_000_000)
        smoothed_checks = [st for st in status_checks if st.stage == history.StatusCheckStage.smoothed]
        smoothed = {elem_num_to_indices[st.elem_num]: st.metric_val for st in smoothed_checks}

    new_active_elems = evolve_decider(optim_params, design_n_min_1, smoothed, iter_n)
    n_new = len(new_active_elems - design_n_min_1.active_elements)
    n_gone = len(design_n_min_1.active_elements - new_active_elems)

    print(f"Gone: {n_gone}\tNew: {n_new}")




if __name__ == '__main__':

    # plot_history_gradient()

    # evolve_decider_test()
    # make_plot_tests()

    working_dir = pathlib.Path(r"E:\Simulations\StentOpt\AA-179")  # 89
    iter_this = 1
    iter_prev = iter_this - 1
    make_plot_tests(working_dir, iter_this)

    one_design, one_ranking = process_completed_simulation(working_dir, iter_prev)
    new_design = produce_new_generation(working_dir, one_design, one_ranking, iter_this)
    # print(new_design)


