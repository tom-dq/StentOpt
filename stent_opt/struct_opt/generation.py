import pathlib
import statistics
import typing
import enum
import operator


import numpy
import scipy.ndimage
import matplotlib.pyplot as plt

from stent_opt.odb_interface import datastore, db_defs
from stent_opt.struct_opt import design
from stent_opt.abaqus_model import base


from stent_opt.struct_opt import display, score, history


class Tail(enum.Enum):
    bottom = enum.auto()
    top = enum.auto()


MAX_CHANGE_IN_VOLUME_RATIO = 0.0025  # No more than this change in volume ratio between increments.
NUM_REDUCE_ITERS = 60

MAKE_PLOTS = False
REGION_GRADIENT_COMPONENT = db_defs.ElementPEEQ  # Either db_defs.ElementStress or ElementPEEQ say, or None to not do region gradient

USED_ELEM_COMPONENTS = [
    db_defs.ElementStress,
    db_defs.ElementPEEQ,
]
SINGLE_COMPONENT = db_defs.ElementStress #  db_defs.ElementStress, or None to

# TODO - hinge behaviour
#   - Try the Jacobian or somesuch to get the deformation in an element.
#   - Rolling window of "rigid body removed deformation" to favour hinges.
#   - Be able to plot objective functions...
#   - Stop it disconnecting bits


# Zero based node indexes from Figure 28.1.1–1 in the Abaqus manual


def get_gradient_input_data(
        working_dir,
        raw_component,
        iteration_nums: typing.Iterable[int],
) -> typing.Iterable[score.GradientInputData]:
    """raw_component is db_defs.ElementStress or db_defs.ElementPEEQ"""

    working_dir = pathlib.Path(working_dir)
    history_db_fn = history.make_history_db(working_dir)

    with history.History(history_db_fn) as hist:
        stent_params = hist.get_stent_params()

        for iter_num in iteration_nums:
            output_db_fn = history.make_fn_in_dir(working_dir, ".db", iter_num)

            snapshot = hist.get_snapshot(iter_num)

            with datastore.Datastore(output_db_fn) as data:
                one_frame = data.get_last_frame_of_instance("STENT-1")

                raw_rows = list(data.get_all_rows_at_frame(raw_component, one_frame))
                raw_ranking_components = list(score.get_primary_ranking_components(raw_rows))
                element_ranking_components = {one_comp.elem_id: one_comp for one_comp in raw_ranking_components}
                yield score.GradientInputData(
                    iteration_num=iter_num,
                    stent_design=design.make_design_from_snapshot(stent_params, snapshot),
                    element_ranking_components=element_ranking_components,
                )


T_index_to_val = typing.Dict[design.PolarIndex, float]
def gaussian_smooth(design_space: design.PolarIndex, unsmoothed: T_index_to_val) -> T_index_to_val:

    GAUSSIAN_SIGMA = 2.0 # Was 0.5, and 2.0

    # Make a 3D array
    design_space_elements = design.node_to_elem_design_space(design_space)
    raw = numpy.zeros(shape=design_space_elements)
    for (r, th, z), val in unsmoothed.items():
        raw[r, th, z] = val

    # Gaussian smooth with wraparound in the Theta direction...
    step1 = scipy.ndimage.gaussian_filter1d(raw, sigma=GAUSSIAN_SIGMA, axis=1, mode='wrap')

    # With the other two ordinates, replicate the closest cell we have.
    step2 = scipy.ndimage.gaussian_filter1d(step1, sigma=GAUSSIAN_SIGMA, axis=0, mode='nearest')
    step3 = scipy.ndimage.gaussian_filter1d(step2, sigma=GAUSSIAN_SIGMA, axis=2, mode='nearest')

    # Put it back and return
    out_nd_array = step3
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



def _target_volume_ratio_ideal(iter_n) -> float:
    """Gradually remove material, then taper off."""

    START_RATIO = 0.12
    FLOOR_RATIO = 0.05

    delta = START_RATIO - FLOOR_RATIO

    reducing = delta * (NUM_REDUCE_ITERS-iter_n)/NUM_REDUCE_ITERS + FLOOR_RATIO

    target_ratio = max(FLOOR_RATIO, reducing)

    return target_ratio


def _target_volume_ratio_clamped(stent_design, iter_n) -> float:

    target_ratio = _target_volume_ratio_ideal(iter_n)

    # If we are too far from the ideal volume ratio, clip it.
    existing_volume_ratio = stent_design.volume_ratio()

    clamped_target = _clamp_update(existing_volume_ratio, target_ratio, MAX_CHANGE_IN_VOLUME_RATIO)
    print(existing_volume_ratio, target_ratio, clamped_target, sep='\t')
    return clamped_target

def _target_num_elems(stent_design: design.StentDesign, iter_n: int) -> int:
    """Get a target element count"""
    fully_populated = stent_design.stent_params.divs.fully_populated_elem_count()
    return int(fully_populated * _target_volume_ratio_clamped(stent_design, iter_n))


def evolve_decider(design_n_min_1, sensitivity_result, iter_n) -> typing.Set[design.PolarIndex]:
    """Decides which elements are coming in and going out."""

    target_count = _target_num_elems(design_n_min_1, iter_n)
    delta_n_elems = target_count - len(design_n_min_1.active_elements)

    # If there was no change in volume ratio, what would be the maximum number of elements we could add?
    max_new_num_unconstrained = int(MAX_CHANGE_IN_VOLUME_RATIO * design_n_min_1.stent_params.divs.fully_populated_elem_count())
    max_new_num = min(max_new_num_unconstrained, max_new_num_unconstrained+delta_n_elems)
    if max_new_num < 0:
        max_new_num = 0

    top_new_potential_elems = get_top_n_elements(sensitivity_result, Tail.top, max_new_num)
    actual_new_elems = top_new_potential_elems - design_n_min_1.active_elements
    num_new = len(actual_new_elems)

    num_with_new_additions = len(design_n_min_1.active_elements) + num_new
    num_to_go = max(num_with_new_additions - target_count, 0)

    # Remove however many we added.
    existing_elems_only_ranked = {idx: val for idx,val in sensitivity_result.items() if idx in design_n_min_1.active_elements}
    to_go = get_top_n_elements(existing_elems_only_ranked, Tail.bottom, num_to_go)

    new_active_elems = (design_n_min_1.active_elements | top_new_potential_elems) - to_go
    return new_active_elems


def _hist_status_stage_from_rank_comp(rank_comp: typing.Union[score.PrimaryRankingComponent, score.SecondaryRankingComponent]) -> history.StatusCheckStage:
    if isinstance(rank_comp, score.PrimaryRankingComponent):
        return history.StatusCheckStage.primary

    elif isinstance(rank_comp, score.SecondaryRankingComponent):
        if "smoothed" in rank_comp.comp_name.lower():
            return history.StatusCheckStage.smoothed

        else:
            return history.StatusCheckStage.secondary

    raise TypeError(rank_comp)



def make_new_generation(working_dir: pathlib.Path, iter_n: int) -> design.StentDesign:
    iter_n_min_1 = iter_n - 1  # Previous iteration number.
    title_if_plotting = f"Iteration {iter_n}"

    db_fn_prev = history.make_fn_in_dir(working_dir, ".db", iter_n-1)
    history_db = history.make_history_db(working_dir)
    inp_fn = history.make_fn_in_dir(working_dir, ".inp", iter_n)

    all_ranks = []

    if REGION_GRADIENT_COMPONENT:
        # Gradient tracking
        n_gradient_tracking = 5
        final_grad_track = iter_n
        first_grad_track = max(0, final_grad_track-n_gradient_tracking)
        grad_track_steps = range(first_grad_track, final_grad_track)

        print(f"Iter {iter_n}, grad track = {list(grad_track_steps)}")
        recent_gradient_input_data = get_gradient_input_data(working_dir, REGION_GRADIENT_COMPONENT, grad_track_steps)
        vicinity_ranking = list(score.get_primary_ranking_local_region_gradient(recent_gradient_input_data, statistics.mean))
        all_ranks.append(vicinity_ranking)

    with history.History(history_db) as hist:
        stent_params = hist.get_stent_params()

    # Go between num (1234) and idx (5, 6, 7)...
    elem_num_to_indices = {iElem: idx for iElem, idx in design.generate_elem_indices(stent_params.divs)}
    elem_indices_to_num = {idx: iElem for iElem, idx in design.generate_elem_indices(stent_params.divs)}

    with history.History(history_db) as hist:
        snapshot_n_min_1 = hist.get_snapshot(iter_n_min_1)
        design_n_min_1 = design.StentDesign(
            stent_params=stent_params,
            active_elements=frozenset( (elem_num_to_indices[iElem] for iElem in snapshot_n_min_1.active_elements))
        )

    # Get the old data.
    with datastore.Datastore(db_fn_prev) as data:

        one_frame = data.get_last_frame_of_instance("STENT-1")

        raw_elem_rows = [
            list(score.get_primary_ranking_components(data.get_all_rows_at_frame(one_component, one_frame)))
            for one_component in USED_ELEM_COMPONENTS
        ]

        pos_rows = list(data.get_all_rows_at_frame(db_defs.NodePos, one_frame))

    pos_lookup = {row.node_num: base.XYZ(x=row.X, y=row.Y, z=row.Z) for row in pos_rows}

    # Compute the primary and overall ranking components
    all_ranks.extend(raw_elem_rows)

    all_ranks.extend([
        # list(score.get_primary_ranking_element_distortion(design_n_min_1, pos_rows)), # Node position based scores
        list(score.get_primary_ranking_macro_deformation(design_n_min_1, pos_rows)),
    ])

    # Compute a secondary rank from all the first ones.
    sec_rank = list(score.get_secondary_ranking_sum_of_norm(all_ranks))
    all_ranks.append(sec_rank)

    # Log the history
    with history.History(history_db) as hist:
        status_checks = (history.StatusCheck(
            iteration_num=iter_n_min_1,
            elem_num=rank_comp.elem_id,
            stage=_hist_status_stage_from_rank_comp(rank_comp),
            metric_name=rank_comp.comp_name,
            metric_val=rank_comp.value) for rank_comp_list in all_ranks for rank_comp in rank_comp_list)

        hist.add_many_status_checks(status_checks)

        # Note the node positions for rendering later on.
        node_pos_for_hist = (history.NodePosition(
            iteration_num=iter_n_min_1,
            node_num=node_num,
            x=pos.x,
            y=pos.y,
            z=pos.z) for node_num, pos in pos_lookup.items())
        hist.add_node_positions(node_pos_for_hist)

    if MAKE_PLOTS:
        stress_rank = list(score.get_primary_ranking_components(stress_rows))
        #local_def_rank = list(score.get_primary_ranking_macro_deformation(design_n_min_1, pos_rows))
        # for plot_rank in [vicinity_ranking, stress_rank, sec_rank]:
        for plot_rank in all_ranks:
            display.render_status(design_n_min_1, pos_lookup, plot_rank, title_if_plotting)

    overall_rank = {one.elem_id: one.value for one in sec_rank}

    unsmoothed = {elem_num_to_indices[iElem]: val for iElem, val in overall_rank.items()}
    smoothed = gaussian_smooth(design_n_min_1.stent_params.divs, unsmoothed)

    # Put back into the StatusCheck form so we can serialise it...
    with history.History(history_db) as hist:
        sec_rank_name = sec_rank[0].comp_name + " Smoothed"
        sec_rank_smoothed = (history.StatusCheck(
                iteration_num=iter_n_min_1,
                elem_num=elem_indices_to_num[elem_id],
                stage=history.StatusCheckStage.smoothed,
                metric_name=sec_rank_name,
                metric_val=val) for elem_id, val in smoothed.items())

        hist.add_many_status_checks(sec_rank_smoothed)

    if MAKE_PLOTS and False:  # Turn this off for now.
        # Dress the smoothed value up as a primary ranking component to display

        smoothed_rank_comps = []
        for elem_idx, value in smoothed.items():
            one_sec = score.SecondaryRankingComponent(
                comp_name=sec_rank[0].comp_name + " Smoothed",
                elem_id=elem_indices_to_num[elem_idx],
                value=value)
            smoothed_rank_comps.append(one_sec)

        display.render_status(design_n_min_1, pos_lookup, smoothed_rank_comps, title_if_plotting)

    new_active_elems = evolve_decider(design_n_min_1, smoothed, iter_n)

    # Save the latest design snapshot
    # Log the history
    new_active_elem_nums = frozenset(elem_indices_to_num[idx] for idx in new_active_elems)
    with history.History(history_db) as hist:
        snapshot = history.Snapshot(
            iteration_num=iter_n,
            filename=str(inp_fn),
            active_elements=new_active_elem_nums)

        hist.add_snapshot(snapshot)

    #new_elems = get_top_n_elements(smoothed, len(old_design.active_elements))
    return design.StentDesign(stent_params=design_n_min_1.stent_params, active_elements=frozenset(new_active_elems))


def make_plot_tests():
    # Bad import - just for testing
    from stent_opt import make_stent

    history_iters = [0, 1]  # [0, 1]
    last_iter = history_iters[-1]

    working_dir = pathlib.Path(r"C:\TEMP\aba\AA-24")

    db_history = history.make_history_db(working_dir)

    all_ranks = []
    iter_to_pos = {}

    # Gradient tracking test
    recent_gradient_input_data = list(get_gradient_input_data(working_dir, db_defs.ElementStress, history_iters))


    vicinity_ranking = list(score.get_primary_ranking_local_region_gradient(recent_gradient_input_data, statistics.mean))
    for idx, x in enumerate(vicinity_ranking):
        print(idx, x, sep='\t')


    # TODO - make the vicinity stuff look right!!

    all_ranks.append(vicinity_ranking)

    for one_iter in history_iters:
        db_fn = history.make_fn_in_dir(working_dir, ".db", one_iter)
        with history.History(db_history) as hist:
            stent_params = hist.get_stent_params()
            old_snapshot = hist.get_snapshot(one_iter)
            old_design = design.make_design_from_snapshot(stent_params, old_snapshot)

        # stent_params = design.basic_stent_params
        # old_design = design.make_initial_design(stent_params)

        with datastore.Datastore(db_fn) as data:
            one_frame = data.get_last_frame_of_instance("STENT-1")

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
            all_ranks.append(list(score.get_primary_ranking_components(db_data)))


    sec_rank = list(score.get_secondary_ranking_sum_of_norm(all_ranks))
    #all_ranks.append(sec_rank)

    for one_rank in all_ranks:
        display.render_status(old_design, pos_lookup, one_rank, "Testing")


def track_history_checks_test():
    working_dir = r"E:\Simulations\StentOpt\aba-98"

    old_data = get_gradient_input_data(working_dir, db_defs.ElementStress, [0,1])
    for x in old_data:
        print(x)



def evolve_decider_test():
    iter_n_min_1 = 0
    iter_n = iter_n_min_1 + 1

    history_db = pathlib.Path(r"C:\TEMP\aba\AA-49\History.db")

    with history.History(history_db) as hist:
        stent_params = hist.get_stent_params()

        elem_num_to_indices = {iElem: idx for iElem, idx in design.generate_elem_indices(stent_params.divs)}

        snapshot_n_min_1 = hist.get_snapshot(iter_n_min_1)
        design_n_min_1 = design.StentDesign(
            stent_params=stent_params,
            active_elements=frozenset( (elem_num_to_indices[iElem] for iElem in snapshot_n_min_1.active_elements))
        )

        status_checks = hist.get_status_checks(0)
        smoothed_checks = [st for st in status_checks if st.stage == history.StatusCheckStage.smoothed]
        smoothed = {elem_num_to_indices[st.elem_num]: st.metric_val for st in smoothed_checks}

    new_active_elems = evolve_decider(design_n_min_1, smoothed, iter_n)
    n_new = len(new_active_elems - design_n_min_1.active_elements)
    n_gone = len(design_n_min_1.active_elements - new_active_elems)

    print(f"Gone: {n_gone}\tNew: {n_new}")




if __name__ == '__main__':
    evolve_decider_test()
    # make_plot_tests()











