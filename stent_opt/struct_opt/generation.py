import typing
import enum
import operator


import numpy
import scipy.ndimage
import matplotlib.pyplot as plt

from stent_opt.odb_interface import datastore, db_defs
from stent_opt.struct_opt import design
from stent_opt.abaqus_model import base


from stent_opt.struct_opt import display
from stent_opt.struct_opt import score


class Tail(enum.Enum):
    bottom = enum.auto()
    top = enum.auto()


MAKE_PLOTS = True

# TODO - hinge behaviour
#   - Try the Jacobian or somesuch to get the deformation in an element.
#   - Rolling window of "rigid body removed deformation" to favour hinges.
#   - Be able to plot objective functions...
#   - Stop it disconnecting bits


# Zero based node indexes from Figure 28.1.1–1 in the Abaqus manual


T_index_to_val = typing.Dict[design.PolarIndex, float]
def gaussian_smooth(design_space: design.PolarIndex, unsmoothed: T_index_to_val) -> T_index_to_val:

    GAUSSIAN_SIGMA = 0.5

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
        out_dict[out_idx] = float(out_nd_array[(r, th, z)])

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


def get_top_n_elements(data: T_index_to_val, n_elems: int) -> typing.Set[design.PolarIndex]:
    """Gets the top however many elements"""

    def sort_key(index_val):
        return index_val[1]


    sorted_data = sorted(data.items(), key=sort_key, reverse=True)
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


def make_new_generation(old_design: design.StentDesign, db_fn: str, title_if_plotting: str = '') -> design.StentDesign:

    # Get the old data.
    with datastore.Datastore(db_fn) as data:

        all_frames = list(data.get_all_frames())

        good_frames = [f for f in all_frames if f.instance_name == "STENT-1"]
        one_frame = good_frames.pop()

        peeq_rows = list(data.get_all_rows_at_frame(db_defs.ElementPEEQ, one_frame))
        stress_rows = list(data.get_all_rows_at_frame(db_defs.ElementStress, one_frame))
        pos_rows = list(data.get_all_rows_at_frame(db_defs.NodePos, one_frame))

    pos_lookup = {row.node_num: base.XYZ(x=row.X, y=row.Y, z=row.Z) for row in pos_rows}

    # Compute the primary and overall ranking components
    all_ranks = [
        list(score.get_primary_ranking_components(peeq_rows)),                    # Elem result based scores
        list(score.get_primary_ranking_components(stress_rows)),
        list(score.get_primary_ranking_element_distortion(pos_rows, old_design)), # Node position based scores
    ]

    # Compute a secondary rank from all the first ones.
    sec_rank = list(score.get_secondary_ranking_sum_of_norm(all_ranks))
    all_ranks.append(sec_rank)

    if MAKE_PLOTS:
        stress_rank = list(score.get_primary_ranking_components(stress_rows))
        for plot_rank in [stress_rank, sec_rank]:
            display.render_status(old_design, pos_lookup, plot_rank, title_if_plotting)

    overall_rank = {one.elem_id: one.value for one in sec_rank}

    # Get the element number (1234) to idx (5, 6, 7) mapping.
    elem_num_to_indices = {iElem: idx for iElem, idx in design.generate_elem_indices(old_design.design_space)}

    unsmoothed = {elem_num_to_indices[iElem]: val for iElem, val in overall_rank.items()}
    smoothed = gaussian_smooth(old_design.design_space, unsmoothed)

    if MAKE_PLOTS and False:  # Turn this off for now.
        # Dress the smoothed value up as a primary ranking component to display
        elem_indices_to_num = {idx: iElem for iElem, idx in design.generate_elem_indices(old_design.design_space)}
        smoothed_rank_comps = []
        for elem_idx, value in smoothed.items():
            one_sec = score.SecondaryRankingComponent(
                comp_name=sec_rank[0].comp_name + " Smoothed",
                elem_id=elem_indices_to_num[elem_idx],
                value=value)
            smoothed_rank_comps.append(one_sec)

        display.render_status(old_design, pos_lookup, smoothed_rank_comps, title_if_plotting)

    new_elems = get_top_n_elements(smoothed, len(old_design.active_elements))
    return design.StentDesign(design_space=old_design.design_space, active_elements=frozenset(new_elems))


def make_plot_tests():
    # Bad import - just for testing
    from stent_opt import make_stent

    db_fn = r"C:\TEMP\aba\opt-5\It-000000.db"

    stent_params = make_stent.basic_stent_params
    old_design = make_stent.make_initial_design(stent_params)

    with datastore.Datastore(db_fn) as data:
        all_frames = list(data.get_all_frames())
        good_frames = [f for f in all_frames if f.instance_name == "STENT-1"]
        one_frame = good_frames.pop()

        peeq_rows = list(data.get_all_rows_at_frame(db_defs.ElementPEEQ, one_frame))
        stress_rows = list(data.get_all_rows_at_frame(db_defs.ElementStress, one_frame))
        pos_rows = list(data.get_all_rows_at_frame(db_defs.NodePos, one_frame))

    pos_lookup = {row.node_num: base.XYZ(x=row.X, y=row.Y, z=row.Z) for row in pos_rows}

    # Node position pased effort functions
    eff_funcs = [score.get_primary_ranking_element_distortion, ]

    all_ranks = []
    for one_func in eff_funcs:
        all_ranks.append(list(one_func(pos_rows, old_design)))

    # Element-based effort functions
    for db_data in [peeq_rows, stress_rows]:
        all_ranks.append(list(score.get_primary_ranking_components(db_data)))

    sec_rank = list(score.get_secondary_ranking_sum_of_norm(all_ranks))
    all_ranks.append(sec_rank)


    for one_rank in all_ranks:
        display.render_status(old_design, pos_lookup, one_rank, "Testing")







if __name__ == '__main__':
    make_plot_tests()
