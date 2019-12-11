import collections
import typing
import enum
import operator


import numpy
import scipy.ndimage
import matplotlib.pyplot as plt

from stent_opt.odb_interface import datastore, db_defs
from stent_opt.struct_opt import design


class Tail(enum.Enum):
    bottom = enum.auto()
    top = enum.auto()


# TODO - hinge behaviour
#   - Try the Jacobian or somesuch to get the deformation in an element.
#   - Rolling window of "rigid body removed deformation" to favour hinges.
#   - Be able to plot objective functions...


def get_overall_ranking(*lists):
    overall_rank = collections.Counter()

    for one_list in lists:
        for idx, one_row in enumerate(one_list):
            overall_rank[one_row.elem_num] += idx

    return overall_rank


def get_relevant_measure(nt_row):
    """Gets a single criterion for a result type"""

    if isinstance(nt_row, db_defs.ElementPEEQ):
        return nt_row.PEEQ

    elif isinstance(nt_row, db_defs.ElementStress):
        return nt_row.von_mises

    else:
        raise ValueError(nt_row)


def get_element_efforts(*lists) -> typing.Dict[int, float]:

    elem_effort = collections.Counter()

    for one_list in lists:
        rel_measure = [get_relevant_measure(nt_row) for nt_row in one_list]
        ave = numpy.mean(rel_measure)
        for nt_row in one_list:
            this_rel = get_relevant_measure(nt_row)
            if ave:
                effort = this_rel/ave

            elif this_rel == 0:
                effort = 0.0

            else:
                raise ValueError(f"Nonzero over zero? {this_rel} / {ave}")

            elem_effort[nt_row.elem_num] += effort

    return elem_effort


def overall_score_update(*lists):
    """Prints the status."""

    summary_data = []
    for one_list in lists:
        name = one_list[0].__class__.__name__
        rel_measure = [get_relevant_measure(nt_row) for nt_row in one_list]
        ave = numpy.mean(rel_measure)
        total = numpy.sum(rel_measure)

        out_text = f"{name}: Mean={ave}, Total={total}"
        summary_data.append(out_text)

    print('\t'.join(summary_data))


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


def make_new_generation(old_design: design.StentDesign, db_fn: str) -> design.StentDesign:


    # Get the old data.
    with datastore.Datastore(db_fn) as data:

        all_frames = list(data.get_all_frames())

        good_frames = [f for f in all_frames if f.instance_name == "STENT-1"]
        one_frame = good_frames.pop()

        peeq_rows = list(data.get_all_rows_at_frame(db_defs.ElementPEEQ, one_frame))
        stress_rows = list(data.get_all_rows_at_frame(db_defs.ElementStress, one_frame))


        # Order by relevant quantity. Best elements (highly stressed) come first.
        peeq_rows.sort(key=get_relevant_measure, reverse=True)
        stress_rows.sort(key=get_relevant_measure, reverse=True)

        list_of_lists = [peeq_rows, stress_rows]

        overall_rank = get_element_efforts(*list_of_lists)

        # Overall ranking, unweighted.

        #overall_rank = get_overall_ranking(*list_of_lists)


        overall_score_update(*list_of_lists)


    # Get the element number (1234) to idx (5, 6, 7) mapping.
    elem_num_to_indices = {iElem: idx for iElem, idx in design.generate_elem_indices(old_design.design_space)}


    unsmoothed = {elem_num_to_indices[iElem]: val for iElem, val in overall_rank.items()}
    smoothed = gaussian_smooth(old_design.design_space, unsmoothed)


    new_elems = get_top_n_elements(smoothed, len(old_design.active_elements))
    return design.StentDesign(design_space=old_design.design_space, active_elements=frozenset(new_elems))


    # Old stuff
    bottom_tail = get_keys_of_tail(smoothed, Tail.bottom, 10.0)
    top_tail = get_keys_of_tail(smoothed, Tail.top, 10.0)

    to_remove = {k for k in bottom_tail if k in old_design.active_elements}
    to_add = {k for k in top_tail if k not in old_design.active_elements}

    print(f"To add: {len(to_add)}. To remove: {len(to_remove)}")

    new_active_elements = old_design.active_elements - to_remove | to_add
    return design.StentDesign(design_space=old_design.design_space, active_elements=frozenset(new_active_elements))


if __name__ == '__main__':
    make_new_generation(None, r"C:\TEMP\aba\db-9.db")