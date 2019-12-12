import collections
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
FACES_OF_HEX = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)

class PrimaryRankingComponent(typing.NamedTuple):
    comp_name: str
    elem_id: int
    value: float


class SecondaryRankingComponent(typing.NamedTuple):
    comp_name: str
    elem_id: int
    value: float


def get_overall_ranking(*lists):
    overall_rank = collections.Counter()

    for one_list in lists:
        for idx, one_row in enumerate(one_list):
            overall_rank[one_row.elem_num] += idx

    return overall_rank


def get_primary_ranking_components(nt_rows) -> typing.Iterable[PrimaryRankingComponent]:
    """All the nt_rows should be the same type"""

    nt_rows = list(nt_rows)
    nt_row = nt_rows[0]

    if isinstance(nt_row, db_defs.ElementPEEQ):
        for row in nt_rows:
            yield PrimaryRankingComponent(
                comp_name=row.__class__.__name__,
                elem_id=row.elem_num,
                value=row.PEEQ
            )

    elif isinstance(nt_row, db_defs.ElementStress):
        for row in nt_rows:
            yield PrimaryRankingComponent(
                comp_name=row.__class__.__name__,
                elem_id=row.elem_num,
                value=row.von_mises
            )

    else:
        raise ValueError(nt_row)


def get_primary_ranking_element_distortion(nt_rows_node_pos, old_design: design.StentDesign) -> typing.Iterable[PrimaryRankingComponent]:

    node_num_to_indices = {iNode: idx for iNode, idx in design.generate_node_indices(old_design.design_space)}
    elem_indices_to_num = {idx: iElem for iElem, idx in design.generate_elem_indices(old_design.design_space)}


    node_to_pos = {
        row.node_num: base.XYZ(x=row.X, y=row.Y, z=row.Z) for row in nt_rows_node_pos
    }

    for elem_idx in old_design.active_elements:
        elem_num = elem_indices_to_num[elem_idx]
        elem_connection = design.get_c3d8_connection(old_design.design_space, elem_idx)
        yield PrimaryRankingComponent(
            comp_name="InternalAngle",
            elem_id=elem_num,
            value=_max_delta_angle_of_element(node_to_pos, elem_connection)
        )

    pass

def _max_delta_angle_of_element(node_to_pos: typing.Dict[int, base.XYZ], elem_connection) -> float:

    corner_points_numpy = {
        iNode: numpy.array(node_to_pos[iNode]) for iNode in elem_connection
    }

    angles_delta = []
    CORNERS_OF_QUAD = (
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 0),
        (3, 0, 1),
    )
    for one_face_idx in FACES_OF_HEX:
        one_face_nodes = [elem_connection[i] for i in one_face_idx]

        for one_corner_idx in CORNERS_OF_QUAD:
            ia, ib, ic = [one_face_nodes[j] for j in one_corner_idx]

            a = corner_points_numpy[ia]
            b = corner_points_numpy[ib]
            c = corner_points_numpy[ic]

            ba = a - b
            bc = c - b

            cosine_angle = numpy.dot(ba, bc) / (numpy.linalg.norm(ba) * numpy.linalg.norm(bc))
            angle = numpy.arccos(cosine_angle)

            angle_off_by = abs(angle-numpy.pi/2)
            angles_delta.append(angle_off_by)

    return max(angles_delta)


def get_secondary_ranking_sum_of_norm(list_of_lists: typing.List[typing.List[PrimaryRankingComponent]]) -> typing.Iterable[SecondaryRankingComponent]:
    """Just normalises the ranking components to the mean, and adds each one up."""

    elem_effort = collections.Counter()
    names = []
    for one_list in list_of_lists:
        names.append(one_list[0].comp_name)
        vals = [prim_rank.value for prim_rank in one_list]
        ave = numpy.mean(vals)
        for prim_rank in one_list:

            if ave != 0.0:
                normed = prim_rank.value / ave

            elif prim_rank.value == 0:
                normed = 0.0

            else:
                raise ZeroDivisionError(prim_rank.value, ave)

            elem_effort[prim_rank.elem_id] += normed

    sec_name = "NormSum[" + "+".join(names) + "]"
    for elem_id, sec_rank_val in elem_effort.items():
        yield SecondaryRankingComponent(
            comp_name=sec_name,
            elem_id=elem_id,
            value=sec_rank_val / len(names),
        )


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

    all_ranks = []

    # Element-based effort functions
    for db_data in [peeq_rows, stress_rows]:
        all_ranks.append(list(get_primary_ranking_components(db_data)))

    # Node position pased effort functions
    eff_funcs = [get_primary_ranking_element_distortion, ]
    for one_func in eff_funcs:
        all_ranks.append(list(one_func(pos_rows, old_design)))

    sec_rank = list(get_secondary_ranking_sum_of_norm(all_ranks))
    all_ranks.append(sec_rank)

    if MAKE_PLOTS:
        for one_rank in all_ranks:
            display.render_status(old_design, pos_lookup, one_rank, title_if_plotting)


    overall_rank = {one.elem_id: one.value for one in sec_rank}

    # Get the element number (1234) to idx (5, 6, 7) mapping.
    elem_num_to_indices = {iElem: idx for iElem, idx in design.generate_elem_indices(old_design.design_space)}

    unsmoothed = {elem_num_to_indices[iElem]: val for iElem, val in overall_rank.items()}
    smoothed = gaussian_smooth(old_design.design_space, unsmoothed)

    if MAKE_PLOTS:
        # Dress the smoothed value up as a primary ranking component to display
        elem_indices_to_num = {idx: iElem for iElem, idx in design.generate_elem_indices(old_design.design_space)}
        smoothed_rank_comps = []
        for elem_idx, value in smoothed.items():
            one_sec = SecondaryRankingComponent(
                comp_name=sec_rank[0].comp_name + " Smoothed",
                elem_id=elem_indices_to_num[elem_idx],
                value=value)
            smoothed_rank_comps.append(one_sec)

        display.render_status(old_design, pos_lookup, smoothed_rank_comps, title_if_plotting)

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
    eff_funcs = [get_primary_ranking_element_distortion, ]

    all_ranks = []
    for one_func in eff_funcs:
        all_ranks.append(list(one_func(pos_rows, old_design)))

    # Element-based effort functions
    for db_data in [peeq_rows, stress_rows]:
        all_ranks.append(list(get_primary_ranking_components(db_data)))

    sec_rank = list(get_secondary_ranking_sum_of_norm(all_ranks))
    all_ranks.append(sec_rank)


    for one_rank in all_ranks:
        display.render_status(old_design, pos_lookup, one_rank, "Testing")







if __name__ == '__main__':
    make_plot_tests()
