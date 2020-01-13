import collections
import typing

import numpy

from stent_opt.abaqus_model import base
from stent_opt.odb_interface import db_defs
from stent_opt.struct_opt import design, deformation_grad, graph_connection

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


def get_primary_ranking_element_distortion(old_design: design.StentDesign, nt_rows_node_pos) -> typing.Iterable[PrimaryRankingComponent]:

    elem_indices_to_num = {idx: iElem for iElem, idx in design.generate_elem_indices(old_design.stent_params.divs)}

    node_to_pos = {
        row.node_num: base.XYZ(x=row.X, y=row.Y, z=row.Z) for row in nt_rows_node_pos
    }

    for elem_idx in old_design.active_elements:
        elem_num = elem_indices_to_num[elem_idx]
        elem_connection = design.get_c3d8_connection(old_design.stent_params.divs, elem_idx)
        yield PrimaryRankingComponent(
            comp_name="InternalAngle",
            elem_id=elem_num,
            value=_max_delta_angle_of_element(node_to_pos, elem_connection)
        )


def get_primary_ranking_macro_deformation(old_design: design.StentDesign, nt_rows_node_pos) -> typing.Iterable[PrimaryRankingComponent]:
    """Gets the local-ish deformation within a given number of elements, by removing the rigid body rotation/translation."""

    STENCIL_LENGTH = 0.2  # mm

    elem_to_nodes_in_range = graph_connection.element_nums_to_nodes_within(old_design.stent_params, STENCIL_LENGTH, old_design)

    print(elem_to_nodes_in_range)


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