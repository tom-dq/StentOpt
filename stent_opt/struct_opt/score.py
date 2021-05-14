from __future__ import annotations

import collections
import itertools
import statistics
import typing

import numpy

from stent_opt.abaqus_model import base, element
from stent_opt.abaqus_model.base import XYZ
from stent_opt.odb_interface import db_defs
from stent_opt.struct_opt import design, deformation_grad, graph_connection, optimisation_parameters, common
from stent_opt.struct_opt.deformation_grad import T_NodeMap

_FACES_OF_QUAD = (
    (0, 1, 2, 3,),
)

_FACES_OF_HEX = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)

FACES_OF = {
    element.ElemType.CPE4R: _FACES_OF_QUAD,
    element.ElemType.CPS4R: _FACES_OF_QUAD,
    element.ElemType.C3D8R: _FACES_OF_HEX,
}


class FilterRankingComponent(typing.NamedTuple):
    comp_name: str
    elem_id: int
    value: float  # If there are more than one to remove, highest value is the most "badness"
    include_in_opt: bool

    @property
    def constraint_violation_priority(self) -> float:
        return self.value


class PrimaryRankingComponent(typing.NamedTuple):
    comp_name: str
    elem_id: int
    value: float
    include_in_opt: bool

    @property
    def constraint_violation_priority(self) -> float:
        return 0.0


class SecondaryRankingComponent(typing.NamedTuple):
    comp_name: str
    elem_id: int
    value: float
    include_in_opt: bool

    @property
    def constraint_violation_priority(self) -> float:
        return 0.0


T_PrimaryList = typing.List[PrimaryRankingComponent]
T_SecondaryList = typing.List[SecondaryRankingComponent]
T_AnyRankingComponent = typing.Union[PrimaryRankingComponent, SecondaryRankingComponent, FilterRankingComponent]
T_ListOfComponentLists = typing.List[typing.List[T_AnyRankingComponent]]  # A list of lists, where the sub lists are all primary or all secondary


class GradientInputData(typing.NamedTuple):
    """Used to encapsulate the design at a particular iteration."""
    iteration_num: int
    stent_design: "design.StentDesign"
    element_ranking_components: typing.Dict[int, PrimaryRankingComponent]


class GradientDataPoint(typing.NamedTuple):
    """Used to determine the trend lines for vicinity stress/strain/whatever as elements are turning on and off."""
    elem_id: int
    iteration_num: int
    vicinity_ranking: float
    contributions: int
    element_activation: float  # 0.0 === Void, 1.0 === Solid.


class LineOfBestFit(typing.NamedTuple):
    m: float
    c: float


def composite_stress_peeq_energy(row_type_to_range, elem_num_to_type_to_rows) -> typing.Iterable[db_defs.ElementCustomComposite]:
    """(0.1+vM) * (0.01+PEEQ) * (MaxOverallElasticEnergy - ElasticEnergy + 0.01)"""

    for elem_num, type_to_val in elem_num_to_type_to_rows.items():
        vM = type_to_val[db_defs.ElementStress]
        PEEQ = type_to_val[db_defs.ElementPEEQ]
        ElasticEnergy = type_to_val[db_defs.ElementEnergyElastic]

        elast_energy_min, elast_enery_max = row_type_to_range[db_defs.ElementEnergyElastic]

        one_val = (0.1 + vM) * (0.01 + PEEQ) * (elast_enery_max - ElasticEnergy + 0.01)
        yield db_defs.ElementCustomComposite(
            frame_rowid=None,
            elem_num=elem_num,
            comp_val=one_val,
        )

composite_active = composite_stress_peeq_energy

def compute_composite_ranking_component(nt_rows_all_from_frame) -> typing.Iterable[db_defs.ElementCustomComposite]:
    """This computes a composite function based on existing results in the Datastore. Called after Abaqus has populated it."""

    # Get the min and max of all the primary quantities
    nt_rows_all_from_frame = list(nt_rows_all_from_frame)

    row_type_to_value = collections.defaultdict(list)
    elem_num_to_type_to_rows = collections.defaultdict(dict)

    for row in nt_rows_all_from_frame:
        row_type = type(row)
        row_type_to_value[row_type].append(row.get_last_value())

        # Let us look up, say, elem 12, vM Stress.
        elem_num_to_type_to_rows[row.elem_num][row_type] = row.get_last_value()

    row_type_to_range = {}
    for row_type, all_vals in row_type_to_value.items():
        row_type_to_range[row_type] = min(all_vals), max(all_vals)

    yield from composite_active(row_type_to_range, elem_num_to_type_to_rows)


def _get_primary_ranking_components_raw(include_in_opt, nt_rows) -> typing.Iterable[PrimaryRankingComponent]:
    """All the nt_rows should be the same type"""

    nt_rows = list(nt_rows)
    if not nt_rows:
        return

    nt_row = nt_rows[0]

    if isinstance(nt_row, db_defs.ElementPEEQ):
        for row in nt_rows:
            yield PrimaryRankingComponent(
                comp_name=row.__class__.__name__,
                elem_id=row.elem_num,
                value=row.PEEQ,
                include_in_opt=include_in_opt,
            )

    elif isinstance(nt_row, db_defs.ElementStress):
        for row in nt_rows:
            yield PrimaryRankingComponent(
                comp_name=row.__class__.__name__,
                elem_id=row.elem_num,
                value=row.von_mises,
                include_in_opt=include_in_opt,
            )

    elif isinstance(nt_row, db_defs.ElementEnergyElastic):
        for row in nt_rows:
            yield PrimaryRankingComponent(
                comp_name=row.__class__.__name__,
                elem_id=row.elem_num,
                value=row.ESEDEN,
                include_in_opt=include_in_opt,
            )

    elif isinstance(nt_row, db_defs.ElementEnergyPlastic):
        for row in nt_rows:
            yield PrimaryRankingComponent(
                comp_name=row.__class__.__name__,
                elem_id=row.elem_num,
                value=row.EPDDEN,
                include_in_opt=include_in_opt,
            )

    elif isinstance(nt_row, db_defs.ElementFatigueResult):
        for row in nt_rows:
            yield PrimaryRankingComponent(
                comp_name=row.__class__.__name__,
                elem_id=row.elem_num,
                value=row.LGoodman,
                include_in_opt=include_in_opt,
            )

    elif isinstance(nt_row, db_defs.ElementCustomComposite):
        for row in nt_rows:
            yield PrimaryRankingComponent(
                comp_name=composite_active.__doc__,
                elem_id=row.elem_num,
                value=row.comp_val,
                include_in_opt=include_in_opt,
            )

    elif isinstance(nt_row, db_defs.ElementGlobalPatchSensitivity):
        for row in nt_rows:
            yield PrimaryRankingComponent(
                comp_name=row.__class__.__name__,
                elem_id=row.elem_num,
                value=row.gradient_from_patch,
                include_in_opt=include_in_opt,
            )

    else:
        raise ValueError(nt_row)


def _get_primary_ranking_deviation(include_in_opt: bool, central_value_producer, nt_rows) -> typing.Iterable[PrimaryRankingComponent]:
    """Gets a ranking component which makes elements close to the mean the best."""

    raw_primary_res = list(_get_primary_ranking_components_raw(include_in_opt, nt_rows))

    raw_data = [prc.value for prc in raw_primary_res]
    central_value = central_value_producer(raw_data)

    # Deviations (zero is "right at mean")
    deviations = [prc._replace(value=central_value - prc.value) for prc in raw_primary_res]
    max_dev = max( abs(prc.value) for prc in deviations)

    # Most "fit" is minimum deviation...
    final = [prc._replace(value=max_dev - abs(prc.value)) for prc in deviations]

    return final


def get_primary_ranking_components(include_in_opt: bool, one_filter: common.PrimaryRankingComponentFitnessFilter, nt_rows) -> typing.Iterable[PrimaryRankingComponent]:

    nt_rows = list(nt_rows)
    if not nt_rows:
        return

    if one_filter.is_deviation_from_central_value:
        nt_producer = _get_primary_ranking_deviation(include_in_opt, one_filter.get_central_value_function(), nt_rows)

    else:
        nt_producer = _get_primary_ranking_components_raw(include_in_opt, nt_rows)

    # Add on the suffix for the filter.
    with_filter_suffix = (prc._replace(comp_name=f"{prc.comp_name} {one_filter.name}") for prc in nt_producer)
    yield from with_filter_suffix


def get_primary_ranking_element_distortion(optim_params: optimisation_parameters.OptimParams, include_in_opt: bool, old_design: "design.StentDesign", nt_rows_node_pos) -> typing.Iterable[PrimaryRankingComponent]:

    elem_indices_to_num = {idx: iElem for iElem, idx in design.generate_elem_indices(old_design.stent_params.divs)}

    node_to_pos = {row.node_num: base.XYZ(x=row.X, y=row.Y, z=row.Z) for row in nt_rows_node_pos}

    for elem_idx in old_design.active_elements:
        elem_num = elem_indices_to_num[elem_idx]

        # For submodels there mostly won't be this result...
        try:
            elem_connection = design.get_single_element_connection(old_design.stent_params, elem_idx)
            yield PrimaryRankingComponent(
                comp_name="InternalAngle",
                elem_id=elem_num,
                value=_max_delta_angle_of_element(old_design.stent_params, node_to_pos, elem_connection),
                include_in_opt=include_in_opt,
            )

        except KeyError:
            pass


def _transform_patch_over_boundary(stent_design: "design.StentDesign", node_dict_xyz):

    node_dict_r_th_z = {node_key: xyz.to_r_th_z() for node_key, xyz in node_dict_xyz.items()}

    half_angle = stent_design.stent_params.angle / 2

    def span(node_dict_polar):
        theta_vals = [r_th_z.theta_deg for r_th_z in node_dict_polar.values()]
        return max(theta_vals, default=0) - min(theta_vals, default=0)

    span_orig = span(node_dict_r_th_z)
    if span_orig < half_angle:
        return node_dict_xyz

    # Actually have to rotate some of the angles
    def maybe_rotated(r_th_z: base.RThZ) -> base.RThZ:
        if r_th_z.theta_deg > half_angle:
            return r_th_z._replace(theta_deg=r_th_z.theta_deg - stent_design.stent_params.angle)

        else:
            return r_th_z

    rotated_node_dict = {node_key: maybe_rotated(r_th_z) for node_key, r_th_z in node_dict_r_th_z.items()}
    span_rotated = span(rotated_node_dict)

    if span_rotated < half_angle:
        rotated_in_xyz = {node_key: r_th_z.to_xyz() for node_key, r_th_z in rotated_node_dict.items()}
        return rotated_in_xyz

    raise ValueError(f"Could not find an rotation under {half_angle} degrees - original was {span_orig}, rotated was {span_rotated}.")


def get_line_of_best_fit(x_vals, y_vals) -> LineOfBestFit:
    """Solves y = mx + c given x and y"""
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html

    x = numpy.array(x_vals)
    y = numpy.array(y_vals)

    A = numpy.vstack([x, numpy.ones(len(x))]).T
    m, c = numpy.linalg.lstsq(A, y, rcond=None)[0]

    return LineOfBestFit(m=float(m), c=float(c))


def get_primary_ranking_local_region_gradient(
        stent_params: design.StentParams,
        include_in_opt: bool,
        recent_gradient_input_data: typing.Iterable[GradientInputData],
        region_reducer: typing.Callable[[typing.Iterable[float]], float],  # e.g., max or statistics.mean.
        ) -> typing.Iterable[PrimaryRankingComponent]:
    """Determine how the local stress is correlated with the a particular element being active or inactive."""

    REGION_GRADIENT = "RegionGradient"

    recent_gradient_input_data = list(recent_gradient_input_data)

    # For the first few iterations this doesn't make sense...
    if len(recent_gradient_input_data) < 2:
        return StopIteration()

    # Global lookup for all elements - the divisions will not change.
    elem_indices_to_num = {idx: iElem for iElem, idx in design.generate_elem_indices(recent_gradient_input_data[0].stent_design.stent_params.divs)}
    elem_num_to_index = {iElem: idx for idx, iElem in elem_indices_to_num.items()}

    # For each element, get the local mean or local max stress based on the adjacent elements within some connectivity
    # radius. This needs to be based on the total connectivity of all the which have been included in any considered
    # analysis, not just the current one, so voids can be correctly considered.

    all_active_element_nums_sets = (set(one_snapshot.element_ranking_components.keys()) for one_snapshot in recent_gradient_input_data)
    all_active_element_nums = set.union(*all_active_element_nums_sets)
    all_active_element_idxs = {elem_num_to_index[iElem] for iElem in all_active_element_nums}
    stent_design_all_elems_active = recent_gradient_input_data[0].stent_design._replace(active_elements=frozenset(all_active_element_idxs))

    STENCIL_LENGTH = 4 * stent_params.minimum_element_length  # mm
    elem_to_elems_in_range = graph_connection.element_idx_to_elems_within(STENCIL_LENGTH, stent_design_all_elems_active)
    elem_num_to_region_elem_nums = {
        elem_indices_to_num[elem_idx]: [elem_indices_to_num[one_elem_idx] for one_elem_idx in many_elem_idxs]
        for elem_idx, many_elem_idxs
        in elem_to_elems_in_range.items()
    }

    # Get the maximum or average stress, for each element, within a radius, at each historical snapshot.
    gradient_datapoints = []
    for one_snapshot, (elem_num, region_elem_nums) in itertools.product(recent_gradient_input_data, elem_num_to_region_elem_nums.items()):
        raw_values = [one_snapshot.element_ranking_components[one_elem_num].value for one_elem_num in region_elem_nums if one_elem_num in one_snapshot.element_ranking_components]

        if raw_values:
            # Sometimes we will have a void element which is surrounded by more void elements. Needless to say, we can't get any data from that!
            vicinity_ranking = region_reducer(raw_values)
            elem_is_active = elem_num in one_snapshot.element_ranking_components
            one_gradient_point = GradientDataPoint(
                elem_id=elem_num,
                iteration_num=one_snapshot.iteration_num,
                vicinity_ranking=vicinity_ranking,
                contributions=len(raw_values),
                element_activation=1.0 if elem_is_active else 0.0,
            )
            gradient_datapoints.append(one_gradient_point)

    gradient_datapoints.sort()

    def get_elem_num(one_gradient_point):
        return one_gradient_point.elem_id

    for elem_num, iterable_of_datapoints in itertools.groupby(gradient_datapoints, get_elem_num):
        status_vals = []
        response_vals = []
        for dp in iterable_of_datapoints:

            status_vals.append(dp.element_activation)
            response_vals.append(dp.vicinity_ranking)

        can_make_line = len(set(status_vals)) > 1
        if can_make_line:
            line_of_best_fit = get_line_of_best_fit(status_vals, response_vals)
            grad = -1.0 * line_of_best_fit.m

        else:
            grad = 0.0

        yield PrimaryRankingComponent(
            comp_name=REGION_GRADIENT,
            elem_id=elem_num,
            value=grad,
            include_in_opt=include_in_opt,
        )

    # TODO - include the "authority"


def get_primary_ranking_macro_deformation(optim_params: optimisation_parameters.OptimParams, include_in_opt: bool, old_design: "design.StentDesign", nt_rows_node_pos) -> typing.Iterable[PrimaryRankingComponent]:
    """Gets the local-ish deformation within a given number of elements, by removing the rigid body rotation/translation."""

    # TODO - speed this up! Need to:
    #  - Cache the connectivity somehow
    #  - Prepare one big matrix and submit sub-matrices to nodal_deformation

    elem_to_nodes_in_range = graph_connection.element_idx_to_nodes_within(optim_params.local_deformation_stencil_length, old_design)

    idx_num_elem = [(idx, iElem, elem) for idx, iElem, elem in design.generate_stent_part_elements(old_design.stent_params) if idx in old_design.active_elements]
    elem_indices_to_num = {idx: iElem for (idx, iElem, elem) in idx_num_elem}

    node_to_pos_deformed = {row.node_num: base.XYZ(x=row.X, y=row.Y, z=row.Z) for row in nt_rows_node_pos}

    node_to_pos_original = {
        iNode: r_th_z.to_xyz() for iNode, r_th_z in
        design.get_node_num_to_pos(stent_params=old_design.stent_params).items() if
        iNode in node_to_pos_deformed
    }

    # Make big numpy arrays of all the positions.
    if old_design.stent_params.wrap_around_theta:
        """Here's what I'm thinking for wrap-around theta node patches:
        - Have "raw" data with theta indices like this:
           raw = [0, 1, 2, 3]
        
        - Make a "padded" array which repeats the wrap-around data like this:
           padded = [0, 1, 2, 3, 4, 5, -2, -1]
        
         - The padded array would have entries 4 and 5 as a bit above ThetaMax, and
           entries -2 and -1 as a bit below 0.
           
         - Use numpy.take with mode='wrap', just with the indices from "raw".
         
         - I have not thought this all the way through!
        """

        raise ValueError("Time to write this bit!")

    node_key_order = sorted(node_to_pos_deformed.keys())
    node_key_to_index = {node_key: idx for idx, node_key in enumerate(node_key_order)}
    np_orig = get_points_array(node_key_order, node_to_pos_original)
    np_deformed = get_points_array(node_key_order, node_to_pos_deformed)


    def prepare_node_patch(node_dict_xyz, patch_node_nums):
        raw = {iNode: xyz for iNode, xyz in node_dict_xyz.items() if iNode in patch_node_nums}
        if old_design.stent_params.wrap_around_theta:
            return _transform_patch_over_boundary(old_design, raw)

        else:
            return raw

    node_num_to_contribs = collections.defaultdict(list)

    for elem_idx, node_num_set in elem_to_nodes_in_range.items():

        # if node_key in node_key_to_index is needed if this is a sub-model...
        node_indices = [node_key_to_index[node_key] for node_key in node_num_set if node_key in node_key_to_index]
        np_orig_sub = np_orig.take(node_indices, axis=0)
        np_def_sub = np_deformed.take(node_indices, axis=0)

        #orig = prepare_node_patch(node_to_pos_original, node_num_set)
        #deformed = prepare_node_patch(node_to_pos_deformed, node_num_set)

        this_val = deformation_grad.nodal_deformation(optim_params.local_deformation_stencil_length, np_orig_sub, np_def_sub)
        for node_num in node_num_set:
            node_num_to_contribs[node_num].append(this_val)

    node_num_to_overall_ave = {node_num: statistics.mean(data) for node_num, data in node_num_to_contribs.items()}

    elem_idx_to_nodes = {idx: elem.connection for idx, iElem, elem in idx_num_elem}

    for elem_idx, elem_num in elem_indices_to_num.items():
        nodal_vals = [node_num_to_overall_ave[node_num] for node_num in elem_idx_to_nodes[elem_idx]]
        yield PrimaryRankingComponent(
            comp_name="LocalDeformation",
            elem_id=elem_num,
            value=statistics.mean(nodal_vals),
            include_in_opt=include_in_opt,
        )


def constraint_filter_not_yielded_out(include_in_opt, nt_rows) -> typing.Iterable[FilterRankingComponent]:
    PEEQ_LIMIT = 0.25

    for nt_row in nt_rows:
        if isinstance(nt_row, db_defs.ElementPEEQ):
            should_filter_out = nt_row.PEEQ > PEEQ_LIMIT
            if should_filter_out:
                yield FilterRankingComponent(
                    comp_name=f"PEEQAllowable>{PEEQ_LIMIT}",
                    elem_id=nt_row.elem_num,
                    value=nt_row.PEEQ,
                    include_in_opt=include_in_opt,
                )


def constraint_filter_within_fatigue_life(include_in_opt, nt_rows) -> typing.Iterable[FilterRankingComponent]:
    FATIGUE_LIMIT = 1.0

    for nt_row in nt_rows:
        if isinstance(nt_row, db_defs.ElementFatigueResult):
            should_filter_out = nt_row.LGoodman > FATIGUE_LIMIT
            if should_filter_out:
                yield FilterRankingComponent(
                    comp_name=f"GoodmanAllowable>{FATIGUE_LIMIT}",
                    elem_id=nt_row.elem_num,
                    value=nt_row.LGoodman,
                    include_in_opt=include_in_opt,
                )




def _max_delta_angle_of_element(stent_params: "design.StentParams", node_to_pos: typing.Dict[int, base.XYZ], elem_connection) -> float:
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

    for one_face_idx in FACES_OF[stent_params.stent_element_type]:
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


def _removed_observer_only_components(list_of_lists: typing.List[typing.List[PrimaryRankingComponent]]) -> typing.List[typing.List[PrimaryRankingComponent]]:
    out_list_of_lists = []

    for one_list in list_of_lists:
        for_inclusion_set = {prc.include_in_opt for prc in one_list}
        if len(for_inclusion_set) != 1:
            raise ValueError("How did we get here?", for_inclusion_set, one_list[0:])

        for_inclusion = for_inclusion_set.pop()
        if for_inclusion:
            out_list_of_lists.append(one_list)

    return out_list_of_lists


def get_secondary_ranking_scaled_sum(list_of_lists: typing.List[typing.List[PrimaryRankingComponent]]) -> typing.Iterable[SecondaryRankingComponent]:

    """
        one list                min     min_for_offsetting      scale_factor
        [1, 2, 3]               1       0
    """
    # NB - this kind of assumes all the primary ranking components are zero for low, positive for high.
    # If there are negatives, the scaling will shift them up a bit so the lowest is zero.

    elem_effort = collections.Counter()
    names = []

    include_in_opt_lists = _removed_observer_only_components(list_of_lists)

    # Step 1 - normalise all the primary ranking components to one as a maximum
    for one_list in include_in_opt_lists:
        names.append(one_list[0].comp_name)
        vals = [prim_rank.value for prim_rank in one_list]

        min_val = min(vals)
        max_val = max(vals)

        # Don't shift below 0
        min_for_offsetting = min(0.0, min_val)

        if min_val < 0.0:
            # Shift it all up a bit if the lowest value is negative. But leave the max value output by the method at 1.0
            scale_factor = max_val - min_val
            offset_value = abs(min_val)

        else:
            scale_factor = max_val
            offset_value = 0.0

        # If there's a zero in there, get rid of it!
        scale_factor_safe = scale_factor if scale_factor > 0 else 1.0
        scaled_primary = {prim_rank.elem_id: (prim_rank.value + offset_value) / scale_factor_safe for prim_rank in one_list}

        # Update on a counter adds to the existing value...
        elem_effort.update(scaled_primary)

    sec_name = "ScaledSum[" + "+".join(names) + "]"
    for elem_id, sec_rank_val in elem_effort.items():
        yield SecondaryRankingComponent(
            comp_name=sec_name,
            elem_id=elem_id,
            value=sec_rank_val / len(names),
            include_in_opt=True,
        )


def OLD_get_secondary_ranking_sum_of_norm(list_of_lists: typing.List[typing.List[PrimaryRankingComponent]]) -> typing.Iterable[SecondaryRankingComponent]:
    """Just normalises the ranking components to the mean, and adds each one up."""

    elem_effort = collections.Counter()
    names = []

    include_in_opt_lists = _removed_observer_only_components(list_of_lists)
    for one_list in include_in_opt_lists:
        names.append(one_list[0].comp_name)
        vals = [prim_rank.value for prim_rank in one_list]

        # Shift to the zero-to-one range.
        min_val = min(vals)
        max_val = max(vals)
        span = max_val - min_val

        can_divide = span != 0.0

        ave = numpy.mean(vals)
        for prim_rank in one_list:

            if can_divide:
                normed = (prim_rank.value - min_val) / span

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
            include_in_opt=True,
        )


def get_relevant_measure(nt_row):
    """Gets a single criterion for a result type"""

    if isinstance(nt_row, db_defs.ElementPEEQ):
        return nt_row.PEEQ

    elif isinstance(nt_row, db_defs.ElementStress):
        return nt_row.von_mises

    else:
        raise ValueError(nt_row)


def get_points_array(key_order, nodes: T_NodeMap) -> numpy.array:
    points_in_order = tuple(nodes[k] for k in key_order)
    points_array = numpy.array(points_in_order)
    return points_array


def get_node_map(key_order, points: numpy.array) -> T_NodeMap:
    out_points = {}
    for idx, k in enumerate(key_order):
        x, y, z = [float(_x) for _x in points[idx, :]]
        out_points[k] = XYZ(x, y, z)

    return out_points