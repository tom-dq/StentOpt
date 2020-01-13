import itertools
import math
import os
import pathlib
import subprocess
import typing
import collections
import statistics
import enum

import psutil

from stent_opt.abaqus_model import base, amplitude, step, element, part, material
from stent_opt.abaqus_model import instance, main, surface, load, interaction_property
from stent_opt.abaqus_model import interaction, node, boundary_condition, section

from stent_opt.struct_opt.design import PolarIndex, StentDesign, node_from_index, generate_node_indices, generate_elem_indices, get_c3d8_connection
from stent_opt.struct_opt.generation import make_new_generation

from stent_opt.struct_opt import history

working_dir_orig = os.getcwd()
working_dir_extract = os.path.join(working_dir_orig, "odb_interface")

# This seems to be required to get
try:
    os.environ.pop('PYTHONIOENCODING')
except KeyError:
    pass


class GlobalPartNames:
    STENT = "Stent"
    BALLOON = "Balloon"
    CYL_INNER = "CylInner"


class GlobalSurfNames:
    INNER_SURFACE = "InAll"
    INNER_BOTTOM_HALF = "InLowHalf"
    INNER_MIN_RADIUS = "InMinR"
    BALLOON_INNER = "BalInner"
    CYL_INNER = "CylInner"


class GlobalNodeSetNames(enum.Enum):
    BalloonTheta0 = enum.auto()
    BalloonThetaMax = enum.auto()
    BalloonZ0 = enum.auto()
    BalloonZMax = enum.auto()
    RigidCyl = enum.auto()


class Actuation(enum.Enum):
    direct_pressure = enum.auto()
    rigid_cylinder = enum.auto()
    balloon = enum.auto()


class Balloon(typing.NamedTuple):
    inner_radius_ratio: float
    overshoot_ratio: float
    foldover_param: float
    divs: PolarIndex


class Cylinder(typing.NamedTuple):
    initial_radius_ratio: float
    final_radius_ratio: float
    overshoot_ratio: float
    divs: PolarIndex


class StentParams(typing.NamedTuple):
    angle: float
    divs: PolarIndex
    r_min: float
    r_max: float
    length: float
    balloon: typing.Optional[Balloon]
    cylinder: typing.Optional[Cylinder]

    @property
    def actuation(self) -> Actuation:
        has_balloon = bool(self.balloon)
        has_cyl = bool(self.cylinder)

        if has_balloon and has_cyl:
            raise ValueError("Both Cyl and Balloon?")

        if has_balloon:
            return Actuation.balloon

        elif has_cyl:
            return Actuation.rigid_cylinder

        else:
            return Actuation.direct_pressure


    @property
    def actuation_surface_ratio(self) -> typing.Optional[float]:
        if self.actuation == Actuation.rigid_cylinder:
            return self.r_min * self.cylinder.initial_radius_ratio

        elif self.actuation == Actuation.balloon:
            return self.r_min * self.balloon.inner_radius_ratio

        else:
            return None


dylan_r10n1_params = StentParams(
    angle=60,
    divs=PolarIndex(
        R=2,
        Th=31,  # 31
        Z=120,  # 120
    ),
    r_min=0.65,
    r_max=0.75,
    length=11.0,
    balloon=Balloon(
        inner_radius_ratio=0.85,
        overshoot_ratio=0.05,
        foldover_param=0.4,
        divs=PolarIndex(
            R=1,
            Th=20,
            Z=30,
        ),
    ),
    cylinder=Cylinder(
        initial_radius_ratio=0.6,
        final_radius_ratio=3.0,
        overshoot_ratio=0.1,
        divs=PolarIndex(
            R=1,
            Th=51,
            Z=2,
        ),
    )
)

basic_stent_params = dylan_r10n1_params._replace(balloon=None)
def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def _gen_ordinates(n, low, high):
    vals = [ low + (high - low) * (i/(n-1)) for i in range(n)]
    return vals


def generate_nodes_stent_polar(stent_params: StentParams) -> typing.Dict[int, base.RThZ]:

    def gen_nodes():
        r_vals = _gen_ordinates(stent_params.divs.R, stent_params.r_min, stent_params.r_max)
        th_vals = _gen_ordinates(stent_params.divs.Th, 0.0, stent_params.angle)
        z_vals = _gen_ordinates(stent_params.divs.Z, 0.0, stent_params.length)

        for iNode, (r, th, z) in enumerate(itertools.product(r_vals, th_vals, z_vals), start=1):
            polar_coord = base.RThZ(r, th, z)
            yield iNode, polar_coord

    return {iNode: n_p for iNode, n_p in gen_nodes()}


def generate_nodes_inner_cyl(stent_params: StentParams) -> typing.Dict[int, base.RThZ]:
    if stent_params.cylinder.divs.R != 1:
        raise ValueError(stent_params.cylinder.divs.R)

    def gen_nodes():
        theta_half_overshoot = 0.5 * stent_params.angle * stent_params.cylinder.overshoot_ratio
        z_half_overshoot = 0.5 * stent_params.length * stent_params.cylinder.overshoot_ratio

        r_val = stent_params.actuation_surface_ratio
        th_vals = _gen_ordinates(stent_params.cylinder.divs.Th, -theta_half_overshoot, stent_params.angle+theta_half_overshoot)
        z_vals = _gen_ordinates(stent_params.cylinder.divs.Z, -z_half_overshoot, stent_params.length+z_half_overshoot)

        for iNode, (th, z) in enumerate(itertools.product(th_vals, z_vals), start=1):
            polar_coord = base.RThZ(r_val, th, z)
            yield iNode, polar_coord

    return {iNode: n_p for iNode, n_p in gen_nodes()}


def generate_nodes_balloon_polar(stent_params: StentParams) -> typing.Iterable[
    typing.Tuple[base.RThZ, typing.Set[GlobalNodeSetNames]]]:

    # Theta values fall into one of a few potential segments.
    RATIO_A = 0.05
    RATIO_B = 0.15
    RATIO_HEIGHT = 0.1

    # Zero-to-one to radius-ratio
    theta_mapping_unit = [
        (0.0, 0.0),
        (RATIO_A, 0.0),
        (RATIO_A + RATIO_B + stent_params.balloon.foldover_param, RATIO_HEIGHT * stent_params.balloon.foldover_param),
        (1.0 - RATIO_A - RATIO_B - stent_params.balloon.foldover_param, -1 * RATIO_HEIGHT * stent_params.balloon.foldover_param),
        (1.0 - RATIO_A, 0.0),
        (1.0, 0.0),
    ]

    nominal_r = stent_params.actuation_surface_ratio
    theta_half_overshoot = 0.5 * stent_params.angle * stent_params.balloon.overshoot_ratio
    theta_total_span = stent_params.angle + 2*theta_half_overshoot
    theta_mapping = [
        ( -theta_half_overshoot + th*theta_total_span, nominal_r * (1.0+rat)) for th, rat in theta_mapping_unit
    ]

    # Segments are bounded by endpoints in polar coords
    polar_points = [base.RThZ(r=r, theta_deg=theta, z=0.0) for theta, r in theta_mapping]
    segments = [(p1, p2) for p1, p2 in _pairwise(polar_points)]

    # Try to get the nodal spacing more or less even.
    segment_lengths = [abs(p1.to_xyz()-p2.to_xyz()) for p1, p2 in segments]
    total_length = sum(segment_lengths)
    nominal_theta_spacing = total_length / stent_params.balloon.divs.Th

    r_th_points = []

    # Might need to adjust some of the numbers of points to match the spacing requirements.
    planned_num_points = {}

    for idx, (p1, p2) in enumerate(segments):
        this_length = abs(p1.to_xyz()-p2.to_xyz())
        planned_num_points[idx] = round(this_length/nominal_theta_spacing)

    # The plus one is because the balloon spacing is based on the element, not node indices
    out_by_num_points = (sum(planned_num_points.values()) + 1) - stent_params.balloon.divs.Th

    if out_by_num_points == 0:
        pass

    elif abs(out_by_num_points) == 1:
        # Change the middle one
        planned_num_points[2] -= out_by_num_points

    elif abs(out_by_num_points) == 2:
        # Change the two either side
        adjustment = out_by_num_points // 2
        planned_num_points[1] -= adjustment
        planned_num_points[2] -= adjustment

    elif abs(out_by_num_points) == 3:
        # Change the two either side and the middle
        adjustment = out_by_num_points // 3
        planned_num_points[1] -= adjustment
        planned_num_points[2] -= adjustment
        planned_num_points[3] -= adjustment

    else:
        raise ValueError(f"Need to adjust {abs(out_by_num_points)} points?")

    for idx, (p1, p2) in enumerate(segments):
        diff = p2-p1

        num_points = planned_num_points[idx]
        for one_int_point in range(num_points):
            r_th_points.append(p1 + (one_int_point/num_points * diff))

    # Add on the final point
    r_th_points.append(polar_points[-1])

    if len(r_th_points) != stent_params.balloon.divs.Th:
        raise ValueError("Didn't satisfy the spacing requirements.")

    z_overshoot = 0.5 * stent_params.length * (stent_params.balloon.overshoot_ratio)
    z_vals = _gen_ordinates(stent_params.balloon.divs.Z, -z_overshoot, stent_params.length+z_overshoot)
    for (idx_r, r_th), (idx_z, z) in itertools.product(enumerate(r_th_points), enumerate(z_vals)):
        boundary_set = set()
        if idx_r == 0: boundary_set.add(GlobalNodeSetNames.BalloonTheta0)
        if idx_r == len(r_th_points) - 1: boundary_set.add(GlobalNodeSetNames.BalloonThetaMax)
        if idx_z == 0: boundary_set.add(GlobalNodeSetNames.BalloonZ0)
        if idx_z == len(z_vals) - 1: boundary_set.add(GlobalNodeSetNames.BalloonZMax)

        this_point = r_th._replace(z=z)
        yield this_point, boundary_set


def generate_brick_elements_all(divs: PolarIndex) -> typing.Iterable[typing.Tuple[PolarIndex, int, element.Element]]:
    for iElem, one_elem_idx in generate_elem_indices(divs):
        yield one_elem_idx, iElem, element.Element(
            name=element.ElemType.C3D8R,
            connection=get_c3d8_connection(divs, one_elem_idx),
        )

def generate_plate_elements_all(divs: PolarIndex, elem_type: element.ElemType) -> typing.Iterable[typing.Tuple[int, element.Element]]:

    if divs.R != 1:
        raise ValueError(divs)

    def get_iNode(iR, iTh, iZ):
        return node_from_index(divs, iR, iTh, iZ)

    divs_one_extra_thickness = divs._replace(R=divs.R+1)

    for iElem, i in generate_elem_indices(divs_one_extra_thickness):
        connection = [
            get_iNode(i.R, i.Th, i.Z),
            get_iNode(i.R, i.Th+1, i.Z),
            get_iNode(i.R, i.Th+1, i.Z+1),
            get_iNode(i.R, i.Th, i.Z+1),
        ]

        yield iElem, element.Element(
            name=elem_type,
            connection=tuple(connection),
        )

def make_a_stent(stent_params: StentParams, stent_design: StentDesign):

    model = main.AbaqusModel("StentModel")

    def make_stent_part():

        stress_strain_table = (
            material.Point(stress=205., strain=0.0),
            material.Point(stress=515., strain=0.6),
        )
        steel = material.MaterialElasticPlastic(
            name="Steel",
            density=7.85e-09,
            elast_mod=196000.0,
            elast_possion=0.3,
            plastic=stress_strain_table,
        )

        common_section = section.SolidSection(
            name="SolidSteel",
            mat=steel,
        )

        stent_part = part.Part(
            name=GlobalPartNames.STENT,
            common_section=common_section,
        )

        nodes_polar = generate_nodes_stent_polar(stent_params=stent_params)

        for iNode, one_node_polar in nodes_polar.items():
            stent_part.add_node_validated(iNode, one_node_polar.to_xyz())

        for idx, iElem, one_elem in generate_brick_elements_all(divs=stent_params.divs):
            if idx in stent_design.active_elements:
                stent_part.add_element_validate(iElem, one_elem)

        one_instance = instance.Instance(base_part=stent_part)

        model.add_instance(one_instance)

    def make_balloon_part():
        # Material properties are from Dylan's model.
        rubber = material.MaterialElasticPlastic(
            name="Rubber",
            density=1.1E-009,
            elast_mod=920.0,
            elast_possion=0.4,
            plastic=None,
        )

        common_section_balloon = section.MembraneSection(
            name="RubberMembrane",
            mat=rubber,
            thickness=0.02,
        )

        balloon_part = part.Part(
            name=GlobalPartNames.BALLOON,
            common_section=common_section_balloon,
        )

        # Make the nodes and keep track of the leading/trailing edges.
        nodes_polar = {}
        boundary_set_name_to_nodes = collections.defaultdict(set)
        for iNode, (n_p, boundary_sets) in enumerate(generate_nodes_balloon_polar(stent_params=stent_params), start=1):
            nodes_polar[iNode] = n_p
            for boundary_set in boundary_sets:
                boundary_set_name_to_nodes[boundary_set.name].add(iNode)

        for node_set_name, nodes in boundary_set_name_to_nodes.items():
            one_node_set = node.NodeSet(balloon_part, node_set_name, frozenset(nodes))
            balloon_part.add_node_set(one_node_set)

        elems_all = {iElem: e for iElem, e in generate_plate_elements_all(divs=stent_params.balloon.divs, elem_type=element.ElemType.M3D4R)}

        for iNode, one_node_polar in nodes_polar.items():
            balloon_part.add_node_validated(iNode, one_node_polar.to_xyz())

        for iElem, one_elem in elems_all.items():
            balloon_part.add_element_validate(iElem, one_elem)

        one_instance = instance.Instance(base_part=balloon_part)
        model.add_instance(one_instance)

    def make_cyl_part():
        cyl_mat = material.MaterialElasticPlastic(
            name="Cyl",
            density=1.1E-007,
            elast_mod=1e3,
            elast_possion=0.4,
            plastic=None,
        )

        common_section_cyl = section.SurfaceSection(
            name="CylSurf",
            mat=None,
            surf_density=1e-3,
        )

        cyl_inner_part = part.Part(
            name=GlobalPartNames.CYL_INNER,
            common_section=common_section_cyl,
        )

        nodes_polar = generate_nodes_inner_cyl(stent_params=stent_params)

        elems_all = {iElem: e for iElem, e in generate_plate_elements_all(divs=stent_params.cylinder.divs, elem_type=element.ElemType.SFM3D4R)}

        for iNode, one_node_polar in nodes_polar.items():
            cyl_inner_part.add_node_validated(iNode, one_node_polar.to_xyz())

        for iElem, one_elem in elems_all.items():
            cyl_inner_part.add_element_validate(iElem, one_elem)

        # All the nodes go in the boundary set.
        all_node_set = node.NodeSet(cyl_inner_part, GlobalNodeSetNames.RigidCyl.name, frozenset(nodes_polar.keys()))
        cyl_inner_part.add_node_set(all_node_set)

        one_instance = instance.Instance(base_part=cyl_inner_part)

        model.add_instance(one_instance)

    make_stent_part()

    if stent_params.actuation == Actuation.balloon:
        make_balloon_part()

    elif stent_params.actuation == Actuation.rigid_cylinder:
        make_cyl_part()

    return model


def create_surfaces(stent_params: StentParams, model: main.AbaqusModel):
    """Sets up the standard surfaces on the model."""

    stent_instance = model.get_only_instance_base_part_name(GlobalPartNames.STENT)
    stent_part = stent_instance.base_part

    elem_indexes_all = {iElem: i for iElem, i in generate_elem_indices(divs=stent_params.divs)}
    elem_indexes = {iElem: i for iElem, i in elem_indexes_all.items() if iElem in stent_part.elements}

    def create_elem_surface(one_instance: instance.Instance, elem_nums, name, one_surf: surface.SurfaceFace):
        one_part = one_instance.base_part
        elem_set_name = f"ElemSet_{name}"
        the_elements_raw = {iElem: one_part.elements[iElem] for iElem in elem_nums}
        elem_set = element.ElementSet(one_part, elem_set_name, element.Elements(the_elements_raw))
        data = [
            (elem_set, one_surf)
        ]
        the_surface = surface.Surface(name, data)

        one_part.add_element_set(elem_set)
        one_instance.add_surface(the_surface)

    def create_elem_surface_S1(elem_nums, name):
        create_elem_surface(stent_instance, elem_nums, name, surface.SurfaceFace.S1)

    # Get the minimum radius surface
    min_r = min(i.R for i in elem_indexes.values())
    min_r_elems = {iElem for iElem, i in elem_indexes.items() if i.R == min_r}
    create_elem_surface_S1(min_r_elems, GlobalSurfNames.INNER_MIN_RADIUS)

    # Minimum R at any point
    all_elems_through_thickness = collections.defaultdict(set)
    for iElem, i in elem_indexes.items():
        key = (i.Th, i.Z)
        all_elems_through_thickness[key].add( (i.R, iElem) )

    min_elem_through_thickness = set()
    for _, elems in all_elems_through_thickness.items():
        min_r_and_elem = min(elems)
        min_r, min_elem = min_r_and_elem[:]
        min_elem_through_thickness.add(min_elem)

    create_elem_surface_S1(min_elem_through_thickness, GlobalSurfNames.INNER_SURFACE)

    # Minimum R, as long as the R in on the inner half.
    all_r = [i.R for i in elem_indexes.values()]
    median_r = statistics.median(all_r)
    min_bottom_half = set()
    for _, elems in all_elems_through_thickness.items():
        min_r_and_elem = min(elems)
        min_r, min_elem = min_r_and_elem[:]
        if min_r <= median_r:
            min_bottom_half.add(min_elem)

    create_elem_surface_S1(min_bottom_half, GlobalSurfNames.INNER_BOTTOM_HALF)

    if stent_params.actuation == Actuation.balloon:
        instance_balloon = model.get_only_instance_base_part_name(GlobalPartNames.BALLOON)
        all_balloon_elems = instance_balloon.base_part.elements.keys()
        create_elem_surface(instance_balloon, all_balloon_elems, GlobalSurfNames.BALLOON_INNER, surface.SurfaceFace.SNEG)

    elif stent_params.actuation == Actuation.rigid_cylinder:
        instance_cyl = model.get_only_instance_base_part_name(GlobalPartNames.CYL_INNER)
        all_cyl_elements = instance_cyl.base_part.elements.keys()
        create_elem_surface(instance_cyl, all_cyl_elements, GlobalSurfNames.CYL_INNER, surface.SurfaceFace.SNEG)



def apply_loads(stent_params: StentParams, model: main.AbaqusModel):
    if stent_params.actuation == Actuation.rigid_cylinder:
        _apply_loads_enforced_disp(stent_params, model)

    else:
        _apply_loads_pressure(stent_params, model)


def _apply_loads_enforced_disp(stent_params: StentParams, model: main.AbaqusModel):
    init_radius = stent_params.actuation_surface_ratio
    final_radius = stent_params.r_min * stent_params.cylinder.final_radius_ratio
    dr = final_radius - init_radius

    total_time = .3  # TODO! Back to 3.0

    one_step = step.StepDynamicExplicit(
        name=f"Expand",
        final_time=total_time,
        bulk_visc_b1=0.06,
        bulk_visc_b2=1.2,
    )

    model.add_step(one_step)

    amp_data = (
        amplitude.XY(0.0, 0.0),
        amplitude.XY(0.75 * total_time, dr),
        amplitude.XY(total_time, 0),
    )

    amp = amplitude.Amplitude("Amp-1", amp_data)

    cyl_inst = model.get_only_instance_base_part_name(GlobalPartNames.CYL_INNER)
    node_set = cyl_inst.base_part.node_sets[GlobalNodeSetNames.RigidCyl.name]

    enf_disp = boundary_condition.BoundaryDispRot(
        name="ExpRad",
        with_amplitude=amp,
        components=(
            boundary_condition.DispRotBoundComponent(node_set=node_set, dof=1, value=1.0),
            boundary_condition.DispRotBoundComponent(node_set=node_set, dof=2, value=0.0),
            boundary_condition.DispRotBoundComponent(node_set=node_set, dof=3, value=0.0),
        ),
    )

    model.add_load_specific_steps([one_step], enf_disp)



def _apply_loads_pressure(stent_params: StentParams, model: main.AbaqusModel):
    # Some nominal amplitude from Dylan's model.

    if stent_params.balloon:
        max_pressure = 2.0
        total_time = 5.0
        surf = model.get_only_instance_base_part_name(GlobalPartNames.BALLOON).surfaces[GlobalSurfNames.BALLOON_INNER]

    else:
        max_pressure = 0.2
        total_time = 1.5
        surf = model.get_only_instance_base_part_name(GlobalPartNames.STENT).surfaces[GlobalSurfNames.INNER_BOTTOM_HALF]

    one_step = step.StepDynamicExplicit(
        name=f"Expand",
        final_time=total_time,
        bulk_visc_b1=0.06,
        bulk_visc_b2=1.2,
    )

    model.add_step(one_step)

    amp_data = (
        amplitude.XY(0.0, 0.0),
        amplitude.XY(0.65*total_time, 0.16*max_pressure),
        amplitude.XY(0.75*total_time, 0.5*max_pressure),
        amplitude.XY(0.9*total_time, max_pressure),
        amplitude.XY(total_time, 0.05*max_pressure),
    )

    amp = amplitude.Amplitude("Amp-1", amp_data)

    one_load = load.PressureLoad(
        name="Pressure",
        with_amplitude=amp,
        on_surface=surf,
        value=1.0,
    )

    model.add_load_specific_steps([one_step], one_load)


def add_interaction(stent_params: StentParams, model: main.AbaqusModel):
    """Simple interaction between the balloon/cyl and the stent."""



    if stent_params.actuation != Actuation.direct_pressure:

        if stent_params.actuation == Actuation.balloon:
            inner_surf = model.get_only_instance_base_part_name(GlobalPartNames.BALLOON).surfaces[GlobalSurfNames.BALLOON_INNER]

        elif stent_params.actuation == Actuation.rigid_cylinder:
            inner_surf = model.get_only_instance_base_part_name(GlobalPartNames.CYL_INNER).surfaces[GlobalSurfNames.CYL_INNER]

        else:
            raise ValueError(stent_params.actuation)

        outer_surf = model.get_only_instance_base_part_name(GlobalPartNames.STENT).surfaces[GlobalSurfNames.INNER_SURFACE]


        int_prop = interaction_property.SurfaceInteraction(
            name="FrictionlessContact",
            lateral_friction=0.0,
        )

        one_int_general = interaction.GeneralContact(
            name="Interaction",
            int_property=int_prop,
            included_surface_pairs=(
                (inner_surf, outer_surf),
            )
        )


        one_int_specified = interaction.ContactPair(
            name="Interaction",
            int_property=int_prop,
            mechanical_constraint=interaction.MechanicalConstraint.Penalty,
            surface_pair=(inner_surf, outer_surf),
        )

        model.interactions.add(one_int_general)


def gen_active_pairs(stent_params: StentParams, active_node_nums):
    nodes_on_bound = {
        iNode: i for iNode, i in generate_node_indices(stent_params.divs)
        if (i.Th == 0 or i.Th == stent_params.divs.Th - 1) and iNode in active_node_nums
    }

    pairs = collections.defaultdict(set)
    for iNode, i in nodes_on_bound.items():
        key = (i.R, i.Z)
        pairs[key].add(iNode)

    for pair_of_nodes in pairs.values():
        if len(pair_of_nodes) == 1:
            pass  # Only have elements on one side.

        elif len(pair_of_nodes) == 2:
            #if all(iNode in stent_part.nodes for iNode in pair_of_nodes):
            yield tuple(pair_of_nodes)

        else:
            raise ValueError(pair_of_nodes)

def apply_boundaries(stent_params: StentParams, model: main.AbaqusModel):
    """Find the node pairs and couple them, and apply sym conditions."""

    # Stent has coupled boundaries
    stent_instance = model.get_only_instance_base_part_name(GlobalPartNames.STENT)
    stent_part = stent_instance.base_part

    # Only generate the equations on the active nodes.
    uncoupled_nodes_in_part = set()
    for elem in stent_part.elements.values():
        uncoupled_nodes_in_part.update(elem.connection)

    if not uncoupled_nodes_in_part:
        raise ValueError()

    for n1, n2 in gen_active_pairs(stent_params, uncoupled_nodes_in_part):
        stent_instance.add_node_couple(n1, n2, False)
        uncoupled_nodes_in_part.remove(n1)
        uncoupled_nodes_in_part.remove(n2)

    # Fix a single node in Theta and Z to stop rigid body rotation
    arbitrary_uncoupled_node = min(uncoupled_nodes_in_part)
    rigid_node_set = node.NodeSet(stent_part, "RigidRest", frozenset( [arbitrary_uncoupled_node] ))
    stent_part.add_node_set(rigid_node_set)
    bc_components = (
        boundary_condition.DispRotBoundComponent(node_set=rigid_node_set, dof=2, value=0.0),
        boundary_condition.DispRotBoundComponent(node_set=rigid_node_set, dof=3, value=0.0),
    )
    rigid_bc = boundary_condition.BoundaryDispRot(
        name="RigidRestraint",
        with_amplitude=None,
        components=bc_components,
    )
    model.boundary_conditions.add(rigid_bc)

    # Balloon has sym boundaries.
    sym_conds = {
        GlobalNodeSetNames.BalloonTheta0: boundary_condition.BoundaryType.YSYMM,    # YSYMM fixes DoF 2 (i.e., Theta)
        GlobalNodeSetNames.BalloonThetaMax: boundary_condition.BoundaryType.YSYMM,
        GlobalNodeSetNames.BalloonZ0: boundary_condition.BoundaryType.ZSYMM,        # ZSYMM fixes DoF 3 (i.e., Z)
        GlobalNodeSetNames.BalloonZMax: boundary_condition.BoundaryType.ZSYMM,
    }

    if stent_params.actuation == Actuation.balloon:
        balloon_part = model.get_only_instance_base_part_name(GlobalPartNames.BALLOON).base_part

        for sym_name, boundary_type in sym_conds.items():
            relevant_set = balloon_part.node_sets[sym_name.name]
            one_bc = boundary_condition.BoundarySymEncastre(
                name=f"{sym_name} - Sym",
                with_amplitude=None,
                boundary_type=boundary_type,
                node_set=relevant_set,
            )

            model.boundary_conditions.add(one_bc)



def write_model(model: main.AbaqusModel, fn_inp):
    print(fn_inp)
    with open(fn_inp, "w") as fOut:
        for l in model.produce_inp_lines():
            fOut.write(l + "\n")



def ___include_elem_decider_cavities() -> typing.Callable:
    """As a first test, make some cavities in the segment."""

    rs = [0.5 * (basic_stent_params.r_min + basic_stent_params.r_max), ]
    thetas = [0.0, basic_stent_params.angle]
    z_vals = [0, 0.5 * basic_stent_params.length, basic_stent_params.length]

    cavities = []
    for r, th, z in itertools.product(rs, thetas, z_vals):
        cavities.append((base.RThZ(r=r, theta_deg=th, z=z).to_xyz(), 0.25))

    def check_elem(e: element.Element) -> bool:
        for cent, max_dist in cavities:
            this_elem_cent = elem_cent_polar(e)
            dist = abs(cent - this_elem_cent.to_xyz())
            if dist < max_dist:
                return False

        return True

    return check_elem


def make_initial_design_dylan(stent_params: StentParams) -> StentDesign:
    """Makes in initial design (where elements are to be located) based on one of Dylan's early stents."""
    long_strut_width = 0.2
    crossbar_width = 0.25

    long_strut_angs = [0.25, 0.75]
    long_strut_cents = [ratio * stent_params.angle for ratio in long_strut_angs]

    theta_struts_h1 = [0.25* stent_params.length, 0.75 * stent_params.length]
    theta_struts_h2 = [0.0, 0.5 * stent_params.length, stent_params.length]

    nominal_radius = 0.5 * (stent_params.r_min + stent_params.r_max)

    nodes_polar = generate_nodes_stent_polar(stent_params=stent_params)

    def elem_cent_polar(e: element.Element) -> base.RThZ:
        node_pos = [nodes_polar[iNode] for iNode in e.connection]
        ave_node_pos = sum(node_pos) / len(node_pos)
        return ave_node_pos


    def check_elem(e: element.Element) -> bool:
        this_elem_cent = elem_cent_polar(e)

        # Check if on the long struts first
        long_strut_dist_ang = [abs(this_elem_cent.theta_deg - one_ang) for one_ang in long_strut_cents]
        linear_distance = math.radians(min(long_strut_dist_ang)) * nominal_radius
        if linear_distance <= 0.5 * long_strut_width:
            return True

        # If not, check if on the cross bars.
        in_first_half = long_strut_cents[0] < this_elem_cent.theta_deg <= long_strut_cents[1]
        theta_struts = theta_struts_h1 if in_first_half else theta_struts_h2

        theta_strut_dist = [abs(this_elem_cent.z - one_z) for one_z in theta_struts]
        if min(theta_strut_dist) <= 0.5 * crossbar_width:
            return True

        return False

    included_elements = (idx for idx, iElem, e in generate_brick_elements_all(divs=stent_params.divs) if check_elem(e))
    return StentDesign(
        design_space=stent_params.divs,
        active_elements=frozenset(included_elements)
    )

def make_initial_design_curve(stent_params: StentParams) -> StentDesign:
    """Simple sin wave"""

    width = 0.05
    span_ratio = 0.25

    nominal_radius = 0.5 * (stent_params.r_min + stent_params.r_max)

    nodes_polar = generate_nodes_stent_polar(stent_params=stent_params)

    def elem_cent_polar(e: element.Element) -> base.RThZ:
        node_pos = [nodes_polar[iNode] for iNode in e.connection]
        ave_node_pos = sum(node_pos) / len(node_pos)
        return ave_node_pos

    # Discretise line (with a bit of overshoot)
    n_points = 30
    th_points = [ i / (n_points-1) * stent_params.angle for i in range(-2, n_points+2)]
    z_mid = 0.5 * stent_params.length
    z_amp = span_ratio * stent_params.length / 2
    line_z_th_points = [base.RThZ(
        r=nominal_radius,
        theta_deg=th,
        z=z_mid + z_amp * math.sin(2 * math.pi * th / stent_params.angle)).to_xyz()
        for th in th_points
    ]

    def _dist(x1, y1, x2, y2, x3, y3):  # x3,y3 is the point
        px = x2 - x1
        py = y2 - y1

        norm = px * px + py * py

        u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        x = x1 + u * px
        y = y1 + u * py

        dx = x - x3
        dy = y - y3

        # Note: If the actual distance does not matter,
        # if you only want to compare what this function
        # returns to other results of this function, you
        # can just return the squared distance instead
        # (i.e. remove the sqrt) to gain a little performance

        dist = (dx * dx + dy * dy) ** .5

        return dist

    def point_line_dist(p1: base.XYZ, p2: base.XYZ, x0: base.RThZ):
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points

        # Put everything at the nominal radius
        x0_xyz = x0._replace(r=nominal_radius).to_xyz()

        return _dist(p1.x, p1.y, p2.x, p2.y, x0_xyz.x, x0_xyz.y)

        num = abs(
            (p2.y - p1.y) * x0_xyz.x -
            (p2.x - p1.x) * x0_xyz.y +
            p2.x * p1.y -
            p2.y * p1.x
        )
        den = math.sqrt((p2.y - p1.y) ** 2 + (p2.x - p1.x) ** 2)
        return num / den


    def check_elem(e: element.Element) -> bool:
        this_elem_cent = elem_cent_polar(e)

        def all_dists():
            for p1, p2 in _pairwise(line_z_th_points):
                yield point_line_dist(p1, p2, this_elem_cent)

        return min(all_dists()) <= (0.5 * width)

    included_elements = (idx for idx, iElem, e in generate_brick_elements_all(divs=stent_params.divs) if check_elem(e))
    return StentDesign(
        design_space=stent_params.divs,
        active_elements=frozenset(included_elements)
    )



make_initial_design = make_initial_design_curve

def make_design_from_snapshot(stent_params: StentParams, snapshot: history.Snapshot) -> StentDesign:
    elem_num_to_idx = {iElem: idx for idx, iElem, e in generate_brick_elements_all(divs=stent_params.divs)}
    active_elements_idx = {elem_num_to_idx[iElem] for iElem in snapshot.active_elements}

    return StentDesign(
        design_space=stent_params.divs,
        active_elements=frozenset(active_elements_idx),
    )


def make_stent_model(stent_params: StentParams, stent_design: StentDesign, fn_inp: str):
    model = make_a_stent(stent_params, stent_design)
    create_surfaces(stent_params, model)
    apply_loads(stent_params, model)
    add_interaction(stent_params, model)
    apply_boundaries(stent_params, model)
    write_model(model, fn_inp)


def kill_process_id(proc_id: int):
    process = psutil.Process(proc_id)
    for proc in process.children(recursive=True):
        proc.kill()

    process.kill()

def run_model(inp_fn):
    old_working_dir = os.getcwd()


    path, fn = os.path.split(inp_fn)
    fn_solo = os.path.splitext(fn)[0]
    #print(multiprocessing.current_process().name, fn_solo)
    args = ['abaqus.bat', 'cpus=4', f'job={fn_solo}', "ask_delete=OFF", 'interactive']

    os.chdir(path)

    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    TIMEOUT = 3600 * 24  # 1 day

    try:
        ret_code = proc.wait(timeout=TIMEOUT)

    except subprocess.TimeoutExpired:
        ret_code = 'TimeoutExpired on {0}'.format(args)

        try:
            kill_process_id(proc.pid)

        except psutil.NoSuchProcess:
            # No problem if it's already gone in the meantime...
            pass

    if ret_code:
        raise subprocess.SubprocessError(ret_code)

    os.chdir(old_working_dir)


def perform_extraction(odb_fn, out_db_fn):
    old_working_dir = os.getcwd()
    os.chdir(working_dir_extract)
    args = ["abaqus.bat", "cae", "noGui=odb_extract.py", "--", str(odb_fn), str(out_db_fn)]

    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ret_code = proc.wait()
    if ret_code:
        raise subprocess.SubprocessError(ret_code)



def do_opt(stent_params: StentParams):
    working_dir = pathlib.Path(r"E:\Simulations\StentOpt\aba-70")
    history_db = working_dir / "History.db"

    os.makedirs(working_dir, exist_ok=True)

    def make_fn(ext: str, iter_num: int):
        """e.g., iter_num=123 and ext=".inp" """
        intermediary = working_dir / f'It-{str(iter_num).rjust(6, "0")}.XXX'
        return intermediary.with_suffix(ext)

    # If we've already started, use the most recent snapshot in the history.
    with history.History(history_db) as hist:
        hist.set_design_space(stent_params.divs)
        restart_i = hist.max_saved_iteration_num()

    start_from_scratch = restart_i is None

    if start_from_scratch:
        # Do the initial setup from a first model.
        starting_i = 0
        fn_inp = make_fn(".inp", starting_i)
        current_design = make_initial_design(stent_params)
        make_stent_model(stent_params, current_design, fn_inp)
        run_model(fn_inp)
        perform_extraction(make_fn(".odb", starting_i), make_fn(".db", starting_i))

        with history.History(history_db) as hist:
            elem_indices_to_num = {idx: iElem for iElem, idx in generate_elem_indices(stent_params.divs)}
            active_elem_nums = (elem_indices_to_num[idx] for idx in current_design.active_elements)
            snapshot = history.Snapshot(
                iteration_num=starting_i,
                filename=str(fn_inp),
                active_elements=frozenset(active_elem_nums))

            hist.add_snapshot(snapshot)

        main_loop_start_i = 1
    else:
        print(f"Restarting from {restart_i}")
        main_loop_start_i = restart_i

    done = False
    with history.History(history_db) as hist:
        old_design = hist.get_most_recent_design()

    for i_current in range(main_loop_start_i, 1000):
        i_prev = i_current - 1
        fn_inp = make_fn(".inp", i_current)
        fn_db_prev = make_fn(".db", i_prev)
        fn_odb = make_fn(".odb", i_current)

        design = make_new_generation(fn_db_prev, history_db, i_current, fn_inp)

        num_new = len(design.active_elements - old_design.active_elements)
        num_removed = len(old_design.active_elements - design.active_elements)
        print(f"Added: {num_new}\tRemoved: {num_removed}.")
        if design == old_design and (i_current != main_loop_start_i):
            done = True

        make_stent_model(stent_params, design, fn_inp)
        run_model(fn_inp)

        fn_db_current = make_fn(".db", i_current)
        perform_extraction(fn_odb, fn_db_current)

        old_design = design

        if done:
            break



if __name__ == "__main__":
    do_opt(basic_stent_params)

