import itertools
import math
import typing
import collections
import statistics
import enum

from stent_opt.abaqus_model import base, amplitude, step, element, part, material
from stent_opt.abaqus_model import instance, main, surface, load, interaction_property
from stent_opt.abaqus_model import interaction, node, boundary_condition, section


class GlobalPartNames:
    STENT = "Stent"
    BALLOON = "Balloon"


class GlobalSurfNames:
    INNER_SURFACE = "InAll"
    INNER_BOTTOM_HALF = "InLowHalf"
    INNER_MIN_RADIUS = "InMinR"
    BALLOON_INNER = "BalInner"


class GlobalNodeSetNames(enum.Enum):
    BalloonTheta0 = enum.auto()
    BalloonThetaMax = enum.auto()
    BalloonZ0 = enum.auto()
    BalloonZMax = enum.auto()


class PolarIndex(typing.NamedTuple):
    R: int
    Th: int
    Z: int


class Balloon(typing.NamedTuple):
    inner_radius_ratio: float
    overshoot_ratio: float
    foldover_param: float
    divs: PolarIndex


class StentParams(typing.NamedTuple):
    angle: float
    divs: PolarIndex
    r_min: float
    r_max: float
    length: float
    balloon: typing.Optional[Balloon]




dylan_r10n1_params = StentParams(
    angle=60,
    divs=PolarIndex(
        R=3,
        Th=21,
        Z=100,
    ),
    r_min=0.65,
    r_max=0.75,
    length=11.0,
    balloon=Balloon(
        inner_radius_ratio=0.8,
        overshoot_ratio=0.2,
        foldover_param=0.2,
        divs=PolarIndex(
            R=1,
            Th=20,
            Z=30,
        ),
    ),
)

basic_stent_params = dylan_r10n1_params

def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def _gen_ordinates(n, low, high):
    vals = [ low + (high - low) * (i/(n-1)) for i in range(n)]
    return vals

def generate_nodes_stent_polar(stent_params: StentParams) -> typing.Iterable[base.RThZ]:


    r_vals = _gen_ordinates(stent_params.divs.R, stent_params.r_min, stent_params.r_max)
    th_vals = _gen_ordinates(stent_params.divs.Th, 0.0, stent_params.angle)
    z_vals = _gen_ordinates(stent_params.divs.Z, 0.0, stent_params.length)

    for r, th, z in itertools.product(r_vals, th_vals, z_vals):
        polar_coord = base.RThZ(r, th, z)
        yield polar_coord


def generate_nodes_balloon_polar(stent_params: StentParams) -> typing.Iterable[
    typing.Tuple[base.RThZ, typing.Set[GlobalNodeSetNames]]]:

    # Theta values fall into one of a few potential segments.
    RATIO_A = 0.1
    RATIO_B = 0.3
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

    nominal_r = stent_params.balloon.inner_radius_ratio * stent_params.r_min
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
    for p1, p2 in segments:
        this_length = abs(p1.to_xyz()-p2.to_xyz())
        num_points = round(this_length/nominal_theta_spacing)

        diff = p2-p1

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



def _stent_params_node_to_elem_idx(divs: PolarIndex) -> PolarIndex:
    return divs._replace(
        R=divs.R - 1,
        Th=divs.Th - 1,
        Z=divs.Z - 1)

def node_from_index(divs: PolarIndex, iR, iTh, iZ):
    """Node numbers (fully populated)"""
    return (iR * (divs.Th * divs.Z) +
            iTh * divs.Z +
            iZ +
            1)  # One based node numbers!

def elem_from_index(stent_params: StentParams, iR, iTh, iZ):
    """Element numbers (fully populated) - just one fewer number in each ordinate."""
    divs_minus_one = _stent_params_node_to_elem_idx(stent_params.divs)
    return node_from_index(divs_minus_one, iR, iTh, iZ)


def generate_node_indices(divs: PolarIndex) -> typing.Iterable[typing.Tuple[int, PolarIndex]]:
    for iNode, (iR, iTh, iZ) in enumerate(itertools.product(
            range(divs.R),
            range(divs.Th),
            range(divs.Z)), start=1):

        yield iNode, PolarIndex(R=iR, Th=iTh, Z=iZ)

def generate_elem_indices(divs: PolarIndex) -> typing.Iterable[typing.Tuple[int, PolarIndex]]:
    divs_minus_one = _stent_params_node_to_elem_idx(divs)
    yield from generate_node_indices(divs_minus_one)


def generate_brick_elements_all(divs: PolarIndex) -> typing.Iterable[typing.Tuple[int, element.Element]]:

    def get_iNode(iR, iTh, iZ):
        return node_from_index(divs, iR, iTh, iZ)

    for iElem, i in generate_elem_indices(divs):
        connection = [
            get_iNode(i.R, i.Th, i.Z),
            get_iNode(i.R, i.Th+1, i.Z),
            get_iNode(i.R, i.Th+1, i.Z+1),
            get_iNode(i.R, i.Th, i.Z+1),
            get_iNode(i.R+1, i.Th, i.Z),
            get_iNode(i.R+1, i.Th+1, i.Z),
            get_iNode(i.R+1, i.Th+1, i.Z+1),
            get_iNode(i.R+1, i.Th, i.Z+1),
        ]

        yield iElem, element.Element(
            name="C3D8R",
            connection=tuple(connection),
        )

def generate_plate_elements_all(divs: PolarIndex) -> typing.Iterable[typing.Tuple[int, element.Element]]:

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
            name="M3D4R",
            connection=tuple(connection),
        )

def make_a_stent():
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

        nodes_polar = {
            iNode: n_p for iNode, n_p in enumerate(generate_nodes_stent_polar(stent_params=basic_stent_params), start=1)
        }

        elems_all = {iElem: e for iElem, e in generate_brick_elements_all(divs=basic_stent_params.divs)}

        def elem_cent_polar(e: element.Element) -> base.RThZ:
            node_pos = [nodes_polar[iNode] for iNode in e.connection]
            ave_node_pos = sum(node_pos) / len(node_pos)
            return ave_node_pos

        def include_elem_decider_cavities() -> typing.Callable:
            """As a first test, make some cavities in the segment."""

            rs = [0.5 * (basic_stent_params.r_min + basic_stent_params.r_max),]
            thetas = [0.0, basic_stent_params.angle]
            z_vals = [0, 0.5*basic_stent_params.length, basic_stent_params.length]

            cavities = []
            for r, th, z in itertools.product(rs, thetas, z_vals):
                cavities.append( (base.RThZ(r=r, theta_deg=th, z=z).to_xyz(), 0.25))

            def check_elem(e: element.Element) -> bool:
                for cent, max_dist in cavities:
                    this_elem_cent = elem_cent_polar(e)
                    dist = abs(cent - this_elem_cent.to_xyz())
                    if dist < max_dist:
                        return False

                return True

            return check_elem

        def include_elem_decider_dylan() -> typing.Callable:
            """Make an include decider something like Dylan's early stent."""

            long_strut_width = 0.2
            crossbar_width = 0.25

            long_strut_angs = [0.25, 0.75]
            long_strut_cents = [ratio * basic_stent_params.angle for ratio in long_strut_angs]

            theta_struts_h1 = [0.25* basic_stent_params.length, 0.75 * basic_stent_params.length]
            theta_struts_h2 = [0.0, 0.5 * basic_stent_params.length, basic_stent_params.length]

            nominal_radius = 0.5 * (basic_stent_params.r_min + basic_stent_params.r_max)

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

            return check_elem


        for iNode, one_node_polar in nodes_polar.items():
            stent_part.add_node_validated(iNode, one_node_polar.to_xyz())

        should_include_elem = include_elem_decider_dylan()

        for iElem, one_elem in elems_all.items():
            #print(one_elem, elem_cent_polar(one_elem))
            if should_include_elem(one_elem):
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
        for iNode, (n_p, boundary_sets) in enumerate(generate_nodes_balloon_polar(stent_params=basic_stent_params), start=1):
            nodes_polar[iNode] = n_p
            for boundary_set in boundary_sets:
                boundary_set_name_to_nodes[boundary_set.name].add(iNode)

        for node_set_name, nodes in boundary_set_name_to_nodes.items():
            one_node_set = node.NodeSet(balloon_part, node_set_name, frozenset(nodes))
            balloon_part.add_node_set(one_node_set)

        elems_all = {iElem: e for iElem, e in generate_plate_elements_all(divs=basic_stent_params.balloon.divs)}

        for iNode, one_node_polar in nodes_polar.items():
            balloon_part.add_node_validated(iNode, one_node_polar.to_xyz())

        for iElem, one_elem in elems_all.items():
            balloon_part.add_element_validate(iElem, one_elem)

        one_instance = instance.Instance(base_part=balloon_part)
        model.add_instance(one_instance)

    make_stent_part()

    if basic_stent_params.balloon:
        make_balloon_part()

    return model


def create_surfaces(stent_params: StentParams, model: main.AbaqusModel):
    """Sets up the standard surfaces on the model."""

    stent_instance = model.get_only_instance_base_part_name(GlobalPartNames.STENT)
    stent_part = stent_instance.base_part

    elem_indexes_all = {iElem: i for iElem, i in generate_elem_indices(divs=basic_stent_params.divs)}
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

    if stent_params.balloon:
        instance_balloon = model.get_only_instance_base_part_name(GlobalPartNames.BALLOON)
        all_balloon_elems = instance_balloon.base_part.elements.keys()
        create_elem_surface(instance_balloon, all_balloon_elems, GlobalSurfNames.BALLOON_INNER, surface.SurfaceFace.SNEG)


def apply_loads(model: main.AbaqusModel):
    # Some nominal amplitude from Dylan's model.

    one_step = step.StepDynamicExplicit(
        name=f"Expand",
        final_time=4.0,
        bulk_visc_b1=0.06,
        bulk_visc_b2=1.2,
    )

    model.add_step(one_step)

    amp_data = (
        amplitude.XY(0.0, 0.0),
        amplitude.XY(3.25, 0.18),
        amplitude.XY(3.75, 0.2),
        amplitude.XY(4, 0.0),
    )

    amp = amplitude.Amplitude("Amp-1", amp_data)
    surf = model.get_only_instance_base_part_name(GlobalPartNames.BALLOON).surfaces[GlobalSurfNames.BALLOON_INNER]

    one_load = load.PressureLoad(
        name="Pressure",
        amplitude=amp,
        on_surface=surf,
        value=1.0,
    )

    model.add_load_specific_steps([one_step], one_load)


def add_interaction(model: main.AbaqusModel):
    """Simple interaction between the balloon and the stent."""
    if len(model.instances) > 1:
        int_prop = interaction_property.SurfaceInteraction(
            name="FrictionlessContact",
            lateral_friction=0.0,
        )

        one_int = interaction.Interaction(
            name="Interaction",
            int_property=int_prop,
            interaction_surface_type=interaction.InteractionSurfaces.AllWithSelf,
            surface_pairs=None,
        )

        model.interactions.add(one_int)


def apply_boundaries(model: main.AbaqusModel):
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

    def gen_active_pairs():
        nodes_on_bound = {
            iNode: i for iNode, i in generate_node_indices(basic_stent_params.divs)
            if (i.Th == 0 or i.Th == basic_stent_params.divs.Th-1) and iNode in uncoupled_nodes_in_part
        }

        pairs = collections.defaultdict(set)
        for iNode, i in nodes_on_bound.items():

            key = (i.R, i.Z)
            pairs[key].add(iNode)

        for pair_of_nodes in pairs.values():
            if len(pair_of_nodes) != 2:
                raise ValueError(pair_of_nodes)

            if all(iNode in stent_part.nodes for iNode in pair_of_nodes):
                yield tuple(pair_of_nodes)

    for n1, n2 in gen_active_pairs():
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

    has_balloon = len(model.instances) > 1
    if has_balloon:
        balloon_part = model.get_only_instance_base_part_name(GlobalPartNames.BALLOON).base_part

        for sym_name, boundary_type in sym_conds.items():
            relevant_set = balloon_part.node_sets[sym_name.name]
            one_bc = boundary_condition.BoundarySymEncastre(
                name=f"{sym_name} - Sym",
                boundary_type=boundary_type,
                node_set=relevant_set,
            )

            model.boundary_conditions.add(one_bc)



def write_model(model: main.AbaqusModel):
    fn = r"c:\temp\aba\stent-23.inp"
    print(fn)
    with open(fn, "w") as fOut:
        for l in model.produce_inp_lines():
            fOut.write(l + "\n")


if __name__ == "__main__":
    model = make_a_stent()
    create_surfaces(basic_stent_params, model)
    apply_loads(model)
    add_interaction(model)
    apply_boundaries(model)
    write_model(model)
