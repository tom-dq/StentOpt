import collections
import enum
import functools
import itertools
import math
import typing
import hashlib

from scipy.optimize import root_scalar

from stent_opt.abaqus_model import base, element



# Convert to and from database string formats.
from stent_opt.struct_opt import history
from stent_opt.struct_opt.common import BaseModelForDB

# These can be serialised and deserialised
# from stent_opt.struct_opt.history import nt_to_db_strings, nt_from_db_strings, Snapshot
from stent_opt.struct_opt.graph_connection import get_elems_in_centre


class PolarIndex(BaseModelForDB):
    R: int
    Th: int
    Z: int

    class Config:
        frozen = True

    def to_db_strings(self):
        yield from history.nt_to_db_strings(self)

    @classmethod
    def from_db_strings(cls, data):
        return history.nt_from_db_strings(cls, data)

    def fully_populated_elem_count(self) -> int:
        z_elems = self.Z - 1
        th_elems = self.Th - 1
        r_elems = max(self.R - 1, 1)
        return z_elems * th_elems * r_elems

    def fully_populated_node_count(self) -> int:
        return self.R * self.Th * self.Z

    def to_tuple(self) -> typing.Tuple[int]:
        return (self.R, self.Th, self.Z)

    def __lt__(self, other):
        return self.to_tuple() < other.to_tuple()

    def __hash__(self):
        """For some reason the default hash was not reliable."""
        return hash(self.to_tuple())

class Actuation(enum.Enum):
    direct_pressure = enum.auto()
    rigid_cylinder = enum.auto()
    balloon = enum.auto()
    enforced_displacement_plane = enum.auto()


class Balloon(BaseModelForDB):
    inner_radius_ratio: float
    overshoot_ratio: float
    foldover_param: float
    divs: PolarIndex

    def to_db_strings(self):
        yield from history.nt_to_db_strings(self)

    @classmethod
    def from_db_strings(cls, data):
        return history.nt_from_db_strings(cls, data)


class Cylinder(BaseModelForDB):
    initial_radius_ratio: float
    overshoot_ratio: float
    divs: PolarIndex

    def to_db_strings(self):
        yield from history.nt_to_db_strings(self)

    @classmethod
    def from_db_strings(cls, data):
        return history.nt_from_db_strings(cls, data)


class InadmissibleRegion(BaseModelForDB):
    theta_min: float
    theta_max: float
    z_min: float
    z_max: float

    def to_db_strings(self):
        yield from history.nt_to_db_strings(self)

    @classmethod
    def from_db_strings(cls, data):
        return history.nt_from_db_strings(cls, data)


class StentParams(BaseModelForDB):
    angle: float
    divs: PolarIndex
    r_min: float
    r_max: float
    length: float
    stent_element_type: element.ElemType
    balloon: typing.Optional[Balloon]
    cylinder: typing.Optional[Cylinder]
    expansion_ratio: typing.Optional[float]
    inadmissible_regions: typing.Tuple[InadmissibleRegion, ...]

    def to_db_strings(self):
        yield from history.nt_to_db_strings(self)

    def polar_index_admissible(self, elem_polar_index: PolarIndex):
        elem_theta_delta = self.angle / (self.divs.Th - 1)
        elem_z_delta = self.length / (self.divs.Z - 1)

        def region_is_ok(inadmissible_region):
            elem_cent_th = (elem_polar_index.Th + 0.5) * elem_theta_delta  # The half is to get to the centroid
            elem_cent_z = (elem_polar_index.Z + 0.5) * elem_z_delta

            if not inadmissible_region.theta_min < elem_cent_th < inadmissible_region.theta_max:
                return True

            if not inadmissible_region.z_min < elem_cent_z < inadmissible_region.z_max:
                return True

            return False

        for inadmissible_region in self.inadmissible_regions:
            # TODO - get this watertight for off-by-one, rounding and centroid concerns.
            if not region_is_ok(inadmissible_region):
                return False

        return True



    @classmethod
    def from_db_strings(cls, data):
        return history.nt_from_db_strings(cls, data)

    @property
    def actuation(self) -> Actuation:
        has_balloon = bool(self.balloon)
        has_cyl = bool(self.cylinder)
        is_2d = self.stent_element_dimensions == 2

        if is_2d:
            if has_balloon or has_cyl:
                raise ValueError("2D but Cyl or Balloon?")

            return Actuation.enforced_displacement_plane

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

    @property
    def stent_element_dimensions(self) -> int:
        return 2 if self._is_planar else 3

    @property
    def radial_thickness(self) -> float:
        return self.r_max - self.r_min

    @property
    def radial_midplane_initial(self) -> float:
        return 0.5 * (self.r_max + self.r_min)

    @property
    def theta_arc_initial(self):
        return self.radial_midplane_initial * math.radians(self.angle)

    @property
    def single_element_r_span(self) -> float:
        if self.stent_element_dimensions == 2:
            return 0.0

        else:
            return self.radial_thickness / self.divs.R

    @property
    def single_element_theta_span(self) -> float:
        return self.theta_arc_initial / self.divs.Th

    @property
    def single_element_z_span(self) -> float:
        return self.length / self.divs.Z

    def get_span_of_patch_connectivity_buffered(self, patch_hops: typing.Optional[int]) -> base.RThZ:
        if patch_hops:
            num_elem_span = 3 * patch_hops

        else:
            num_elem_span = 1

        return num_elem_span * base.RThZ(self.single_element_r_span, self.single_element_theta_span, self.single_element_z_span)

    @property
    def wrap_around_theta(self) -> bool:
        """Defines whether or not there are symmetrical boundary conditions between theta=0 and theta=th_max"""
        return self.stent_element_dimensions == 3

    @property
    def nodal_z_override_in_odb(self) -> typing.Optional[float]:
        """Sometimes we need to put the nodes back at a fixed z location."""
        if self._is_planar:
            return self.radial_midplane_initial

        else:
            return None

    @property
    def _is_planar(self) -> bool:
        if self.stent_element_type in (element.ElemType.CPS4R, element.ElemType.CPE4R):
            return True

        elif self.stent_element_type == element.ElemType.C3D8R:
            return False

        else:
            raise ValueError("Invalid element type for stent.", self.stent_element_type)

    @property
    def minimum_element_length(self) -> float:
        options = [self.single_element_theta_span, self.single_element_z_span,]
        if self.stent_element_dimensions == 3:
            options.append(self.radial_thickness / self.divs.R)

        return min(options)


class StentDesign(typing.NamedTuple):
    stent_params: StentParams
    active_elements: typing.FrozenSet[PolarIndex]
    label: str

    def volume_ratio(self) -> float:
        active_elements = len(self.active_elements)
        fully_populated = self.stent_params.divs.fully_populated_elem_count()
        return active_elements / fully_populated

    def with_additional_elements(self, new_active: typing.Iterable[PolarIndex]) -> "StentDesign":
        working_active = set(self.active_elements)
        working_active.update(new_active)

        return self._replace(active_elements=frozenset(working_active))


def node_to_elem_design_space(divs: PolarIndex) -> PolarIndex:
    return PolarIndex(
        R=max(divs.R - 1, 1),  # Special case if we're at R=1 - that means plate elements.
        Th=divs.Th - 1,
        Z=divs.Z - 1,
    )


def node_from_index(divs: PolarIndex, iR, iTh, iZ):
    """Node numbers (fully populated)"""
    return (iR * (divs.Th * divs.Z) +
            iTh * divs.Z +
            iZ +
            1)  # One based node numbers!


def elem_from_index(divs: PolarIndex, iR, iTh, iZ):
    """Element numbers (fully populated) - just one fewer number in each ordinate."""
    divs_minus_one = node_to_elem_design_space(divs)
    return node_from_index(divs_minus_one, iR, iTh, iZ)


def generate_node_indices(divs: PolarIndex) -> typing.Iterable[typing.Tuple[int, PolarIndex]]:
    for iNode, (iR, iTh, iZ) in enumerate(itertools.product(
            range(divs.R),
            range(divs.Th),
            range(divs.Z)), start=1):

        yield iNode, PolarIndex(R=iR, Th=iTh, Z=iZ)

def generate_stent_boundary_nodes(stent_params: StentParams) -> typing.Iterable[PolarIndex]:
    """Not in order!"""

    if stent_params.stent_element_dimensions == 2:
        for theta_limit in (0, stent_params.divs.Th-1):
            yield from (PolarIndex(R=0, Th=theta_limit, Z=z) for z in range(stent_params.divs.Z))

        for z_limit in (0, stent_params.divs.Z-1):
            yield from (PolarIndex(R=0, Th=th, Z=z_limit) for th in range(1, stent_params.divs.Th-1))

    else:
        raise ValueError("3D Code Writing Time!")

def generate_elem_indices(divs: PolarIndex) -> typing.Iterable[typing.Tuple[int, PolarIndex]]:
    divs_minus_one = node_to_elem_design_space(divs)
    yield from generate_node_indices(divs_minus_one)


def generate_elem_indices_admissible(stent_params: StentParams) -> typing.Iterable[typing.Tuple[int, PolarIndex]]:
    yield from ((i, pe) for i, pe in generate_elem_indices(stent_params.divs) if stent_params.polar_index_admissible(pe))



def get_c3d8_connection(design_space: PolarIndex, i: PolarIndex):

    connection = [
        node_from_index(design_space, i.R, i.Th, i.Z),
        node_from_index(design_space, i.R, i.Th + 1, i.Z),
        node_from_index(design_space, i.R, i.Th + 1, i.Z + 1),
        node_from_index(design_space, i.R, i.Th, i.Z + 1),
        node_from_index(design_space, i.R + 1, i.Th, i.Z),
        node_from_index(design_space, i.R + 1, i.Th + 1, i.Z),
        node_from_index(design_space, i.R + 1, i.Th + 1, i.Z + 1),
        node_from_index(design_space, i.R + 1, i.Th, i.Z + 1),
    ]

    return tuple(connection)


def get_2d_plate_connection(design_space: PolarIndex, i: PolarIndex):

    connection = [
        node_from_index(design_space, i.R, i.Th, i.Z),
        node_from_index(design_space, i.R, i.Th + 1, i.Z),
        node_from_index(design_space, i.R, i.Th + 1, i.Z + 1),
        node_from_index(design_space, i.R, i.Th, i.Z + 1),
    ]

    return tuple(connection)


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
    PLANAR_LEADING_EDGE = "Lead"


class GlobalNodeSetNames(enum.Enum):
    BalloonTheta0 = enum.auto()
    BalloonThetaMax = enum.auto()
    BalloonZ0 = enum.auto()
    BalloonZMax = enum.auto()
    RigidCyl = enum.auto()
    PlanarStentTheta0 = enum.auto()  # These are the sides - only enforce the displacement at the base
    PlanarStentThetaMax = enum.auto()
    PlanarStentZMin = enum.auto()




def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def gen_ordinates(n, low, high):
    if n == 1:
        if math.isclose(low, high):
            return [low]

        else:
            raise ValueError(low, high)

    else:
        vals = [ low + (high - low) * (i/(n-1)) for i in range(n)]
        return vals


def generate_nodes(stent_params: StentParams) -> typing.List[typing.Tuple[int, PolarIndex, typing.Union[base.XYZ, base.RThZ]]]:
    if stent_params.stent_element_dimensions == 2:
        return list(_generate_nodes_stent_planar(stent_params))

    elif stent_params.stent_element_dimensions == 3:
        return list(_generate_nodes_stent_polar(stent_params))

    else:
        raise ValueError(stent_params.stent_element_dimensions)


def get_node_num_to_pos(stent_params) -> typing.Dict[int, typing.Union[base.XYZ, base.RThZ]]:
    return {iNode: pos for iNode, _, pos in generate_nodes(stent_params)}


def _generate_nodes_stent_polar(stent_params: StentParams) -> typing.Iterable[typing.Tuple[int, PolarIndex, base.RThZ]]:
    r_vals = gen_ordinates(stent_params.divs.R, stent_params.r_min, stent_params.r_max)
    th_vals = gen_ordinates(stent_params.divs.Th, 0.0, stent_params.angle)
    z_vals = gen_ordinates(stent_params.divs.Z, 0.0, stent_params.length)

    for iNode, ((iR, r), (iTh, th), (iZ, z)) in enumerate(itertools.product(
            enumerate(r_vals),
            enumerate(th_vals),
            enumerate(z_vals)
    ), start=1):
        polar_index = PolarIndex(R=iR, Th=iTh, Z=iZ)
        polar_coord = base.RThZ(r, th, z)
        yield iNode, polar_index, polar_coord


def _generate_nodes_stent_planar(stent_params: StentParams) -> typing.Iterable[typing.Tuple[int, PolarIndex, base.XYZ]]:
    """Unwrap the stent into a 2D representation.
    Polar      Cartesian
    R      <->    Z
    Theta  <->    X
    Z      <->    Y
    """

    x_vals = gen_ordinates(stent_params.divs.Th, 0.0, stent_params.theta_arc_initial)
    y_vals = gen_ordinates(stent_params.divs.Z, 0.0, stent_params.length)
    z_vals = gen_ordinates(stent_params.divs.R, stent_params.radial_midplane_initial, stent_params.radial_midplane_initial)

    for iNode, ((ix, x), (iy, y), (iz, z)) in enumerate(
            itertools.product(
                enumerate(x_vals),
                enumerate(y_vals),
                enumerate(z_vals)
            ), start=1):
        polar_index = PolarIndex(R=iz, Th=ix, Z=iy)
        xyz_coord = base.XYZ(x, y, z)
        yield iNode, polar_index, xyz_coord


def generate_nodes_inner_cyl(stent_params: StentParams) -> typing.Dict[int, base.RThZ]:
    if stent_params.cylinder.divs.R != 1:
        raise ValueError(stent_params.cylinder.divs.R)

    def gen_nodes():
        theta_half_overshoot = 0.5 * stent_params.angle * stent_params.cylinder.overshoot_ratio
        z_half_overshoot = 0.5 * stent_params.length * stent_params.cylinder.overshoot_ratio

        r_val = stent_params.actuation_surface_ratio
        th_vals = gen_ordinates(stent_params.cylinder.divs.Th, -theta_half_overshoot, stent_params.angle + theta_half_overshoot)
        z_vals = gen_ordinates(stent_params.cylinder.divs.Z, -z_half_overshoot, stent_params.length + z_half_overshoot)

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
    z_vals = gen_ordinates(stent_params.balloon.divs.Z, -z_overshoot, stent_params.length + z_overshoot)
    for (idx_r, r_th), (idx_z, z) in itertools.product(enumerate(r_th_points), enumerate(z_vals)):
        boundary_set = set()
        if idx_r == 0: boundary_set.add(GlobalNodeSetNames.BalloonTheta0)
        if idx_r == len(r_th_points) - 1: boundary_set.add(GlobalNodeSetNames.BalloonThetaMax)
        if idx_z == 0: boundary_set.add(GlobalNodeSetNames.BalloonZ0)
        if idx_z == len(z_vals) - 1: boundary_set.add(GlobalNodeSetNames.BalloonZMax)

        this_point = r_th._replace(z=z)
        yield this_point, boundary_set


def get_single_element_connection(stent_params: StentParams, i: PolarIndex) -> tuple:
    if stent_params.stent_element_dimensions == 2:
        return get_2d_plate_connection(stent_params.divs, i)

    elif stent_params.stent_element_dimensions == 3:
        return get_c3d8_connection(stent_params.divs, i)

    else:
        raise ValueError(stent_params.stent_element_dimensions)


def generate_stent_part_elements(stent_params: StentParams) -> typing.Iterable[typing.Tuple[PolarIndex, int, element.Element]]:
    if stent_params.stent_element_dimensions == 2:
        all_elems = generate_plate_elements_all(stent_params.divs, stent_params.stent_element_type)

    elif stent_params.stent_element_dimensions == 3:
        all_elems = generate_brick_elements_all(stent_params.divs)

    else:
        raise ValueError(stent_params.stent_element_dimensions)

    yield from ((pe, i, e) for (pe, i, e) in all_elems if stent_params.polar_index_admissible(pe))


def generate_brick_elements_all(divs: PolarIndex) -> typing.Iterable[typing.Tuple[PolarIndex, int, element.Element]]:
    for iElem, one_elem_idx in generate_elem_indices(divs):
        yield one_elem_idx, iElem, element.Element(
            name=element.ElemType.C3D8R,
            connection=get_c3d8_connection(divs, one_elem_idx),
        )


def generate_plate_elements_all(divs: PolarIndex, elem_type: element.ElemType) -> typing.Iterable[typing.Tuple[PolarIndex, int, element.Element]]:

    if divs.R != 1:
        raise ValueError(divs)

    for iElem, one_elem_idx in generate_elem_indices(divs):
        yield one_elem_idx, iElem, element.Element(
            name=elem_type,
            connection=get_2d_plate_connection(divs, one_elem_idx),
        )


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


def make_initial_design_dylan(stent_params: StentParams) -> StentDesign:
    """Makes in initial design (where elements are to be located) based on one of Dylan's early stents."""
    long_strut_width = 0.2
    crossbar_width = 0.25

    long_strut_angs = [0.25, 0.75]
    long_strut_cents = [ratio * stent_params.angle for ratio in long_strut_angs]

    theta_struts_h1 = [0.25* stent_params.length, 0.75 * stent_params.length]
    theta_struts_h2 = [0.0, 0.5 * stent_params.length, stent_params.length]

    nominal_radius = 0.5 * (stent_params.r_min + stent_params.r_max)

    nodes_polar = get_node_num_to_pos(stent_params=stent_params)

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

    included_elements = (idx for idx, iElem, e in generate_stent_part_elements(stent_params) if check_elem(e))
    return StentDesign(
        stent_params=stent_params,
        active_elements=frozenset(included_elements),
        label="InitialDylan",
    )


def _make_design_from_line_segments(stent_params: StentParams, line_z_th_points_and_widths: typing.List[typing.Tuple[base.XYZ, float]], nominal_radius: float) -> StentDesign:

    node_pos_dict = get_node_num_to_pos(stent_params=stent_params)

    dims_3d = stent_params.stent_element_dimensions == 3

    def elem_cent_polar(e: element.Element) -> base.RThZ:
        node_pos = [node_pos_dict[iNode] for iNode in e.connection]
        ave_node_pos = sum(node_pos) / len(node_pos)
        return ave_node_pos



    def _dist(x1, y1, x2, y2, x3, y3):  # x3,y3 is the point
        px = x2 - x1
        py = y2 - y1

        norm = px * px + py * py

        num = ((x3 - x1) * px + (y3 - y1) * py)
        den = float(norm)
        try:
            u = num / den

        except ZeroDivisionError as e:
            if num == 0:
                u = 0.0

            else:
                raise e

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

        if dims_3d:
            # Put everything at the nominal radius
            x0_xyz = x0._replace(r=nominal_radius).to_xyz()

        else:
            # Planar
            x0_xyz = x0

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

        def dists_ok():
            for (p1, w1), (p2, w2) in _pairwise(line_z_th_points_and_widths):
                w = 0.5 * (w1+w2)
                yield point_line_dist(p1, p2, this_elem_cent) <= (0.5 * w)

        return any(dists_ok())

    included_elements = frozenset(idx for idx, iElem, e in generate_stent_part_elements(stent_params) if check_elem(e))
    return StentDesign(
        stent_params=stent_params,
        active_elements=included_elements,
        label="InitialLineSegments",
    )


def make_initial_design_curve(stent_params: StentParams) -> StentDesign:
    """Simple sin wave"""

    width = 0.05
    span_ratio = 0.25
    nominal_radius = 0.5 * (stent_params.r_min + stent_params.r_max)

    # Discretise line (with a bit of overshoot)
    n_points = 30
    th_points = [ i / (n_points-1) * stent_params.angle for i in range(-2, n_points+2)]
    z_mid = 0.5 * stent_params.length
    z_amp = span_ratio * stent_params.length / 2
    line_z_th_points_and_widths = [(base.RThZ(
        r=nominal_radius,
        theta_deg=th,
        z=z_mid + z_amp * math.sin(2 * math.pi * th / stent_params.angle)).to_xyz(), width)
                                   for th in th_points
                                   ]

    return _make_design_from_line_segments(stent_params, line_z_th_points_and_widths, nominal_radius)


def make_initial_design_sharp(stent_params: StentParams) -> StentDesign:
    """Make a sharp corner."""

    width = 0.05
    span_ratio = 0.5
    nominal_radius = 0.5 * (stent_params.r_min + stent_params.r_max)
    n_points_half = 15

    # Make an ellipse with no elegance at all.
    theta_quarter_circle = [i * math.pi/(2*(n_points_half-1)) for i in range(n_points_half)]
    quarter_circle_points = [base.RThZ(r=nominal_radius, theta_deg=math.sin(th_param), z=math.cos(th_param)) for th_param in theta_quarter_circle]

    z_amp = span_ratio * stent_params.length
    th_amp = 0.5 * stent_params.angle

    scaled_points = [base.RThZ(r=p.r, theta_deg=p.theta_deg * th_amp, z=p.z * z_amp) for p in quarter_circle_points]

    z_cent = 0.5 * (stent_params.length-z_amp)

    offset = base.RThZ(r=0.0, theta_deg=0.0, z=z_cent)
    offset_points = [p + offset for p in scaled_points]
    reflected_points = [base.RThZ(r=p.r, theta_deg=stent_params.angle-p.theta_deg, z=p.z) for p in reversed(offset_points[:-1])]
    all_points = offset_points + reflected_points

    line_z_th_points_and_widths = [(p.to_xyz(), width) for p in all_points]
    return _make_design_from_line_segments(stent_params, line_z_th_points_and_widths, nominal_radius)


def make_initial_straight_edge(stent_params: StentParams) -> StentDesign:
    """Make a sharp corner with straight edges."""
    width = 0.1
    nominal_radius = 0.5 * (stent_params.r_min + stent_params.r_max)

    z_left = 0.3333333 * stent_params.length
    z_right = 0.6666666 * stent_params.length
    polar_points = [
        base.RThZ(r=nominal_radius, theta_deg=0.0 * stent_params.angle, z=z_left),
        base.RThZ(r=nominal_radius, theta_deg=0.1 * stent_params.angle, z=z_left),
        base.RThZ(r=nominal_radius, theta_deg=0.5 * stent_params.angle, z=z_right),
        base.RThZ(r=nominal_radius, theta_deg=0.9 * stent_params.angle, z=z_left),
        base.RThZ(r=nominal_radius, theta_deg=1.0 * stent_params.angle, z=z_left),
    ]

    if stent_params.stent_element_dimensions == 2:
        xyz_point = [p.to_planar_unrolled() for p in polar_points]

    elif stent_params.stent_element_dimensions == 3:
        xyz_point = [p.to_xyz() for p in polar_points]

    else:
        raise ValueError(stent_params.stent_element_dimensions)

    line_z_th_points_and_widths = [(p, width) for p in xyz_point]
    return _make_design_from_line_segments(stent_params, line_z_th_points_and_widths, nominal_radius)


def make_initial_all_in(stent_params: StentParams) -> StentDesign:
    included_elements = (idx for idx, iElem, e in generate_stent_part_elements(stent_params))
    return StentDesign(
        stent_params=stent_params,
        active_elements=frozenset(included_elements),
        label="InitialAllIn",
    )


def make_initial_all_in_with_hole(stent_params: StentParams) -> StentDesign:
    all_in_design = make_initial_all_in(stent_params)

    cent_elems = get_elems_in_centre(all_in_design)
    without_centre = set(all_in_design.active_elements) - set(cent_elems)

    return StentDesign(
        stent_params=stent_params,
        active_elements=frozenset(without_centre),
        label="WithoutCent",
    )


def make_initial_two_lines(stent_params: StentParams) -> StentDesign:
    """Make a simpler sharp corner with straight edges."""
    width = 0.25
    nominal_radius = stent_params.radial_midplane_initial

    z_left = 0.2 * stent_params.length
    z_right = 0.8 * stent_params.length
    polar_points = [
        base.RThZ(r=nominal_radius, theta_deg=0.0 * stent_params.angle, z=z_left),
        base.RThZ(r=nominal_radius, theta_deg=0.5 * stent_params.angle, z=z_right),
        base.RThZ(r=nominal_radius, theta_deg=1.0 * stent_params.angle, z=z_left),
    ]

    if stent_params.stent_element_dimensions == 2:
        xyz_point = [p.to_planar_unrolled() for p in polar_points]

    elif stent_params.stent_element_dimensions == 3:
        xyz_point = [p.to_xyz() for p in polar_points]

    else:
        raise ValueError(stent_params.stent_element_dimensions)

    line_z_th_points_and_widths = [(p, width) for p in xyz_point]
    return _make_design_from_line_segments(stent_params, line_z_th_points_and_widths, nominal_radius)


def make_initial_zig_zag(n_steps: int, l: float, target_vol_ratio: float, stent_params: StentParams) -> StentDesign:

    # height is set such that the length is constant...
    if n_steps:
        h = (l - stent_params.theta_arc_initial) / n_steps

        if h <= 0:
            raise Warning(f"Could not satisfy length request. got h={h}")
            h = 0.1

        if h >= stent_params.length:
            raise Warning(f"Could not satisfy length request. got h={h}, which is longer that the design space.")
            h = 0.95 * stent_params.length

    else:
        h=0

    def make_zig_zag_width(width: float) -> StentDesign:

        theta_span = stent_params.angle / (n_steps+1)
        nominal_radius = stent_params.radial_midplane_initial

        def get_z_val_for(i_span:int) -> float:
            z_mid = 0.5 * stent_params.length
            if i_span == 0:
                return z_mid

            elif i_span == n_steps + 1:
                return z_mid

            else:
                delta = h / 2 if i_span % 2 == 1 else -h/2
                return z_mid + delta

        polar_points = [
            base.RThZ(r=nominal_radius, theta_deg=0.0, z=get_z_val_for(0)),
            base.RThZ(r=nominal_radius, theta_deg=theta_span*0.5, z=get_z_val_for(0)),
        ]

        for i_span in range(1, n_steps+1):
            th_start = theta_span * (i_span-0.5)
            th_end = theta_span * (i_span+0.5)
            polar_points.extend([
                base.RThZ(r=nominal_radius, theta_deg=th_start, z=get_z_val_for(i_span)),
                base.RThZ(r=nominal_radius, theta_deg=th_end, z=get_z_val_for(i_span)),
            ])

        # End flat bit.
        polar_points.extend([
            base.RThZ(r=nominal_radius, theta_deg=stent_params.angle - theta_span*0.5, z=get_z_val_for(n_steps+1)),
            base.RThZ(r=nominal_radius, theta_deg=stent_params.angle, z=get_z_val_for(n_steps+1)),
        ])

        if stent_params.stent_element_dimensions == 2:
            xyz_point = [p.to_planar_unrolled() for p in polar_points]

        else:
            raise ValueError(stent_params.stent_element_dimensions)

        line_z_th_points_and_widths = [(p, width) for p in xyz_point]
        return _make_design_from_line_segments(stent_params, line_z_th_points_and_widths, nominal_radius)

    def zig_zag_vol_frac(width: float) -> float:
        stent_design = make_zig_zag_width(width)
        return stent_design.volume_ratio() - target_vol_ratio

    min_res = root_scalar(zig_zag_vol_frac, bracket=(0.0, 10.0), maxiter=15)
    print(f"{n_steps=}, {h=}, {l=}, width={min_res.root}, {min_res.flag}")

    return make_zig_zag_width(min_res.root)





def _radius_test_param_curve(stent_params: StentParams, r_minor: float, r_major: float, D: float) -> typing.Callable[[float], base.XYZ]:

    if (D/2) >= (r_major + r_minor):
        raise ValueError("Need larger radius or smaller gap")

    phi = math.acos(D / (2 * (r_minor + r_major)))
    y_span_total_centreline = r_minor + math.sin(phi) * (r_minor + r_major) + r_major
    y_left = 0.5 * (stent_params.length - y_span_total_centreline)
    y_right = 0.5 * (stent_params.length + y_span_total_centreline)

    nominal_radius = stent_params.radial_midplane_initial

    top_bound = base.RThZ(r=nominal_radius, theta_deg=stent_params.angle, z=stent_params.length).to_planar_unrolled()

    # Do all this work in the "unrolled XYZ" coordinate system so the circles are circles
    minor_centroid = base.XYZ(x=0.5*top_bound.x - D/2, y=y_left+r_minor, z=nominal_radius)
    major_centroid = base.XYZ(x=0.5*top_bound.x, y=y_right-r_major, z=nominal_radius)

    P1 = base.RThZ(r=nominal_radius, theta_deg=0.0, z=y_left).to_planar_unrolled()
    P2 = base.XYZ(x=0.5*top_bound.x - D/2, y=y_left, z=nominal_radius)
    #P2 = base.RThZ(r=nominal_radius, theta_deg=minor_cent_theta_deg, z=y_left).to_planar_unrolled()
    P3 = minor_centroid + base.XYZ(x=r_minor * math.cos(phi), y=r_minor * math.sin(phi), z=0.0)
    P4 = base.RThZ(r=nominal_radius, theta_deg=0.5 * stent_params.angle, z=y_right).to_planar_unrolled()

    #To fix - make the circles circles on the real domain (so, taking into account the raduis and theta). Either set an appropriate radius? or unroll-reroll?
    # TODO -plot and check
    # TODO -interpolate

    def _make_param_polar_point_radius_test(t: float) -> base.XYZ:
        # https://math.stackexchange.com/a/3324260

        if t > 0.5:
            mirror_point = _make_param_polar_point_radius_test(1.0 - t)
            return mirror_point._replace(x=top_bound.x - mirror_point.x)

        def line_between(p_start, p_end, r):
            delta = (p_end - p_start)
            return p_start + (r * delta)

        def circle_between(flipped, p_mid, p_start, p_end, rat):
            def angle(p):
                p_from_cent = p - p_mid
                return math.atan2(p_from_cent.y, p_from_cent.x)

            def radius(p):
                p_from_cent = p - p_mid
                return math.sqrt(p_from_cent.x ** 2 + p_from_cent.y ** 2)

            ang_start = angle(p_start)
            ang_end = angle(p_end)

            angle_delta = ang_end - ang_start
            if flipped:
                angle_delta = -1 * (math.pi * 2 - angle_delta)

            ang_out = ang_start + rat * angle_delta

            rad_start = radius(p_start)
            rad_end = radius(p_end)

            rad_out = rad_start + rat * (rad_end-rad_start)

            this_p_from_cent = base.XYZ(x=rad_out * math.cos(ang_out), y=rad_out * math.sin(ang_out), z=nominal_radius)
            return p_mid + this_p_from_cent

        if t < 1/6:
            # First segment - on the flat
            r = t * 6
            return line_between(P1, P2, r)

        elif t < 2/6:
            # Minor circle
            r = (t - 1/6) * 6
            return circle_between(False, minor_centroid, P2, P3, r)

        else:
            # Major circle
            r = (t - 2/6) * 6
            return circle_between(True, major_centroid, P3, P4, r)


    # TEMP - just while getting this going...
    TEMP_TESTING_PLOT = False

    if TEMP_TESTING_PLOT:
        import matplotlib.pyplot as plt
        def plot_point(p, label):
            loc = f"{p.x}, {p.y}"
            plt.text(p.x, p.y, f"{label}")


        def plot_bounds():
            points = (
                base.RThZ(r=nominal_radius, theta_deg=0.0, z=y_left - 0.2 * y_span_total_centreline),
                base.RThZ(r=nominal_radius, theta_deg=stent_params.angle, z=y_left - 0.2 * y_span_total_centreline),
                base.RThZ(r=nominal_radius, theta_deg=stent_params.angle, z=y_right + 0.2 * y_span_total_centreline),
                base.RThZ(r=nominal_radius, theta_deg=0.0, z=y_right + 0.2 * y_span_total_centreline),
                base.RThZ(r=nominal_radius, theta_deg=0.0, z=y_left - 0.2 * y_span_total_centreline),
            )

            xx = [p.to_planar_unrolled().x for p in points]
            yy = [p.to_planar_unrolled().y  for p in points]
            plt.plot(xx, yy)

            # Centreline
            points_cline = (
                base.RThZ(r=nominal_radius, theta_deg=0.5 * stent_params.angle, z=y_left - 0.2 * y_span_total_centreline),
                base.RThZ(r=nominal_radius, theta_deg=0.5 * stent_params.angle, z=y_right + 0.2 * y_span_total_centreline),
            )
            xx_c = [p.to_planar_unrolled().x for p in points_cline]
            yy_c = [p.to_planar_unrolled().y for p in points_cline]

            plt.plot( xx_c, yy_c, '--')

        plot_point(P1, "P1")
        plot_point(P2, "P2")
        plot_point(P3, "P3")
        plot_point(P4, "P4")
        plot_point(minor_centroid, "Min")
        plot_point(major_centroid, "Maj")

        plot_bounds()

        def plot_cent_line():
            ts = [x / 100 for x in range(101)]
            points = [_make_param_polar_point_radius_test(t) for t in ts]
            xx = [p.x for p in points]
            yy = [p.y for p in points]
            plt.plot(xx, yy, 'k')

        plot_cent_line()

        plt.gca().axis('equal')
        plt.show()


    return _make_param_polar_point_radius_test




def make_initial_design_radius_test(stent_params: StentParams) -> StentDesign:
    """Simple outline for testing the radius of curvature"""

    def width(x):
        end_w = 0.05
        middle_w = 0.025
        trans_start=0.3
        trans_end=0.4

        if x > 0.5:
            return width(1.0-x)

        if x < trans_start:
            return end_w

        elif x < trans_end:
            return end_w + (middle_w-end_w) * (x-trans_start)/(trans_end-trans_start)

        else:
            return middle_w

    ref_length = basic_stent_params.theta_arc_initial / 6
    f = _radius_test_param_curve(basic_stent_params, r_minor=0.3 * ref_length, r_major=2.8*ref_length, D=5.5*ref_length )

    N = 100
    xyz_point = [(f(t/N), width(t/N)) for t in range(N+1)]

    return _make_design_from_line_segments(stent_params, xyz_point, stent_params.radial_midplane_initial)




def _s_curve(stent_params: StentParams) -> typing.Callable[[float], base.XYZ]:


    nominal_radius = stent_params.radial_midplane_initial

    def make_p(t: float) -> base.XYZ:
        t_bar = math.pi * 2 * t

        r_th_z = base.RThZ(
            theta_deg=stent_params.angle * (t + 1.0 * math.sin(t_bar)),
            z= bottom_gap + (target_height/2)*(1 - math.cos(t_bar)),
            r=nominal_radius)

        return r_th_z.to_planar_unrolled()

    return make_p



def make_initial_design_s_curve(stent_params: StentParams) -> StentDesign:

    width = 0.20375
    nominal_radius = stent_params.radial_midplane_initial

    z_span = stent_params.length / 5
    z_bottom = 0.5 * (stent_params.length - z_span)

    def make_p(t: float) -> base.RThZ:
        t_bar = math.pi * 2 * t

        return base.RThZ(
            r=nominal_radius,
            theta_deg= stent_params.angle * (t + math.sin(2 * t_bar) / math.pi / 2),  #    stent_params.angle * (t + math.sin(2*t_bar)),
            z=z_bottom + z_span * 0.5 * (1 - math.cos(t_bar)),
            )

    N = 300
    polar_points = [make_p(t/(N-1)) for t in range(N)]

    if stent_params.stent_element_dimensions == 2:
        xyz_point = [p.to_planar_unrolled() for p in polar_points]

    elif stent_params.stent_element_dimensions == 3:
        xyz_point = [p.to_xyz() for p in polar_points]

    else:
        raise ValueError(stent_params.stent_element_dimensions)

    line_z_th_points_and_widths = [(p, width) for p in xyz_point]
    return _make_design_from_line_segments(stent_params, line_z_th_points_and_widths, nominal_radius)


# make_initial_design = make_initial_all_in
# make_initial_design = make_initial_all_in_with_hole
# make_initial_design = make_initial_design_radius_test
# make_initial_design = make_initial_two_lines
# make_initial_design = make_initial_design_s_curve

make_initial_design = functools.partial(make_initial_zig_zag, 1, 5.5, 0.2)


def make_design_from_snapshot(stent_params: StentParams, snapshot: "history.Snapshot") -> StentDesign:
    elem_num_to_idx = {iElem: idx for idx, iElem, e in generate_brick_elements_all(divs=stent_params.divs)}
    active_elements_idx = {elem_num_to_idx[iElem] for iElem in snapshot.active_elements}

    return StentDesign(
        stent_params=stent_params,
        active_elements=frozenset(active_elements_idx),
        label=snapshot.label,
    )

def show_initial_model_test():
    from stent_opt.struct_opt import display
    stent_design = make_initial_design(basic_stent_params)
    display.show_design(stent_design)


dylan_r10n1_params = StentParams(
    angle=60,
    divs=PolarIndex(
        R=1,
        Th=60,  # 20
        Z=200,  # 80
    ),
    r_min=0.65,
    r_max=0.75,
    length=5.0, # Was 11.0
    stent_element_type=element.ElemType.CPS4R,
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
        overshoot_ratio=0.1,
        divs=PolarIndex(
            R=1,
            Th=51,
            Z=2,
        ),
    ),
    expansion_ratio=1.5,  # 2.0
    inadmissible_regions=[
        InadmissibleRegion(
            theta_min=20,
            theta_max=40,
            z_min=0,
            z_max=2.5,
        )
    ],
)


basic_stent_params = dylan_r10n1_params.copy_with_updates(balloon=None, cylinder=None, inadmissible_regions=tuple())


if __name__ == "__main__":
    show_initial_model_test()
    #  check we can serialise and deserialse the optimisation parameters


    data = dylan_r10n1_params.json(indent=2)
    print(data)


    again = StentParams.parse_raw(data)

    print(again)
    print(dylan_r10n1_params)
    assert dylan_r10n1_params == again


if 213==234:

    ref_length = basic_stent_params.theta_arc_initial / 6
    f = _radius_test_param_curve(basic_stent_params, r_minor=ref_length, r_major=2*ref_length, D=3*ref_length )

if 111 == 345/456:
    x = [x * 0.01 for x in range(101)]
    f = _radius_test_param_curve()

    ps = [f(xx) for xx in x ]
    print(ps)

    xxx = [p.to_planar_unrolled().x for p in ps]
    yyy = [p.to_planar_unrolled().y for p in ps]

    import matplotlib.pyplot as plt
    plt.plot(xxx, yyy)
    plt.show()

    #stent_design = make_initial_all_in(basic_stent_params)
    #show_initial_model_test(stent_design)