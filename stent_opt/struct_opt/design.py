import collections
import enum
import itertools
import math
import typing

from stent_opt.abaqus_model import base, element



# Convert to and from database string formats.
from stent_opt.struct_opt import history

# These can be serialised an deserialsed
_enum_types = [
    element.ElemType,
]


def _get_type_ignoring_nones(some_type):
    if getattr(some_type, "__origin__", None) is typing.Union:
        non_none_args = [t for t in some_type.__args__ if t != type(None)]

    else:
        non_none_args = [some_type]

    if len(non_none_args) != 1:
        raise ValueError(some_type)

    return non_none_args[0]

def _nt_to_db_strings(nt_instance) -> typing.Iterable[typing.Tuple[str, typing.Optional[str]]]:
    """(key, value) pairs which can go into a database"""
    for key, val in nt_instance._asdict().items():
        this_item_type = _get_type_ignoring_nones(nt_instance._field_types[key])
        matched_enum_types = [et for et in _enum_types if isinstance(val, et)]
        if this_item_type in (int, float):
            yield key, str(val)

        elif val is None:
            # Actual null in the DB.
            yield key, None

        elif hasattr(val, "to_db_strings"):
            # Delegate to the lower level
            for sub_key, sub_val in val.to_db_strings():
                yield f"{key}.{sub_key}", sub_val

        elif len(matched_enum_types) == 1:
            # Enums can also be stored - just use the name and we can figure out the type again later on.
            enum_type = matched_enum_types[0]
            yield key, val.name

        else:
            raise TypeError(f"Don't known what to make of {key}: {val} type {this_item_type}.")


def _nt_from_db_strings(nt_class, data):

    def get_prefix(one_string_and_data: str):
        one_string, _ = one_string_and_data
        if "." in one_string:
            return one_string.split(".", maxsplit=1)[0]

        else:
            return None

    def remove_prefix(one_string_and_data: str):
        one_string, one_data = one_string_and_data
        if "." in one_string:
            return one_string.split(".", maxsplit=1)[1], one_data

        else:
            return None, one_data



    working_data = {}
    for prefix, data_sublist in itertools.groupby(sorted(data), key=get_prefix):
        if prefix:
            # Have to delegate to a child class to create.
            without_prefix_data = [remove_prefix(one_data) for one_data in data_sublist]

            nt_subclass = nt_class._field_types[prefix]
            single_type = _get_type_ignoring_nones(nt_subclass)

            # Is a sub-branch namedtuple
            working_data[prefix] = single_type.from_db_strings(without_prefix_data)

        else:
            # Should be able to create it directly.
            for name, value in data_sublist:
                base_type = _get_type_ignoring_nones(nt_class._field_types[name])

                if value is None:
                    working_data[name] = None

                elif base_type in _enum_types:
                    # Is an enum, lookup from the dictionary.
                    working_data[name] = base_type[value]

                else:
                    working_data[name] = base_type(value)

    return nt_class(**working_data)


class PolarIndex(typing.NamedTuple):
    R: int
    Th: int
    Z: int

    def to_db_strings(self):
        yield from _nt_to_db_strings(self)

    @classmethod
    def from_db_strings(cls, data):
        return _nt_from_db_strings(cls, data)



class Actuation(enum.Enum):
    direct_pressure = enum.auto()
    rigid_cylinder = enum.auto()
    balloon = enum.auto()
    enforced_displacement_plane = enum.auto()


class Balloon(typing.NamedTuple):
    inner_radius_ratio: float
    overshoot_ratio: float
    foldover_param: float
    divs: PolarIndex

    def to_db_strings(self):
        yield from _nt_to_db_strings(self)

    @classmethod
    def from_db_strings(cls, data):
        return _nt_from_db_strings(cls, data)


class Cylinder(typing.NamedTuple):
    initial_radius_ratio: float
    overshoot_ratio: float
    divs: PolarIndex

    def to_db_strings(self):
        yield from _nt_to_db_strings(self)

    @classmethod
    def from_db_strings(cls, data):
        return _nt_from_db_strings(cls, data)


class StentParams(typing.NamedTuple):
    angle: float
    divs: PolarIndex
    r_min: float
    r_max: float
    length: float
    stent_element_type: element.ElemType
    balloon: typing.Optional[Balloon]
    cylinder: typing.Optional[Cylinder]
    expansion_ratio: typing.Optional[float]

    def to_db_strings(self):
        yield from _nt_to_db_strings(self)

    @classmethod
    def from_db_strings(cls, data):
        return _nt_from_db_strings(cls, data)

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
        if self.stent_element_type == element.ElemType.C3D8R:
            return 3

        elif self.stent_element_type in (element.ElemType.CPS4R, element.ElemType.CPE4R):
            return 2

        else:
            raise ValueError("Invalid element type for stent.", self.stent_element_type)

    @property
    def radial_thickness(self) -> float:
        return self.r_max - self.r_min

    @property
    def radial_midplane_initial(self) -> float:
        return 0.5 * (self.r_max + self.r_min)

    @property
    def theta_arc_initial(self):
        return self.radial_midplane_initial * math.radians(self.angle)


class StentDesign(typing.NamedTuple):
    stent_params: StentParams
    active_elements: typing.FrozenSet[PolarIndex]


def node_to_elem_design_space(divs: PolarIndex) -> PolarIndex:
    return divs._replace(
        R=max(divs.R - 1, 1), # Special case if we're at R=1 - that means plate elements.
        Th=divs.Th - 1,
        Z=divs.Z - 1)


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


def generate_elem_indices(divs: PolarIndex) -> typing.Iterable[typing.Tuple[int, PolarIndex]]:
    divs_minus_one = node_to_elem_design_space(divs)
    yield from generate_node_indices(divs_minus_one)



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


class GlobalNodeSetNames(enum.Enum):
    BalloonTheta0 = enum.auto()
    BalloonThetaMax = enum.auto()
    BalloonZ0 = enum.auto()
    BalloonZMax = enum.auto()
    RigidCyl = enum.auto()
    PlanarStentTheta0 = enum.auto()
    PlanarStentThetaMax = enum.auto()
    PlanarStentZMin = enum.auto()


dylan_r10n1_params = StentParams(
    angle=60,
    divs=PolarIndex(
        R=1,
        Th=88,  # 31
        Z=800,  # 120
    ),
    r_min=0.65,
    r_max=0.75,
    length=11.0,
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
    expansion_ratio=3.0,
)

basic_stent_params = dylan_r10n1_params._replace(balloon=None, cylinder=None)


def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def _gen_ordinates(n, low, high):
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
    r_vals = _gen_ordinates(stent_params.divs.R, stent_params.r_min, stent_params.r_max)
    th_vals = _gen_ordinates(stent_params.divs.Th, 0.0, stent_params.angle)
    z_vals = _gen_ordinates(stent_params.divs.Z, 0.0, stent_params.length)

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

    x_vals = _gen_ordinates(stent_params.divs.Th, 0.0, stent_params.theta_arc_initial)
    y_vals = _gen_ordinates(stent_params.divs.Z, 0.0, stent_params.length)
    z_vals = _gen_ordinates(stent_params.divs.R, stent_params.radial_midplane_initial, stent_params.radial_midplane_initial)

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


def get_single_element_connection(stent_params: StentParams, i: PolarIndex) -> tuple:
    if stent_params.stent_element_dimensions == 2:
        return get_2d_plate_connection(stent_params.divs, i)

    elif stent_params.stent_element_dimensions == 3:
        return get_c3d8_connection(stent_params.divs, i)

    else:
        raise ValueError(stent_params.stent_element_dimensions)


def generate_stent_part_elements(stent_params: StentParams) -> typing.Iterable[typing.Tuple[PolarIndex, int, element.Element]]:
    if stent_params.stent_element_dimensions == 2:
        yield from generate_plate_elements_all(stent_params.divs, stent_params.stent_element_type)

    elif stent_params.stent_element_dimensions == 3:
        yield from generate_brick_elements_all(stent_params.divs)

    else:
        raise ValueError(stent_params.stent_element_dimensions)


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
        active_elements=frozenset(included_elements)
    )


def _make_design_from_line_segments(stent_params: StentParams, line_z_th_points: typing.List[base.XYZ], width: float, nominal_radius: float) -> StentDesign:

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

        def all_dists():
            for p1, p2 in _pairwise(line_z_th_points):
                yield point_line_dist(p1, p2, this_elem_cent)

        return min(all_dists()) <= (0.5 * width)

    included_elements = (idx for idx, iElem, e in generate_stent_part_elements(stent_params) if check_elem(e))
    return StentDesign(
        stent_params=stent_params,
        active_elements=frozenset(included_elements)
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
    line_z_th_points = [base.RThZ(
        r=nominal_radius,
        theta_deg=th,
        z=z_mid + z_amp * math.sin(2 * math.pi * th / stent_params.angle)).to_xyz()
        for th in th_points
    ]

    return _make_design_from_line_segments(stent_params, line_z_th_points, width, nominal_radius)


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

    xyz_point = [p.to_xyz() for p in all_points]
    return _make_design_from_line_segments(stent_params, xyz_point, width, nominal_radius)


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

    return _make_design_from_line_segments(stent_params, xyz_point, width, nominal_radius)


def make_initial_all_in(stent_params: StentParams) -> StentDesign:
    included_elements = (idx for idx, iElem, e in generate_stent_part_elements(stent_params))
    return StentDesign(
        stent_params=stent_params,
        active_elements=frozenset(included_elements)
    )


make_initial_design = make_initial_straight_edge



def make_design_from_snapshot(stent_params: StentParams, snapshot: "history.Snapshot") -> StentDesign:
    elem_num_to_idx = {iElem: idx for idx, iElem, e in generate_brick_elements_all(divs=stent_params.divs)}
    active_elements_idx = {elem_num_to_idx[iElem] for iElem in snapshot.active_elements}

    return StentDesign(
        stent_params=stent_params,
        active_elements=frozenset(active_elements_idx),
    )

def show_initial_model_test(stent_design: StentDesign):
    from stent_opt.struct_opt import display

    node_positions = {polar_index: xyz.to_xyz() for _, polar_index, xyz in generate_nodes(stent_design.stent_params)}
    poly_list = []
    for face_nodes, val in single_faces_and_vals:
        verts = [node_position[iNode] for iNode in face_nodes]

        poly_list.append( (verts, val))






if __name__ == "__main__":
    stent_design = make_initial_all_in(basic_stent_params)
    show_initial_model_test(stent_design)