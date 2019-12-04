import itertools
import typing
import collections
import statistics

from stent_opt.abaqus_model import base, amplitude, step, element, part, material
from stent_opt.abaqus_model import instance, main, surface, load


class GlobalSurfNames:
    INNER_SURFACE = "InAll"
    INNER_BOTTOM_HALF = "InLowHalf"
    INNER_MIN_RADIUS = "InMinR"

class StentParams(typing.NamedTuple):
    angle: float
    divs_radial: int
    divs_theta: int
    divs_z: int
    r_min: float
    r_max: float
    length: float


class Index(typing.NamedTuple):
    R: int
    Th: int
    Z: int

dylan_r10n1_params = StentParams(
    angle=60,
    divs_radial=6,
    divs_theta=30,
    divs_z=250,
    r_min=0.65,
    r_max=0.75,
    length=11.0,
)

basic_stent_params = dylan_r10n1_params

def generate_nodes_polar(stent_params: StentParams) -> typing.Iterable[base.RThZ]:
    def gen_ordinates(n, low, high):
        vals = [ low + (high - low) * (i/(n-1)) for i in range(n)]
        return vals

    r_vals = gen_ordinates(stent_params.divs_radial, stent_params.r_min, stent_params.r_max)
    th_vals = gen_ordinates(stent_params.divs_theta, 0.0, stent_params.angle)
    z_vals = gen_ordinates(stent_params.divs_z, 0.0, stent_params.length)

    for r, th, z in itertools.product(r_vals, th_vals, z_vals):
        polar_coord = base.RThZ(r, th, z)
        yield polar_coord


def _stent_params_node_to_elem_idx(stent_params: StentParams) -> StentParams:
    return stent_params._replace(
        divs_radial=stent_params.divs_radial - 1,
        divs_theta=stent_params.divs_theta - 1,
        divs_z=stent_params.divs_z - 1)

def node_from_index(stent_params: StentParams, iR, iTh, iZ):
    """Node numbers (fully populated)"""
    return (iR * (stent_params.divs_theta * stent_params.divs_z) +
            iTh * stent_params.divs_z +
            iZ +
            1)  # One based node numbers!

def elem_from_index(stent_params: StentParams, iR, iTh, iZ):
    """Element numbers (fully populated) - just one fewer number in each ordinate."""
    stent_params_minus_one = _stent_params_node_to_elem_idx(stent_params)
    return node_from_index(stent_params_minus_one, iR, iTh, iZ)


def generate_node_indices(stent_params: StentParams) -> typing.Iterable[typing.Tuple[int, Index]]:
    for iElem, (iR, iTh, iZ) in enumerate(itertools.product(
            range(stent_params.divs_radial),
            range(stent_params.divs_theta),
            range(stent_params.divs_z)), start=1):

        yield iElem, Index(R=iR, Th=iTh, Z=iZ)

def generate_elem_indices(stent_params: StentParams) -> typing.Iterable[typing.Tuple[int, Index]]:
    stent_params_minus_one = _stent_params_node_to_elem_idx(stent_params)
    yield from generate_node_indices(stent_params_minus_one)


def generate_elements_all(stent_params: StentParams) -> typing.Iterable[typing.Tuple[int, element.Element]]:

    def get_iNode(iR, iTh, iZ):
        return node_from_index(stent_params, iR, iTh, iZ)

    for iElem, i in generate_elem_indices(stent_params):
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
            name="C3D8",
            connection=tuple(connection),
        )

def make_a_stent():
    stress_strain_table = (
        material.Point(stress=205., strain=0.0),
        material.Point(stress=515., strain=0.6),
    )
    common_material = material.MaterialElasticPlastic(
        name="Steel",
        density=7.85e-09,
        elast_mod=196000.0,
        elast_possion=0.3,
        plastic=stress_strain_table,
    )

    stent_part = part.Part(
        name="Stent",
        common_material=common_material,
    )

    nodes_polar = {
        iNode: n_p for iNode, n_p in enumerate(generate_nodes_polar(stent_params=basic_stent_params), start=1)
    }

    elems_all = {iElem: e for iElem, e in generate_elements_all(stent_params=basic_stent_params)}

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

    for iNode, one_node_polar in nodes_polar.items():
        stent_part.add_node_validated(iNode, one_node_polar.to_xyz())

    should_include_elem = include_elem_decider_cavities()

    for iElem, one_elem in elems_all.items():
        #print(one_elem, elem_cent_polar(one_elem))
        if should_include_elem(one_elem):
            stent_part.add_element_validate(iElem, one_elem)

    one_instance = instance.Instance(base_part=stent_part)

    model = main.AbaqusModel("StentModel")
    model.add_instance(one_instance)

    return model


def create_surfaces(model: main.AbaqusModel):
    """Sets up the standard surfaces on the model."""

    stent_instance = model.get_only_instance()
    stent_part = stent_instance.base_part

    elem_indexes_all = {iElem: i for iElem, i in generate_elem_indices(stent_params=basic_stent_params)}
    elem_indexes = {iElem: i for iElem, i in elem_indexes_all.items() if iElem in stent_part.elements}

    def create_elem_surface_S1(elem_nums, name):
        elem_set_name = f"ElemSet_{name}"
        the_elements_raw = {iElem: stent_part.elements[iElem] for iElem in elem_nums}
        elem_set = element.ElementSet(stent_part, elem_set_name, element.Elements(the_elements_raw))
        data = [
            (elem_set, surface.SurfaceFace.S1)
        ]
        the_surface = surface.Surface(name, data)

        stent_part.add_element_set(elem_set)
        stent_instance.add_surface(the_surface)

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
        amplitude.XY(3.25, 0.32),
        amplitude.XY(3.75, 1),
        amplitude.XY(4, 0.0),
    )

    amp = amplitude.Amplitude("Amp-1", amp_data)
    surf = model.get_only_instance().surfaces[GlobalSurfNames.INNER_BOTTOM_HALF]

    one_load = load.PressureLoad(
        name="Pressure",
        amplitude=amp,
        on_surface=surf,
        value=1.0,
    )

    model.add_load_specific_steps([one_step], one_load)


def couple_boundaries(model: main.AbaqusModel):
    """Find the node pairs and couples them."""

    stent_instance = model.get_only_instance()
    stent_part = stent_instance.base_part

    def gen_active_pairs():
        nodes_on_bound = {
            iNode: i for iNode, i in generate_node_indices(basic_stent_params)
            if (i.Th == 0 or i.Th == basic_stent_params.divs_theta-1)
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



def write_model(model: main.AbaqusModel):
    with open(r"c:\temp\aba\stent-6.inp", "w") as fOut:
        for l in model.produce_inp_lines():
            fOut.write(l + "\n")


if __name__ == "__main__":
    model = make_a_stent()
    create_surfaces(model)
    apply_loads(model)
    couple_boundaries(model)
    write_model(model)
