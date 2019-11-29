import itertools
import typing


from stent_opt.abaqus_model import base, node, element, part, material, instance, main


class StentParams(typing.NamedTuple):
    angle: float
    divs_radial: int
    divs_theta: int
    divs_z: int
    r_min: float
    r_max: float
    length: float

basic_stent_params = StentParams(
    angle=60,
    divs_radial=30,
    divs_theta=40,
    divs_z=60,
    r_min=10,
    r_max=15,
    length=20.0,
)

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


def node_from_index(stent_params: StentParams, iR, iTh, iZ):
    """Node numbers (fully populated)"""
    return (iR * (stent_params.divs_theta * stent_params.divs_z) +
            iTh * stent_params.divs_z +
            iZ +
            1)  # One based node numbers!

def elem_from_index(stent_params: StentParams, iR, iTh, iZ):
    """Element numbers (fully populated) - just one fewer number in each ordinate."""
    stent_params_minus_one = stent_params._replace(
        divs_radial=stent_params.divs_radial - 1,
        divs_theta=stent_params.divs_theta - 1,
        divs_z=stent_params.divs_z - 1)

    return node_from_index(stent_params_minus_one, iR, iTh, iZ)


def generate_elements(stent_params: StentParams) -> typing.Iterable[element.Element]:

    # get_iNode = functools.partial(node_from_index, stent_params=stent_params)

    def get_iNode(iR, iTh, iZ):
        return node_from_index(stent_params, iR, iTh, iZ)

    for iR, iTh, iZ in itertools.product(
            range(stent_params.divs_radial-1),
            range(stent_params.divs_theta-1),
            range(stent_params.divs_z-1)):

        connection = [
            get_iNode(iR, iTh, iZ),
            get_iNode(iR, iTh+1, iZ),
            get_iNode(iR, iTh+1, iZ+1),
            get_iNode(iR, iTh, iZ+1),
            get_iNode(iR+1, iTh, iZ),
            get_iNode(iR+1, iTh + 1, iZ),
            get_iNode(iR+1, iTh + 1, iZ + 1),
            get_iNode(iR+1, iTh, iZ + 1),
        ]

        yield element.Element(
            name="C3D8",
            connection=tuple(connection),
        )

def make_a_stent():
    common_material = material.MaterialElastic(
        name="ElasticMaterial",
        density=1.0,
        elast_mod=200.0,
        elast_possion=0.3,
    )

    stent_part = part.Part(
        name="Stent",
        common_material=common_material,
    )

    nodes_polar = {
        iNode: n_p for iNode, n_p in enumerate(generate_nodes_polar(stent_params=basic_stent_params), start=1)
    }

    elems_all = {
        iElem: e for iElem, e in enumerate(generate_elements(stent_params=basic_stent_params), start=1)
    }

    def elem_cent_polar(e: element.Element) -> base.RThZ:
        node_pos = [nodes_polar[iNode] for iNode in e.connection]
        ave_node_pos = sum(node_pos) / len(node_pos)
        return ave_node_pos

    def include_elem_decider() -> typing.Callable:
        """As a first test, make some cavities in the segment."""

        rs = [0.5 * (basic_stent_params.r_min + basic_stent_params.r_max),]
        thetas = [0.0, basic_stent_params.angle]
        z_vals = [0, 0.5*basic_stent_params.length, basic_stent_params.length]

        cavities = []
        for r, th, z in itertools.product(rs, thetas, z_vals):
            cavities.append( (base.RThZ(r=r, theta_deg=th, z=z).to_xyz(), 3.5))

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

    should_include_elem = include_elem_decider()

    for iElem, one_elem in elems_all.items():
        #print(one_elem, elem_cent_polar(one_elem))
        if should_include_elem(one_elem):
            stent_part.add_element_validate(iElem, one_elem)

    one_instance = instance.Instance(base_part=stent_part)

    model = main.AbaqusModel("StentModel")
    model.add_instance(one_instance)

    with open(r"c:\temp\stent.inp", "w") as fOut:
        for l in model.produce_inp_lines():
            # print(l)
            fOut.write(l + "\n")


if __name__ == "__main__":
    make_a_stent()