import itertools
import typing
import functools


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
    divs_radial=5,
    divs_theta=6,
    divs_z=7,
    r_min=10,
    r_max=15,
    length=8.0,
)

def generate_nodes(stent_params: StentParams) -> typing.Iterable[base.XYZ]:
    def gen_ordinates(n, low, high):
        vals = [ low + (high - low) * (i/(n-1)) for i in range(n)]
        return vals

    r_vals = gen_ordinates(stent_params.divs_radial, stent_params.r_min, stent_params.r_max)
    th_vals = gen_ordinates(stent_params.divs_theta, 0.0, stent_params.angle)
    z_vals = gen_ordinates(stent_params.divs_z, 0.0, stent_params.length)

    for r, th, z in itertools.product(r_vals, th_vals, z_vals):
        polar_coord = base.RThZ(r, th, z)
        yield polar_coord.to_xyz()


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

    for iNode, one_node in enumerate(generate_nodes(stent_params=basic_stent_params), start=1):
        stent_part.add_node_validated(iNode, one_node)

    for iElem, one_elem in enumerate(generate_elements(stent_params=basic_stent_params), start=1):
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