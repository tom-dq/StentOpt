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
    divs_radial=5,
    divs_theta=4,
    divs_z=6,
    r_min=10,
    r_max=15,
    length=8.0,
)
if __name__ == "__main__":
    pass


def generate_nodes(stent_params: StentParams) -> typing.Iterable[base.XYZ]:
    def gen_ordinates(n, low, high):
        vals = [ low + (high - low) * (i/(n-1)) for i in range(n)]
        return vals

    r_vals = gen_ordinates(stent_params.divs_z, stent_params.r_min, stent_params.r_max)
    th_vals = gen_ordinates(stent_params.divs_theta, 0.0, stent_params.angle)
    z_vals = gen_ordinates(stent_params.divs_z, 0.0, stent_params.length)

    for r, th, z in itertools.product(r_vals, th_vals, z_vals):
        polar_coord = base.RThZ(r, th, z)
        yield polar_coord.to_xyz()


def generate_elements(stent_params: StentParams) -> typing.Iterable[element.Element]:
    for iNode, (iR, iTh, iZ) in enumerate(itertools.product(
            range(stent_params.divs_radial-1),
            range(stent_params.divs_theta-1),
            range(stent_params.divs_z-1)), start=1):

        connection_this_r = [
            iNode,
            iNode+1,
            iNode + stent_params.divs_z + 1,
            iNode + stent_params.divs_z,
        ]
        connection_next_r = [i + stent_params.divs_z * stent_params.divs_theta for i in connection_this_r]
        connection = connection_this_r + connection_next_r
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
            print(l)
            fOut.write(l + "\n")


if __name__ == "__main__":
    make_a_stent()