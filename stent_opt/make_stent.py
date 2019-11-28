import itertools
import typing



from abaqus_model import base, node, element, part, material


class StentParams(typing.NamedTuple):
    angle: float
    divs_radial: int
    divs_theta: int
    divs_z: int
    r_min: float
    r_max: float


basic_stent_params = StentParams(
    angle=60,
    divs_radial=5,
    divs_theta=4,
    divs_z=6,
    r_min=10,
    r_max=15,
)
if __name__ == "__main__":
    pass


def generate_nodes(stent_params: StentParams) -> typing.Iterable[base.XYZ]:
    def gen_vals(n, low, high):
        vals = [ (high - low) + low * (i/(n-1)) for i in range(n)]
        # TODO!

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

