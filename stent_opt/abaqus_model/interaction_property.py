import dataclasses
import typing

from stent_opt.abaqus_model import base


@dataclasses.dataclass(frozen=True)
class SurfaceInteraction:
    name: str
    lateral_friction: float

    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield f"*Surface Interaction, name={self.name}"
        yield "*Friction"
        yield base.abaqus_float(self.lateral_friction) + ","

        """** 
** INTERACTION PROPERTIES
** """


def make_test_int_prop() -> SurfaceInteraction:
    return SurfaceInteraction(
        name="SurfInt1",
        lateral_friction=0.0,
    )


if __name__ == "__main__":
    surf_int = make_test_int_prop()
    for l in surf_int.produce_inp_lines():
        print(l)
