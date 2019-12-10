import dataclasses
import typing
import enum

from stent_opt.abaqus_model import interaction_property, surface

class InteractionSurfaces(enum.Enum):
    AllWithSelf = enum.auto()
    DefinedPairs = enum.auto()

T_SurfacePair = typing.Tuple[surface.Surface, surface.Surface]

@dataclasses.dataclass(frozen=True)
class Interaction:
    name: str
    int_property: interaction_property.SurfaceInteraction
    interaction_surface_type: InteractionSurfaces
    surface_pairs: typing.Optional[typing.Tuple[T_SurfacePair]]


    def produce_inp_lines(self) -> typing.Iterable[str]:

        if self.interaction_surface_type != InteractionSurfaces.AllWithSelf:
            raise ValueError("Time to write this code then!")

        yield f"** Interaction: {self.name}"
        yield f"*Contact, op=NEW"
        yield f"*Contact Inclusions, ALL EXTERIOR"
        yield f"*Contact Property Assignment"
        yield f" ,  , {self.int_property.name}"


def make_test_all_interaction() -> Interaction:
    return Interaction(
        name="IntAAA",
        int_property=interaction_property.make_test_int_prop(),
        interaction_surface_type=InteractionSurfaces.AllWithSelf,
        surface_pairs=None,
    )

if __name__ == "__main__":
    one_int = make_test_all_interaction()
    for l in one_int.produce_inp_lines():
        print(l)

