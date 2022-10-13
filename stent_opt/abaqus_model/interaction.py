import dataclasses
import typing
import enum

from stent_opt.abaqus_model import base, interaction_property, surface

T_SurfacePair = typing.Tuple[surface.Surface, surface.Surface]


class MechanicalConstraint(enum.Enum):
    Kinematic = "KINEMATIC"
    Penalty = "PENALTY"


@dataclasses.dataclass(frozen=True)
class InteractionBase(base.SortableDataclass):
    name: str
    int_property: interaction_property.SurfaceInteraction

    def produce_inp_lines(self) -> typing.Iterable[str]:
        raise NotImplementedError

    @property
    def is_step_dependent(self) -> bool:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class GeneralContact(InteractionBase):
    """Either general all exterior (with included_surface_pairs Falsey) or pairs of surfaces."""

    name: str
    int_property: interaction_property.SurfaceInteraction
    included_surface_pairs: typing.Optional[typing.Tuple[T_SurfacePair]]

    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield f"** Interaction: {self.name}"
        yield f"*Contact, op=NEW"
        if self.included_surface_pairs:
            yield "*Contact Inclusions"
            for s1, s2 in self.included_surface_pairs:
                yield s1.name + " , " + s2.name

        else:
            yield f"*Contact Inclusions, ALL EXTERIOR"

        yield f"*Contact Property Assignment"
        yield f" ,  , {self.int_property.name}"

    @property
    def is_step_dependent(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class ContactPair(InteractionBase):
    name: str
    int_property: interaction_property.SurfaceInteraction
    mechanical_constraint: MechanicalConstraint
    surface_pair: T_SurfacePair

    def produce_inp_lines(self) -> typing.Iterable[str]:
        master_surf, slave_surf = self.surface_pair

        yield f"** Interaction: {self.name}"
        yield f"*Contact Pair, interaction={self.int_property.name}, mechanical constraint={self.mechanical_constraint.value}, cpset={self.name}"
        yield f"{master_surf.name}, {slave_surf.name}"

    @property
    def is_step_dependent(self) -> bool:
        return True


def make_test_all_interaction() -> GeneralContact:
    return GeneralContact(
        name="IntAAA",
        int_property=interaction_property.make_test_int_prop(),
    )


if __name__ == "__main__":
    one_int = make_test_all_interaction()
    for l in one_int.produce_inp_lines():
        print(l)
