import typing
import dataclasses

from abaqus_model import base

@dataclasses.dataclass(frozen=True)
class MaterialBase:
    name: str  # All materials have to have this!

    def produce_inp_lines(self) -> typing.Iterable[str]:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class MaterialElastic(MaterialBase):
    name: str
    density: float
    elast_mod: float
    elast_possion: float

    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield f"*Material, name={self.name}"
        yield "*Density"
        yield base.abaqus_float(self.density) + ","
        yield "*Elastic"
        yield base.abaqus_float(self.elast_mod) + ", " + base.abaqus_float(self.elast_possion)


