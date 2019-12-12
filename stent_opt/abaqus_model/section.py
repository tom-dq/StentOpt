import dataclasses
import typing

from stent_opt.abaqus_model import base, material, element


@dataclasses.dataclass(frozen=True)
class SectionBase:
    name: str
    mat: material.MaterialBase

    def produce_inp_lines(self, elset: element.ElementSet):
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class SolidSection(SectionBase):
    name: str
    mat: material.MaterialBase

    def produce_inp_lines(self, elset: element.ElementSet) -> typing.Iterable[str]:
        yield f"** Section: {self.name}"
        yield f"*Solid Section, elset={elset.get_name(base.SetContext.part)}, material={self.mat.name}"


@dataclasses.dataclass(frozen=True)
class MembraneSection(SectionBase):
    name: str
    mat: material.MaterialBase
    thickness: float

    def produce_inp_lines(self, elset: element.ElementSet) -> typing.Iterable[str]:
        yield f"** Section: {self.name}"
        yield f"*Membrane Section, elset={elset.get_name(base.SetContext.part)}, material={self.mat.name}"
        yield base.abaqus_float(self.thickness) + ", "


@dataclasses.dataclass(frozen=True)
class SurfaceSection(SectionBase):
    name: str
    surf_density: typing.Optional[float]

    def produce_inp_lines(self, elset: element.ElementSet) -> typing.Iterable[str]:
        yield f"** Section: {self.name}"

        maybe_density = f", {base.abaqus_float(self.surf_density)}" if self.surf_density is not None else ""
        yield f"*Surface Section, elset={elset.get_name(base.SetContext.part)}{maybe_density}"


    @property
    def mat(self):
        return None
