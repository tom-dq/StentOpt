import dataclasses
import typing

from stent_opt.abaqus_model import base, material, element


@dataclasses.dataclass(frozen=True)
class HourglassControlEnhanced:
    name: str

    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield f"*SECTION CONTROLS, NAME={self.name}, HOURGLASS=ENHANCED"
        yield "1., 1., 1."


@dataclasses.dataclass(frozen=True)
class SectionBase:
    name: str
    mat: material.MaterialBase
    enhanced_hourglass: typing.Optional[HourglassControlEnhanced]

    def produce_inp_lines(self, elset: element.ElementSet):
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class SolidSection(SectionBase):
    name: str
    mat: material.MaterialBase
    thickness: typing.Optional[float]

    def produce_inp_lines(self, elset: element.ElementSet) -> typing.Iterable[str]:
        yield f"** Section: {self.name}"

        if self.enhanced_hourglass:
            yield f"*Solid Section, elset={elset.get_name(base.SetContext.part)}, controls={self.enhanced_hourglass.name}, material={self.mat.name}"
        else:
            yield f"*Solid Section, elset={elset.get_name(base.SetContext.part)}, material={self.mat.name}"

        if self.thickness is not None:
            yield base.abaqus_float(self.thickness) + ", "


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
    mat: typing.Optional[material.MaterialBase]
    surf_density: typing.Optional[float]

    def produce_inp_lines(self, elset: element.ElementSet) -> typing.Iterable[str]:
        yield f"** Section: {self.name}"

        maybe_density = (
            f", DENSITY={base.abaqus_float(self.surf_density)}"
            if self.surf_density is not None
            else ""
        )
        yield f"*Surface Section, elset={elset.get_name(base.SetContext.part)}{maybe_density}"
