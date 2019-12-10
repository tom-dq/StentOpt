import typing
import dataclasses

from stent_opt.abaqus_model import base

class Point(typing.NamedTuple):
    stress: float
    strain: float


@dataclasses.dataclass(frozen=True)
class MaterialBase:
    name: str  # All materials have to have this!

    def produce_inp_lines(self) -> typing.Iterable[str]:
        raise NotImplementedError()

    def produce_inp_lines_section(self, section_name, elset_name) -> typing.Iterable[str]:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class MaterialElasticPlastic(MaterialBase):
    name: str
    density: float
    elast_mod: float
    elast_possion: float
    plastic: typing.Optional[typing.Tuple[Point, ...]]

    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield f"*Material, name={self.name}"
        yield "*Density"
        yield base.abaqus_float(self.density) + ","
        yield "*Elastic"
        yield base.abaqus_float(self.elast_mod) + ", " + base.abaqus_float(self.elast_possion)
        if self.plastic:
            yield "*Plastic"
            for point in self.plastic:
                yield ", ".join( [base.abaqus_float(point.stress), base.abaqus_float(point.strain)] )




@dataclasses.dataclass(frozen=True)
class MaterialHyperElasticMembrane(MaterialBase):
    name: str
    density: float
    order: int
    coeffs: typing.Tuple[float, ...]
    thickness: float
    elast_possion: float

    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield f"*Material, name={self.name}"
        yield "*Density"
        yield base.abaqus_float(self.density) + ","
        yield f"*Hyperelastic, n={self.order}"
        yield ', '.join(str(one_c) for one_c in self.coeffs)


    def produce_inp_lines_section(self, section_name, elset_name) -> typing.Iterable[str]:
        yield f"** Section: {section_name}"
        yield f"*Membrane Section, elset={elset_name}, material={self.name}, poisson={base.abaqus_float(self.elast_possion)}"
        yield f"{base.abaqus_float(self.thickness)}, "

