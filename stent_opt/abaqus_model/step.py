import typing
import dataclasses

from stent_opt.abaqus_model import base

FALLBACK_VISC_B1 = 0.2  # Linear Bulk Viscosity Paramter - default in Abaqus is 0.06
FALLBACK_VISC_B2 = 5.0  # Quadratic ... - default in Abaqus is 1.2

@dataclasses.dataclass(frozen=True)
class StepBase:
    name: str  # All materials have to have this!

    def produce_inp_lines(self) -> typing.Iterable[str]:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class StepDynamicExplicit(StepBase):
    name: str
    step_time: float
    bulk_visc_b1: float
    bulk_visc_b2: float

    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield f"*Step, name={self.name}, nlgeom=YES"
        yield "*Dynamic, Explicit"
        yield ", " + base.abaqus_float(self.step_time)
        yield "*Bulk Viscosity"
        yield base.abaqus_float(self.bulk_visc_b1) + ", " + base.abaqus_float(self.bulk_visc_b2)



def make_test_step(n: int) -> StepDynamicExplicit:
    one_step = StepDynamicExplicit(
        name=f"TestStep{n}",
        step_time=123.4,
        bulk_visc_b1=0.06,
        bulk_visc_b2=1.2,
    )

    return one_step


if __name__ == "__main__":
    for l in make_test_step(1).produce_inp_lines():
        print(l)
