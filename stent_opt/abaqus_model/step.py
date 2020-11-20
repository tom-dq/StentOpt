import typing
import dataclasses

from stent_opt.abaqus_model import base

FALLBACK_VISC_B1 = 0.2  # Linear Bulk Viscosity Paramter - default in Abaqus is 0.06
FALLBACK_VISC_B2 = 5.0  # Quadratic ... - default in Abaqus is 1.2


@dataclasses.dataclass(frozen=True)
class StepBase:
    name: str  # All materials have to have this!
    step_time: float

    def produce_inp_lines(self) -> typing.Iterable[str]:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class StepDynamicImplicit(StepBase):
    name: str
    step_time: float
    step_init: float = 0.1
    step_minimum: float = 1e-7
    max_incs: int = 100_000

    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield f"*Step, name={self.name}, nlgeom=YES, inc={self.max_incs}"
        yield "*Dynamic"
        step_data = [self.step_init, self.step_time, self.step_minimum]
        yield ", ".join(base.abaqus_float(x) for x in step_data)


@dataclasses.dataclass(frozen=True)
class StepDynamicExplicit(StepBase):
    name: str
    step_time: float
    bulk_visc_b1: float = FALLBACK_VISC_B1
    bulk_visc_b2: float = FALLBACK_VISC_B2

    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield f"*Step, name={self.name}, nlgeom=YES"
        yield "*Dynamic, Explicit"
        yield ", " + base.abaqus_float(self.step_time)
        yield "*Bulk Viscosity"
        yield base.abaqus_float(self.bulk_visc_b1) + ", " + base.abaqus_float(self.bulk_visc_b2)


def is_explicit(step: typing.Union[StepBase, typing.Type[StepBase]]) -> bool:
    """Should work for the dataclass or an instance thereof."""
    if isinstance(step, StepDynamicImplicit) or step == StepDynamicImplicit:
        return False

    elif isinstance(step, StepDynamicExplicit) or step == StepDynamicExplicit:
        return True

    else:
        raise ValueError(step)


def analysis_is_explicit(steps: typing.List[StepBase]) -> bool:
    if not steps:
        raise ValueError("No step!")

    opts = {is_explicit(step) for step in steps}
    if len(opts) != 1:
        raise ValueError("Ambiguous steps")

    return opts.pop()

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

    aaa = StepDynamicExplicit("aaa", 1.2)
    bbb = StepDynamicImplicit("bbb", 1.4)

    for x in [aaa, StepDynamicExplicit, bbb, StepDynamicImplicit]:
        print(x, is_explicit(x))
