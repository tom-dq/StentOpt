import typing
import dataclasses

from stent_opt.abaqus_model import amplitude, base, node, surface


@dataclasses.dataclass(frozen=True)
class LoadBase(base.LoadBoundaryBase):
    pass


@dataclasses.dataclass(frozen=True)
class ConcentratedLoad(LoadBase):
    name: str
    with_amplitude: typing.Optional["amplitude.Amplitude"]
    node_set: node.NodeSet
    axis: int
    value: float

    def produce_inp_lines(self, action: base.Action) -> typing.Iterable[str]:
        yield f"** Name: {self.name}   Type: Concentrated force"
        yield f"*Cload{action.load_new_text()}{self._amplitude_suffix()}"

        if action != base.Action.remove:
            set_name = self.node_set.get_name(base.SetContext.assembly)
            yield f"{set_name}, {self.axis}, {base.abaqus_float(self.value)}"


@dataclasses.dataclass(frozen=True)
class PressureLoad(LoadBase):
    name: str
    with_amplitude: typing.Optional["amplitude.Amplitude"]
    on_surface: surface.Surface
    value: float

    def produce_inp_lines(self, action: base.Action) -> typing.Iterable[str]:
        yield f"** Name: {self.name}   Type: Pressure"
        yield f"*Dsload{action.load_new_text()}{self._amplitude_suffix()}"

        if action != base.Action.remove:
            yield f"{self.on_surface.name}, P, {base.abaqus_float(self.value)}"


def make_test_load_point(node_set: node.NodeSet) -> ConcentratedLoad:
    cload = ConcentratedLoad(
        name="Test Point Load",
        with_amplitude=None,
        node_set=node_set,
        axis=1,
        value=123.456,
    )

    return cload


def make_test_pressure(
    on_surface: surface.Surface, with_amplitude: typing.Optional[amplitude.Amplitude]
) -> PressureLoad:
    pload = PressureLoad(
        name="Test Pressure Load",
        with_amplitude=with_amplitude,
        on_surface=on_surface,
        value=345.678,
    )
    return pload


if __name__ == "__main__":

    cload = make_test_load_point("Set123")
    pload = make_test_load_point("Set123")
    for action in base.Action:
        for l in cload.produce_inp_lines(action):
            print(l)

        print()
        for l in pload.produce_inp_lines(action):
            print(l)

        print()
