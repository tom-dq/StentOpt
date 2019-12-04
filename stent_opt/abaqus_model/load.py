import typing
import dataclasses
import enum

from stent_opt.abaqus_model import amplitude, base, node, surface


class Action(enum.Enum):
    create_first_step = enum.auto()
    create_subsequent_step = enum.auto()
    remove = enum.auto()

    def load_new_text(self) -> str:
        if self == Action.create_first_step:
            return ""

        elif self in (Action.create_subsequent_step, Action.remove):
            return ", op=NEW"

        else:
            raise ValueError


@dataclasses.dataclass(frozen=True)
class LoadBase:
    name: str  # All loads have to have this!
    amplitude: typing.Optional["amplitude.Amplitude"]

    def produce_inp_lines(self, action: Action) -> typing.Iterable[str]:
        raise NotImplementedError()

    def sortable(self):
        type_name = self.__class__.__name__
        return (type_name, self)

    def _amplitude_suffix(self) -> str:
        if self.amplitude:
            return f", amplitude={self.amplitude.name}"

        else:
            return ""

@dataclasses.dataclass(frozen=True)
class ConcentratedLoad(LoadBase):
    name: str
    amplitude: typing.Optional["amplitude.Amplitude"]
    node_set: node.NodeSet
    axis: int
    value: float

    def produce_inp_lines(self, action: Action) -> typing.Iterable[str]:
        yield f"** Name: {self.name}   Type: Concentrated force"
        yield f"*Cload{action.load_new_text()}{self._amplitude_suffix()}"

        if action != Action.remove:
            set_name = self.node_set.get_name(base.SetContext.assembly)
            yield f"{set_name}, {self.axis}, {base.abaqus_float(self.value)}"


@dataclasses.dataclass(frozen=True)
class PressureLoad(LoadBase):
    name: str
    amplitude: typing.Optional["amplitude.Amplitude"]
    on_surface: surface.Surface
    value: float

    def produce_inp_lines(self, action: Action) -> typing.Iterable[str]:
        yield f"** Name: {self.name}   Type: Pressure"
        yield f"*Dsload{action.load_new_text()}{self._amplitude_suffix()}"

        if action != Action.remove:
            yield f"{self.on_surface.name}, P, {base.abaqus_float(self.value)}"


def make_test_load_point(node_set: node.NodeSet) -> ConcentratedLoad:
    cload = ConcentratedLoad(
        name="Test Point Load",
        amplitude=None,
        node_set=node_set,
        axis=1,
        value=123.456,
    )

    return cload


def make_test_pressure(on_surface: surface.Surface, with_amplitude: typing.Optional[amplitude.Amplitude]) -> PressureLoad:
    pload = PressureLoad(
        name="Test Pressure Load",
        amplitude=with_amplitude,
        on_surface=on_surface,
        value=345.678,
    )
    return pload

if __name__ == "__main__":

    cload = make_test_load_point("Set123")
    pload = make_test_load_point("Set123")
    for action in Action:
        for l in cload.produce_inp_lines(action):
            print(l)

        print()
        for l in pload.produce_inp_lines(action):
            print(l)

        print()


