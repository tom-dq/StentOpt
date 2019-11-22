import typing
import dataclasses
import enum

from abaqus_model import base


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

    def produce_inp_lines(self, action: Action) -> typing.Iterable[str]:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class ConcentratedLoad(LoadBase):
    name: str
    set_name: str
    axis: int
    value: float

    def produce_inp_lines(self, action: Action) -> typing.Iterable[str]:
        yield f"** Name: {self.name}   Type: Concentrated force"
        yield f"*Cload{action.load_new_text()}"

        if action != Action.remove:
            yield f"{self.set_name}, {self.axis}, {base.abaqus_float(self.value)}"


@dataclasses.dataclass(frozen=True)
class PressureLoad(LoadBase):
    name: str
    set_name: str
    value: float

    def produce_inp_lines(self, action: Action) -> typing.Iterable[str]:
        yield f"** Name: {self.name}   Type: Pressure"
        yield f"*Dsload{action.load_new_text()}"

        if action != Action.remove:
            yield f"{self.set_name}, P, {base.abaqus_float(self.value)}"


def make_test_load_point(set_name) -> ConcentratedLoad:
    cload = ConcentratedLoad(
        name="Test Point Load",
        set_name=set_name,
        axis=1,
        value=123.456,
    )

    return cload


def make_test_pressure(set_name) -> PressureLoad:
    pload = PressureLoad(
        name="Test Pressure Load",
        set_name=set_name,
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


