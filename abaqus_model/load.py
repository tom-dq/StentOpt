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
    node_set_name: str
    axis: int
    value: float

    def produce_inp_lines(self, action: Action) -> typing.Iterable[str]:
        yield f"** Name: {self.name}  Type: Concentrated force"
        yield f"*Cload{action.load_new_text()}"

        if action != Action.remove:
            yield f"{self.node_set_name}, {self.axis}, {base.abaqus_float(self.value)}"

