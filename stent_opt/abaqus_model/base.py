import abc
import dataclasses
import hashlib
import typing
import enum
import math

# from stent_opt.abaqus_model import part

class XYZ(typing.NamedTuple):
    x: float
    y: float
    z: float

DOFs = (1, 2, 3)


class RThZ(typing.NamedTuple):
    r: float
    theta_deg: float
    z: float

    def to_xyz(self) -> XYZ:
        ang_rad = math.radians(self.theta_deg)
        x = self.r * math.sin(ang_rad)
        y = self.z
        z = self.r * math.cos(ang_rad)
        return XYZ(x, y, z)


class SetContext(enum.Enum):
    part = enum.auto()
    assembly = enum.auto()


class SetType(enum.Enum):
    node = enum.auto()
    element = enum.auto()

    def set_name_key(self) -> str:
        """Either "Elset" or "Nset"""

        if self == SetType.node:
            return "Nset"

        elif self == SetType.element:
            return "Elset"

        else:
            raise ValueError(self)



@dataclasses.dataclass(frozen=True)
class SetBase:
    part: "part.Part"
    name_component: str

    @abc.abstractmethod
    def _entity_numbers(self) -> typing.FrozenSet[int]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def set_type(self) -> SetType:
        raise NotImplementedError()

    def get_name(self, set_context: SetContext) -> str:
        if set_context == SetContext.part:
            node_number_hash = str(hash(self._entity_numbers()))
            return f"Set_{self.name_component}_{deterministic_key(self.part, node_number_hash)}"

        elif set_context == SetContext.assembly:
            name_in_part = self.get_name(SetContext.part)
            return f"{self.part.name}.{name_in_part}"

        else:
            raise ValueError(set_context)

    def generate_inp_lines(self, set_context: SetContext) -> typing.Iterable[str]:
        """Generateds the *Nset or *Elset lines"""
        fully_populated_sequence = frozenset(range(min(self._entity_numbers()), max(self._entity_numbers())+1))
        is_sequential = self._entity_numbers() == fully_populated_sequence
        if is_sequential:
            key = self.set_type.set_name_key()
            yield f"*{key.title()}, {key.lower()}={self.get_name(set_context)}, generate"
            yield f"  {min(self._entity_numbers())},  {max(self._entity_numbers())},   1"

        else:
            raise ValueError("Time to write this code I guess!")


def abaqus_float(x) -> str:
    """Returns a float Abaqus can parse (i.e., with a dot in it)"""

    maybe_float = str(x)
    if "e" in maybe_float.lower():
        if "." not in maybe_float:
            raise ValueError(maybe_float)

    else:
        if "." not in maybe_float:
            return maybe_float + ".0"

    return maybe_float


def inp_heading(text: str) -> typing.Iterable[str]:
    yield "**"
    yield f"** {text}"
    yield "**"


def deterministic_key(class_instance, text) -> str:
    raw_text = f"{class_instance}_{text}"
    return "Z_" + hashlib.md5(raw_text.encode()).hexdigest()[0:8]