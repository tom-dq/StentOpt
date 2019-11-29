import abc
import dataclasses
import hashlib
import itertools
import typing
import enum
import math
import operator

# from stent_opt.abaqus_model import part

T = typing.Type["T"]

def _operator_dot(op: typing.Callable, one: T, two: T):
    """Maps an operator to the components of the class.
    e.g., A(a=3, b=5) + A(a=56, b=67)  => A(3+56, b=5+67)"""

    # Special case for 0 + T -> T (to make the "sum" builtin work)
    if one == 0:
        return two

    if one.__class__ != two.__class__:
        raise TypeError(f"Can't apply {op} to {one} and {two}.")

    field_dict = {
        f: op(getattr(one, f), getattr(two, f)) for f in one._fields
    }

    return type(one)(**field_dict)

def _operator_splat_const_first(op: typing.Callable, a, one: T):
    """Splats an operator across the components of the class.
    e.g., 2 * A(a=3, b=5)  => A(2 * 3, b=2 * 5)"""

    field_dict = {
        f: op(a, getattr(one, f)) for f in one._fields
    }

    return type(one)(**field_dict)


def _operator_splat_const_last(op: typing.Callable, one: T, a):
    """Splats an operator across the components of the class.
    e.g., A(a=3, b=5) / 3.4  => A(3 / 3.4, b=5 / 3.4)"""

    field_dict = {
        f: op(getattr(one, f), a) for f in one._fields
    }

    return type(one)(**field_dict)


def _groups_of(items, n):
    working_list = []
    still_going = True
    it = iter(items)
    while still_going:
        try:
            working_list.append(next(it))
        except StopIteration:
            still_going = False

        if len(working_list) >= n:
            yield working_list
            working_list = []

    if working_list:
        yield working_list


class XYZ(typing.NamedTuple):
    x: float
    y: float
    z: float

    def __add__(self, other):
        return _operator_dot(operator.add, self, other)

    def __radd__(self, other):
        return _operator_dot(operator.add, other, self)

    def __sub__(self, other):
        return _operator_dot(operator.sub, self, other)

    def __rmul__(self, other):
        return _operator_splat_const_first(operator.mul, other, self)

    def __truediv__(self, other):
        return _operator_splat_const_last(operator.truediv, self, other)

    def __abs__(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

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

    def __add__(self, other):
        return _operator_dot(operator.add, self, other)

    def __radd__(self, other):
        return _operator_dot(operator.add, other, self)

    def __sub__(self, other):
        return _operator_dot(operator.sub, self, other)

    def __rmul__(self, other):
        return _operator_splat_const_first(operator.mul, other, self)

    def __truediv__(self, other):
        return _operator_splat_const_last(operator.truediv, self, other)


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

        key = self.set_type.set_name_key()
        MAX_PER_LINE = 10

        fully_populated_sequence = frozenset(range(min(self._entity_numbers()), max(self._entity_numbers())+1))
        is_sequential = self._entity_numbers() == fully_populated_sequence
        if is_sequential:

            yield f"*{key.title()}, {key.lower()}={self.get_name(set_context)}, generate"
            yield f"  {min(self._entity_numbers())},  {max(self._entity_numbers())},   1"

        else:
            yield f"*{key.title()}, {key.lower()}={self.get_name(set_context)}"
            for sub_list in _groups_of(sorted(self._entity_numbers()), MAX_PER_LINE):
                yield ", ".join(str(x) for x in sub_list)


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



if __name__ == "__main__":
    a1 = RThZ(r=2.3, theta_deg=34.1, z=5.6)
    a2 = RThZ(r=1, theta_deg=2, z=3)
    print(a1)
    print(a1+a2)

    print(-1 * a1)
    print(a1 / 4.5)

    a = [1, 2, 3, 4, 5]

    for n in (2, 3, 4, 5, 6):
        print(n)
        for x in _groups_of(a, n):
            print(x)

        print()
