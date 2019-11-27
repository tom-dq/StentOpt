import hashlib
import typing
import enum

class XYZ(typing.NamedTuple):
    x: float
    y: float
    z: float

DOFs = (1, 2, 3)


class SetContext(enum.Enum):
    part = enum.auto()
    assembly = enum.auto()



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
    return hashlib.md5(raw_text.encode()).hexdigest()[0:8]