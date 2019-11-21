import typing

class XYZ(typing.NamedTuple):
    x: float
    y: float
    z: float


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

