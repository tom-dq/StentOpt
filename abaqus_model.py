import typing

class XYZ(typing.NamedTuple):
    x: float
    y: float
    z: float


T_NodeDict = typing.Dict[int, XYZ]

class Element(typing.NamedTuple):
    pass

class AbaqusModel:
    nodes: T_NodeDict

