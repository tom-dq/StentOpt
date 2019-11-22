import typing
import hashlib

from abaqus_model import base
#from .base import XYZ, abaqus_float
from .part import Part


class Nodes(dict):
    def __setitem__(self, key: int, value: base.XYZ):
        if not all(isinstance(x, float) for x in value):
            raise TypeError("Should only be floats in here.")

        super().__setitem__(key, value)

    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield "*Node"
        for iNode, xyz in sorted(self.items()):
            yield f"{iNode:10}, {base.abaqus_float(xyz.x)}, {base.abaqus_float(xyz.y)}, {base.abaqus_float(xyz.z)}"


class NodeCouple(typing.NamedTuple):
    n1: int
    n2: int
    negated: bool


class NodeSet(typing.NamedTuple):
    part: Part
    name_component: str
    nodes: typing.FrozenSet[int]

    @property
    def name_part(self) -> str:
        """The name from the point of view of the part, e.g., "Set_dksfjbh" """
        node_number_hash = str(hash(self.nodes))
        return f"Set_{self.name_component}_{base.deterministic_key(self.part, node_number_hash)}"

    @property
    def name_assembly(self) -> str:
        """The name from the point of view of the assembly, e.g., "Part-1.Set_dksfjbh" """
        return f"{self.part.name}.{self.name_part}"

