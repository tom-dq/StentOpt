import dataclasses
import typing

from stent_opt.abaqus_model import base, part


#from .base import XYZ, abaqus_float


class Nodes(dict):
    def __setitem__(self, key: int, value: "base.XYZ"):
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



@dataclasses.dataclass(frozen=True)
class NodeSet(base.SetBase):
    nodes: typing.FrozenSet[int]

    def _entity_numbers(self) -> typing.FrozenSet[int]:
        return self.nodes

    @property
    def set_type(self) -> base.SetType:
        return base.SetType.node


if __name__ == "__main__":
    pass

