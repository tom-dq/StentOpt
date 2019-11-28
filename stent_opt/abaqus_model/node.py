import typing

from stent_opt.abaqus_model import base, part


#from .base import XYZ, abaqus_float


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
    part: "part.Part"
    name_component: str
    nodes: typing.FrozenSet[int]

    def get_name(self, set_context: base.SetContext) -> str:
        if set_context == base.SetContext.part:
            node_number_hash = str(hash(self.nodes))
            return f"Set_{self.name_component}_{base.deterministic_key(self.part, node_number_hash)}"

        elif set_context == base.SetContext.assembly:
            name_in_part = self.get_name(base.SetContext.part)
            return f"{self.part.name}.{name_in_part}"

        else:
            raise ValueError(set_context)

    def generate_inp_lines(self, set_context: base.SetContext) -> typing.Iterable[str]:
        """Generateds the *Nset lines"""
        seq_nodes = frozenset(range(min(self.nodes), max(self.nodes)+1))
        nodes_are_sequential = self.nodes == seq_nodes
        if nodes_are_sequential:
            yield f"*Nset, nset={self.get_name(set_context)}, generate"
            yield f"  {min(self.nodes)},  {max(self.nodes)},   1"

        else:
            raise ValueError("Time to write this code I guess!")


if __name__ == "__main__":
    pass

