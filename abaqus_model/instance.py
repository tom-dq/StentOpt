
import typing

from abaqus_model import base
from abaqus_model import node
from abaqus_model import part


class Instance:
    base_part: part.Part
    name: str

    node_couples: typing.Set[node.NodeCouple]

    def __init__(self, base_part: part.Part, name: typing.Optional[str] = None):
        if name is None:
            name = base_part.name + "-1"

        self.base_part = base_part
        self.name = name
        self.node_couples = set()

    def make_inp_lines(self) -> typing.Iterable[str]:
        yield f"*Instance, name={self.name}, part={self.base_part.name}"
        yield "*End Instance"

        # Need an assembly-level node set to apply the constraints.
        # As per ANALYSIS_1.pdf, 2.1.1–9, we can reference the node sets in the parts.
        unique_name = base.deterministic_key(self, self.name)
        yield f"*Nset, nset={unique_name}, instance={self.name}"
        yield self.base_part.get_everything_set().get_name(base.SetContext.part)

        # For now, hard code a cylindrical axis system along the Y axis
        yield f"*Transform, nset={unique_name}, type=C"
        yield " 0.,         10.,           0.,           0.,         15.,           0."


    def _one_equation(self, one_couple: node.NodeCouple, dof: int) -> typing.Iterable[str]:

        node2_factor = "1." if one_couple.negated else "-1."

        yield f"** Constraint: Constraint-{self.name}-{one_couple.n1}-{one_couple.n2}-{dof}"
        yield "*Equation"
        yield "2"
        yield f"{self.name}.{one_couple.n1}, {dof}, 1."
        yield f"{self.name}.{one_couple.n2}, {dof}, {node2_factor}"


    def produce_equation_inp_line(self) -> typing.Iterable[str]:
        """Produce the constraint equations. These end up in the assembly part of the .inp
        so they need to be namespaced."""

        for one_couple in sorted(self.node_couples):
            for dof in base.DOFs:
                yield from self._one_equation(one_couple, dof)


    def add_node_couple(self, n1: int, n2: int, negated: bool):
        self.node_couples.add(
            node.NodeCouple(n1=n1, n2=n2, negated=negated)
        )


def make_instance_test() -> Instance:
    base_part = part.make_part_test()

    instance = Instance(base_part=base_part)

    instance.add_node_couple(1, 3, False)

    return instance


if __name__ == "__main__":
    instance = make_instance_test()
    for l in instance.make_inp_lines():
        print(l)

    for l in instance.produce_equation_inp_line():
        print(l)