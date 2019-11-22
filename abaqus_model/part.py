import hashlib
import typing

from abaqus_model import base
from abaqus_model import node
from abaqus_model import element
from abaqus_model import material

class Part:
    name: str
    common_material: material.MaterialBase
    nodes: node.Nodes
    elements: element.Elements
    node_couples: typing.Set[node.NodeCouple]

    def __init__(self, name: str, common_material: material.MaterialBase):
        self.name = name
        self.common_material = common_material
        self.nodes = node.Nodes()
        self.elements = element.Elements()
        self.node_couples = set()

    def add_node_validated(self, iNode: int, node: base.XYZ):
        """Adds a node and validates it."""
        self.nodes[iNode] = node

    def add_element_validate(self, iElem: int, element: element.Element):
        """Adds an element and makes sure all the nodes it references exist."""
        if not all(x in self.nodes for x in element.connection):
            raise ValueError(f"unconnected nodes on {element}")

        self.elements[iElem] = element

    def add_node_couple(self, n1: int, n2: int, negated: bool):
        self.node_couples.add(
            node.NodeCouple(n1=n1, n2=n2, negated=negated)
        )


    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield f"*Part, name={self.name}"
        yield from self.nodes.produce_inp_lines()
        yield from self.elements.produce_inp_lines()
        yield from self._produce_elset_nset_section()
        yield ", "
        yield "*End Part"

    def everything_set_name(self):
        return f"Set_{base.deterministic_key(self, self.name)}"

    def _produce_elset_nset_section(self) -> typing.Iterable[str]:
        unique_name = self.everything_set_name()
        if self.nodes:
            yield f"*Nset, nset={unique_name}, internal, generate"
            yield f"  {min(self.nodes.keys())},  {max(self.nodes.keys())},   1"

        if self.elements:
            yield f"*Elset, elset={unique_name}, internal, generate"
            yield f"  {min(self.elements.keys())},  {max(self.elements.keys())},   1"

        section_name = f"Section-{unique_name}"
        yield f"** Section: {section_name}"
        yield f"*Solid Section, elset={unique_name}, material={self.common_material.name}"

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




def test_make_part() -> Part:
    common_material = material.MaterialElastic(
        name="ElasticMaterial",
        density=1.0,
        elast_mod=200.0,
        elast_possion=0.3,
    )

    part = Part(name="Part-1", common_material=common_material)
    part.add_node_validated(1, base.XYZ(3.4, 3.5, 6.7))
    part.add_node_validated(2, base.XYZ(3.4, 3.6, 6.7))
    part.add_node_validated(3, base.XYZ(3.14, 3.5, 6.7))
    part.add_node_validated(4, base.XYZ(3.14, 3.5, 6.2))
    part.add_element_validate(1, element.Element("C3D4", (1, 2, 3, 4)))

    part.add_node_couple(1, 3, False)

    return part

if __name__ == "__main__":
    part = test_make_part()

    for l in part.produce_inp_lines():
        print(l)





















