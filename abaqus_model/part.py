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
    node_sets: typing.Dict[str, node.NodeSet]
    _EVERYTHING_NAME = "Everything"

    def __init__(self, name: str, common_material: material.MaterialBase):
        self.name = name
        self.common_material = common_material
        self.nodes = node.Nodes()
        self.elements = element.Elements()
        self.node_sets = dict()

    @property
    def name_instance(self) -> str:
        return f"{self.name}-1"

    def add_node_validated(self, iNode: int, node: base.XYZ):
        """Adds a node and validates it."""
        self.nodes[iNode] = node

    def add_element_validate(self, iElem: int, element: element.Element):
        """Adds an element and makes sure all the nodes it references exist."""
        if not all(x in self.nodes for x in element.connection):
            raise ValueError(f"unconnected nodes on {element}")

        self.elements[iElem] = element


    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield f"*Part, name={self.name}"
        yield from self.nodes.produce_inp_lines()
        yield from self.elements.produce_inp_lines()
        yield from self._produce_elset_nset_section()
        yield ", "
        yield "*End Part"

    def _ensure_everything_set_exists(self):
        everything_set = node.NodeSet(
            part=self,
            name_component=self._EVERYTHING_NAME,
            nodes=frozenset(self.nodes.keys()),
        )

        self.add_node_set(everything_set)

    def add_node_set(self, node_set: node.NodeSet):
        self.node_sets[node_set.name_component] = node_set

    def get_everything_set(self):
        self._ensure_everything_set_exists()
        return self.node_sets[self._EVERYTHING_NAME]

    def _produce_elset_nset_section(self) -> typing.Iterable[str]:
        set_context = base.SetContext.part
        self._ensure_everything_set_exists()
        for node_set_name, node_set in self.node_sets.items():
            yield from node_set.generate_inp_lines(set_context)

        unique_name = self.get_everything_set().get_name(set_context)
        if self.elements:
            yield f"*Elset, elset={unique_name}, generate"
            yield f"  {min(self.elements.keys())},  {max(self.elements.keys())},   1"

        section_name = f"Section-{unique_name}"
        yield f"** Section: {section_name}"
        yield f"*Solid Section, elset={unique_name}, material={self.common_material.name}"







def make_part_test() -> Part:
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



    return part

if __name__ == "__main__":
    part = make_part_test()

    for l in part.produce_inp_lines():
        print(l)





















