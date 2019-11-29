import typing

from stent_opt.abaqus_model import node, base, material, element


class Part:
    name: str
    common_material: material.MaterialBase
    nodes: "node.Nodes"
    elements: element.Elements
    node_sets: typing.Dict[str, "node.NodeSet"]
    element_sets: typing.Dict[str, "element.ElementSet"]
    _EVERYTHING_NAME = "Everything"

    def __init__(self, name: str, common_material: material.MaterialBase):
        self.name = name
        self.common_material = common_material
        self.nodes = node.Nodes()
        self.elements = element.Elements()
        self.node_sets = dict()
        self.element_sets = dict()

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

    def _ensure_everything_sets_exists(self):
        everything_set_node = node.NodeSet(
            part=self,
            name_component=self._EVERYTHING_NAME,
            nodes=frozenset(self.nodes.keys()),
        )

        self.add_node_set(everything_set_node)

        everything_set_elements = element.ElementSet(
            part=self,
            name_component=self._EVERYTHING_NAME,
            elements=self.elements,
        )

        self.add_element_set(everything_set_elements)


    def add_node_set(self, node_set: "node.NodeSet"):
        self.node_sets[node_set.name_component] = node_set

    def add_element_set(self, element_set: "element.ElementSet"):
        self.element_sets[element_set.name_component] = element_set

    def get_everything_set(self, set_type: base.SetType):
        self._ensure_everything_sets_exists()

        if set_type == base.SetType.node:
            return self.node_sets[self._EVERYTHING_NAME]

        elif set_type == base.SetType.element:
            return self.element_sets[self._EVERYTHING_NAME]

        else:
            raise ValueError(set_type)


    def _produce_elset_nset_section(self) -> typing.Iterable[str]:
        set_context = base.SetContext.part
        self._ensure_everything_sets_exists()
        for node_set_name, node_set in self.node_sets.items():
            yield from node_set.generate_inp_lines(set_context)

        all_elem_set = self.get_everything_set(base.SetType.element)
        set_name = all_elem_set.get_name(set_context)
        yield from all_elem_set.generate_inp_lines(set_context)

        section_name = f"Section-{set_name}"
        yield f"** Section: {section_name}"
        yield f"*Solid Section, elset={set_name}, material={self.common_material.name}"


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





















