import typing

from stent_opt.abaqus_model import node, base, material, element, section


class Part:
    name: str
    common_section: typing.Optional[section.SectionBase]
    nodes: "node.Nodes"
    elements: element.Elements
    node_sets: typing.Dict[str, "node.NodeSet"]
    element_sets: typing.Dict[str, "element.ElementSet"]
    transform_to_cyl: bool
    _EVERYTHING_NAME = "Everything"

    def __init__(self, name: str, common_section: typing.Optional[section.SectionBase], transform_to_cyl: bool):
        self.name = name
        self.common_section = common_section
        self.transform_to_cyl = transform_to_cyl
        self.nodes = node.Nodes()
        self.elements = element.Elements()
        self.node_sets = dict()
        self.element_sets = dict()

    @property
    def name_instance(self) -> str:
        return f"{self.name}-1"

    def add_node_validated(self, iNode: int, node: base.XYZ, node_elem_offset: int):
        """Adds a node and validates it."""
        if not isinstance(node, base.XYZ):
            raise ValueError(node)

        if iNode + node_elem_offset in self.nodes:
            raise ValueError("Overwriting node?")

        self.nodes[iNode + node_elem_offset] = node

    def add_element_validate(self, iElem: int, in_element: element.Element, node_elem_offset: int = 0):
        """Adds an element and makes sure all the nodes it references exist."""

        # Transform the element if needs be
        iElem_offset = iElem + node_elem_offset
        element_offset = element.Element(name=in_element.name, connection=tuple(iNode+node_elem_offset for iNode in in_element.connection))

        if not all(x in self.nodes for x in element_offset.connection):
            raise ValueError(f"unconnected nodes on {element_offset}")

        self.elements[iElem_offset] = element_offset

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

        for elem_set_name, elem_set in self.element_sets.items():
            yield from elem_set.generate_inp_lines(set_context)

        # Assign the same section to all the elements.
        all_elem_set = self.get_everything_set(base.SetType.element)
        all_elem_set_name = all_elem_set.get_name(set_context)

        if self.common_section:
            yield from self.common_section.produce_inp_lines(all_elem_set)

    def get_used_node_nums(self) -> typing.FrozenSet[int]:
        used_node_nums = set()
        for one_elem in self.elements.values():
            used_node_nums.update(one_elem.connection)

        return frozenset(used_node_nums)

    def squeeze_unused_nodes(self):
        used_nodes = self.get_used_node_nums()

        to_go_node_nums = {iNode for iNode in self.nodes if iNode not in used_nodes}
        for iNode in to_go_node_nums:
            del self.nodes[iNode]


def make_part_test() -> Part:
    common_material = material.MaterialElasticPlastic(
        name="ElasticMaterial",
        density=1.0,
        elast_mod=200.0,
        elast_possion=0.3,
        plastic=None,
    )

    common_section = section.SolidSection(
        name="Solid",
        mat=common_material,
        enhanced_hourglass=None,
    )

    part = Part(name="Part-1", common_section=common_section)
    part.add_node_validated(1, base.XYZ(3.4, 3.5, 6.7))
    part.add_node_validated(2, base.XYZ(3.4, 3.6, 6.7))
    part.add_node_validated(3, base.XYZ(3.14, 3.5, 6.7))
    part.add_node_validated(4, base.XYZ(3.14, 3.5, 6.2))
    part.add_node_validated(5, base.XYZ(2.0, 3.5, 6.2))

    e1 = element.Element(element.ElemType.C3D4, (1, 2, 3, 4))
    e2 = element.Element(element.ElemType.C3D4, (5, 2, 3, 1))
    part.add_element_validate(1, e1)
    part.add_element_validate(2, e2)

    elem_set = element.ElementSet(part, "OneElem", element.Elements( {1: e1} ))
    part.add_element_set(elem_set)

    return part

if __name__ == "__main__":
    part = make_part_test()

    for l in part.produce_inp_lines():
        print(l)





















