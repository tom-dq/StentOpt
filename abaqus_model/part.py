import typing

from .base import XYZ
from .node import Nodes
from .element import Element, Elements


class Part:
    name: str
    nodes: Nodes
    elements: Elements

    def __init__(self, name: str):
        self.name = name
        self.nodes = Nodes()
        self.elements = Elements()

    def add_node_validated(self, iNode: int, node: XYZ):
        """Adds a node and validates it."""
        self.nodes[iNode] = node

    def add_element_validate(self, iElem: int, element: Element):
        """Adds an element and makes sure all the nodes it references exist."""
        if not all(x in self.nodes for x in element.connection):
            raise ValueError(f"unconnected nodes on {element}")

        self.elements[iElem] = element

    def produce_inp_lines(self) -> typing.Iterable[str]:
        yield f"*Part, name={self.name}"
        yield from self.nodes.produce_inp_lines()
        yield from self.elements.produce_inp_lines()



def test_make_part():
    part = Part(name="Part-1")
    part.add_node_validated(1, XYZ(3.4, 3.5, 6.7))
    part.add_node_validated(2, XYZ(3.4, 3.6, 6.7))
    part.add_node_validated(3, XYZ(3.14, 3.5, 6.7))
    part.add_node_validated(4, XYZ(3.14, 3.5, 6.2))
    part.add_element_validate(1, Element("C3D4", (1, 2, 3, 4)))


if __name__ == "__main__":
    test_make_part()








