import itertools
import typing

from stent_opt.abaqus_model import base


class Element(typing.NamedTuple):
    name: str  # e.g., C3D8R
    connection: typing.Tuple[int, ...]


class Elements(dict):
    def __setitem__(self, key: int, value: Element):
        if not isinstance(value, Element):
            raise TypeError("Can only add Elements to this")

        super().__setitem__(key, value)

    def produce_inp_lines(self) -> typing.Iterable[str]:
        """Produce this kind of thing:
            *Element, type=C3D8R
             1, 19, 20, 23, 22,  1,  2,  5,  4
             2, 20, 21, 24, 23,  2,  3,  6,  5"""
        elem_nums_and_data = sorted((elem.name, iElem, elem.connection) for iElem, elem in self.items())

        def group_key(name_num_conn):
            return name_num_conn[0]

        for name, iterable in itertools.groupby(elem_nums_and_data, key=group_key):
            yield f"*Element, type={name}"
            for _, num, conn in iterable:
                conn_str = ", ".join(str(x) for x in conn)
                yield f"{num}, {conn_str}"

            yield ""


class ElementSet(typing.NamedTuple):
    part: "part.Part"
    name_component: str
    elements: Elements

    def get_name(self, set_context: base.SetContext) -> str:
        if set_context == base.SetContext.part:
            node_number_hash = str(hash(self.elements))
            return f"Set_{self.name_component}_{base.deterministic_key(self.part, node_number_hash)}"

        elif set_context == base.SetContext.assembly:
            name_in_part = self.get_name(base.SetContext.part)
            return f"{self.part.name}.{name_in_part}"

        else:
            raise ValueError(set_context)

    def generate_inp_lines(self, set_context: base.SetContext) -> typing.Iterable[str]:
        """Generateds the *Nset lines"""
        seq_nodes = frozenset(range(min(self.elements), max(self.elements)+1))
        nodes_are_sequential = self.elements == seq_nodes
        if nodes_are_sequential:
            yield f"*Elset, elset={self.get_name(set_context)}, generate"
            yield f"  {min(self.elements)},  {max(self.elements)},   1"

        else:
            raise ValueError("Time to write this code I guess!")
