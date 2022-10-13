import dataclasses
import itertools
import typing
import enum

from stent_opt.abaqus_model import base


class ElemType(enum.Enum):
    C3D4 = enum.auto()
    C3D8R = enum.auto()
    M3D4R = enum.auto()
    SFM3D4R = (
        enum.auto()
    )  # Computationally efficient for elements where everything is constrained.
    CPS4R = enum.auto()  # Plane stress, reduced integration, hourglass control.
    CPE4R = enum.auto()  # Plane strain, reduced integration, hourglass control.


class Element(typing.NamedTuple):
    name: ElemType  # e.g., C3D8R
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
        elem_nums_and_data = sorted(
            (elem.name, iElem, elem.connection) for iElem, elem in self.items()
        )

        def group_key(name_num_conn):
            return name_num_conn[0]

        for name, iterable in itertools.groupby(elem_nums_and_data, key=group_key):
            yield f"*Element, type={name.name}"
            for _, num, conn in iterable:
                conn_str = ", ".join(str(x) for x in conn)
                yield f"{num}, {conn_str}"

            yield ""


@dataclasses.dataclass(frozen=True)
class ElementSet(base.SetBase):
    elements: Elements

    def _entity_numbers(self) -> typing.FrozenSet[int]:
        return frozenset(self.elements.keys())

    @property
    def set_type(self) -> base.SetType:
        return base.SetType.element
