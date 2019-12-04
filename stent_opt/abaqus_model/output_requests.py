import enum
import typing

from stent_opt.abaqus_model import base

class NodeOutputs(enum.Enum):
    RF = "Reaction Force & Moment Components"
    U = "Displacement Components"


class ElementOutputs(enum.Enum):
    ELSE = "Total elastic strain energy in the element"
    ELPD = "Total energy dissipated in the element by plastic deformation"
    S = "All stress components"
    SP = "All principal stress components"
    LE = "All logarithmic strain components"
    PEEQ = "Equivalent plastic strain"


T_OutputRequest = typing.Union[NodeOutputs, ElementOutputs]

general_components = [
    NodeOutputs.RF,
    NodeOutputs.U,
    ElementOutputs.S,
    ElementOutputs.SP,
    ElementOutputs.LE,
    ElementOutputs.PEEQ,
    ElementOutputs.ELSE,
    ElementOutputs.ELPD,
]


def produce_inp_lines(output_components: typing.Iterable[T_OutputRequest]) -> typing.Iterable[str]:
    # Output requests
    yield from base.inp_heading("OUTPUT REQUESTS")
    yield "*Restart, write, overlay, number interval=1, time marks=YES"
    yield from base.inp_heading("FIELD OUTPUT: F-Output-1")
    yield "*Output, field, time interval=0.1"

    out_comps = list(output_components)

    node_components = [c for c in out_comps if isinstance(c, NodeOutputs)]
    if node_components:
        yield "*Node Output"
        yield ", ".join(c.name for c in node_components)

    elem_components = [c for c in out_comps if isinstance(c, ElementOutputs)]
    if elem_components:
        yield "*Element Output, directions=YES"
        yield ", ".join(c.name for c in elem_components)

