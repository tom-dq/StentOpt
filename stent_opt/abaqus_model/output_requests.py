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
    ESEDEN = "Total elastic strain energy density in the element."
    EPDDEN = "Total energy dissipated per unit volume in the element by rate-independent and rate-dependent plastic deformation."


T_OutputRequest = typing.Union[NodeOutputs, ElementOutputs]

general_components = list(NodeOutputs) + list(ElementOutputs)

def produce_inp_lines(abaqus_output_time_interval: float, output_components: typing.Iterable[T_OutputRequest]) -> typing.Iterable[str]:
    # Output requests
    yield from base.inp_heading("OUTPUT REQUESTS")
    yield "*Restart, write, overlay, number interval=1, time marks=YES"
    yield from base.inp_heading("FIELD OUTPUT: F-Output-1")

    yield f"*Output, field, time interval={base.abaqus_float(abaqus_output_time_interval)}"

    out_comps = list(output_components)

    node_components = [c for c in out_comps if isinstance(c, NodeOutputs)]
    if node_components:
        yield "*Node Output"
        yield ", ".join(c.name for c in node_components)

    elem_components = [c for c in out_comps if isinstance(c, ElementOutputs)]
    if elem_components:
        yield "*Element Output, directions=YES"
        yield ", ".join(c.name for c in elem_components)

