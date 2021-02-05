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


class EnergyOutputs(enum.Enum):
    ALLSE = "Recoverable strain energy."
    ALLPD = "Energy dissipated by rate-independent and ratedependent plastic deformation."
    ALLKE = "Kinetic energy."
    ALLWK = "External work."


T_OutputRequest = typing.Union[NodeOutputs, ElementOutputs]
T_HistoryRequest = typing.Union[EnergyOutputs]

general_components = list(NodeOutputs) + list(ElementOutputs)
history_components = list(EnergyOutputs)

def produce_inp_lines(
        abaqus_output_time_interval: float,
        output_components: typing.Iterable[T_OutputRequest],
        abaqus_history_time_interval: float,
        output_history: typing.Iterable[T_HistoryRequest]) -> typing.Iterable[str]:
    # Output requests
    yield from base.inp_heading("OUTPUT REQUESTS")
    yield "*Restart, write, overlay, number interval=1, time marks=YES"
    yield from base.inp_heading("FIELD OUTPUT: F-Output-1")

    # Field output
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

    # History output
    out_hist = list(output_history)
    if out_hist:
        yield f"*Output, history, time interval={base.abaqus_float(abaqus_history_time_interval)}"

        energy_output = [h for h in out_hist if isinstance(h, EnergyOutputs)]
        output_not_dealt_with = [h for h in out_hist if not isinstance(h, EnergyOutputs)]

        if output_not_dealt_with:
            raise ValueError("Write this code!")

        if energy_output:
            yield "*Energy Output"
            yield ", ".join(c.name for c in energy_output)
