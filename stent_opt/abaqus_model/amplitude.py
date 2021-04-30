import dataclasses
import typing

from stent_opt.abaqus_model import base

class XY(typing.NamedTuple):
    x: float
    y: float


def _eight_or_fewer_per_line(in_data: typing.Iterable[float]) -> typing.Iterable[str]:
    str_data = (base.abaqus_float(x) for x in in_data)

    for one_row in base.groups_of(str_data, 8):

        if len(one_row) < 8:
            for one_pair in base.groups_of(one_row, 2):
                yield ', '.join(one_pair)

        else:
            yield ', '.join(one_row)


@dataclasses.dataclass(frozen=True)
class AmplitudeBase:
    name: str


@dataclasses.dataclass(frozen=True)
class Amplitude(AmplitudeBase):
    name: str
    data: typing.Tuple[XY, ...]

    def make_inp_lines(self) -> typing.Iterable[str]:
        yield f"*Amplitude, name={base.quoted_if_necessary(self.name)}"

        def generate_single_points() -> typing.Iterable[float]:
            for datum in self.data:
                yield datum.x
                yield datum.y

        """ ***ERROR: ALL DATA LINES ON *AMPLITUDE, DEFINITION=TABULAR, EXCEPTING THE LAST 
           DATA LINE, MUST HAVE EXACTLY FOUR DATA PAIRS (EIGHT ENTRIES) OR ONE 
           SINGLE DATA PAIR (TWO ENTRIES). PLEASE CHECK THE DATA LINES FOR 
           *AMPLITUDE."""

        yield from _eight_or_fewer_per_line(generate_single_points())


@dataclasses.dataclass(frozen=True)
class AmplitudePeriodic(AmplitudeBase):
    name: str
    circ_freq: float
    start_time_step: float
    init_amp: float
    osc_amp: float

    def make_inp_lines(self) -> typing.Iterable[str]:
        yield f"*Amplitude, name={base.quoted_if_necessary(self.name)}, definition=PERIODIC"

        offset_0 = 0.0
        periodic_components = (
            (offset_0, self.osc_amp),
        )

        yield f"{len(periodic_components)}, {base.abaqus_float(self.circ_freq)}, {base.abaqus_float(self.start_time_step)}, {base.abaqus_float(self.init_amp)}"

        def generate_single_points() -> typing.Iterable[float]:
            for pc in periodic_components:
                yield pc[0]
                yield pc[1]

        yield from _eight_or_fewer_per_line(generate_single_points())


def make_test_amplitude() -> Amplitude:
    data = (
        XY(0.0, 0.0),
        XY(8.0, 0.4),
        XY(9.75, 0.8),
        XY(10.0, 0.9),
        XY(11.0, 0.0),
    )

    return Amplitude(
        name="Spike",
        data=data
    )


if __name__ == "__main__":
    amp = make_test_amplitude()
    for l in amp.make_inp_lines():
        print(l)



