import typing

from stent_opt.abaqus_model import base

class XY(typing.NamedTuple):
    x: float
    y: float


class Amplitude(typing.NamedTuple):
    name: str
    data: typing.Tuple[XY, ...]

    def make_inp_lines(self) -> typing.Iterable[str]:
        yield f"*Amplitude, name={base.quoted_if_necessary(self.name)}"

        def generate_single_points() -> typing.Iterable[float]:
            for datum in self.data:
                yield datum.x
                yield datum.y

        str_data = (base.abaqus_float(x) for x in generate_single_points())

        for one_row in base.groups_of(str_data, 4):
            yield ', '.join(one_row)


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



