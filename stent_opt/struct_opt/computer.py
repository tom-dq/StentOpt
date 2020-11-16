
import typing
import os
import itertools

import psutil


class Computer(typing.NamedTuple):
    n_cpus_abaqus: int
    base_working_dir: str
    working_dir: str


def _get_next_free_dir(base_dir):
    def idea(i):
        return f"AA-{i}"

    existing_stuff = set(os.listdir(base_dir))

    def is_already_there(f):
        return f in existing_stuff

    all_proposals = (idea(x) for x in itertools.count())
    good_proposals = itertools.dropwhile(is_already_there, all_proposals)
    one_good_propsal = next(good_proposals)
    return os.path.join(base_dir, one_good_propsal)


# Different stuff for which computer we're on
if psutil.cpu_count(logical=False) == 2:
    n_cpus = 1
    base_working_dir = r"C:\TEMP\aba"

elif psutil.cpu_count(logical=False) == 8:
    n_cpus = 8
    base_working_dir = r"E:\Simulations\StentOpt"

else:
    raise ValueError()

this_computer = Computer(
    n_cpus_abaqus=n_cpus,
    base_working_dir=base_working_dir,
    working_dir=_get_next_free_dir(base_working_dir),
)