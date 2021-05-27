
import typing
import os
import itertools

import psutil


class Computer(typing.NamedTuple):
    n_cpus_abaqus_explicit: int
    n_cpus_abaqus_implicit: int
    n_abaqus_parallel_solves: int
    base_working_dir: str
    working_dir: str
    fig_size: typing.Tuple[int, int]

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
    fig_size = (1200, 750)

elif psutil.cpu_count(logical=False) == 8:
    n_cpus = 1  # TEMP! Was 8
    base_working_dir = r"E:\Simulations\StentOpt"
    # fig_size = (1600, 800)  #  (2000, 1350)
    fig_size = (1800, 900)  #  (2000, 1350)

elif psutil.cpu_count(logical=False) == 6:
    # Macbook - just for viewing results at the moment.
    n_cpus = 1
    # base_working_dir = r"/Users/tomwilson/Dropbox/PhD/StentOptDBs"
    # base_working_dir = r"C:\Users\Tom Wilson\Documents\Stent-Opt-Data\StentOpt"
    base_working_dir = r"C:\Simulations\StentOpt"
    fig_size = (900, 500)  # (2000, 1350)

else:
    raise ValueError()

this_computer = Computer(
    n_cpus_abaqus_explicit=n_cpus,
    n_cpus_abaqus_implicit=1,
    n_abaqus_parallel_solves=2,
    base_working_dir=base_working_dir,
    working_dir=_get_next_free_dir(base_working_dir),
    # working_dir=r"E:\Simulations\StentOpt\AA-134",
    fig_size=fig_size,
)