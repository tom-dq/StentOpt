# Keeps track of how the optimisation has been going.
import itertools
import enum
import pathlib
import sqlite3
import typing
import statistics
import collections
import matplotlib.pyplot as plt

from stent_opt.struct_opt import design


class Snapshot(typing.NamedTuple):
    iteration_num: int
    filename: str
    active_elements: typing.FrozenSet[int]

    def for_db_form(self):
        return self._replace(
            active_elements=",".join(str(x) for x in self.active_elements)
        )

    def from_db(self):
        active_elements = frozenset(int(x) for x in self.active_elements.split(","))
        return self._replace(
            active_elements=active_elements,
        )

class StatusCheckStage(enum.Enum):
    primary = enum.auto()
    secondary = enum.auto()
    smoothed = enum.auto()


class StatusCheck(typing.NamedTuple):
    iteration_num: int
    elem_num: int
    stage: StatusCheckStage
    metric_name: str
    metric_val: float

    def for_db_form(self):
        return self._replace(stage=self.stage.name)

    def from_db(self):
        return self._replace(stage=StatusCheckStage[self.stage])


class ParamKeyValue(typing.NamedTuple):
    param_name: str
    param_value: typing.Optional[str]


class NodePosition(typing.NamedTuple):
    iteration_num: int
    node_num: int
    x: float
    y: float
    z: float


_make_snapshot_table = """CREATE TABLE IF NOT EXISTS Snapshot(
iteration_num INTEGER,
filename TEXT,
active_elements TEXT
)"""

_make_status_check_table = """CREATE TABLE IF NOT EXISTS StatusCheck(
iteration_num INTEGER,
elem_num INTEGER,
stage TEXT,
metric_name TEXT,
metric_val REAL
)"""

_make_parameters_table_ = """CREATE TABLE IF NOT EXISTS ParamKeyValue(
param_name TEXT UNIQUE,
param_value TEXT)"""


_make_node_pos_table_ = """CREATE TABLE IF NOT EXISTS NodePosition(
iteration_num INTEGER,
node_num INTEGER,
x REAL,
y REAL,
z REAL)"""

_make_tables = [
    _make_snapshot_table,
    _make_status_check_table,
    _make_parameters_table_,
    _make_node_pos_table_,
]

class History:
    fn = None
    connection = None

    def __init__(self, fn):
        self.fn = fn
        self.connection = sqlite3.connect(fn)

        with self.connection:
            for mt in _make_tables:
                self.connection.execute(mt)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.commit()
        self.connection.close()


    def _generate_insert_string_nt_class(self, named_tuple_class):
        question_marks = ["?" for _ in named_tuple_class._fields]
        table_name = named_tuple_class.__name__
        return "INSERT INTO {0} VALUES ({1})".format(table_name, ", ".join(question_marks))

    def add_snapshot(self, snapshot: Snapshot):
        ins_string = self._generate_insert_string_nt_class(Snapshot)

        with self.connection:
            self.connection.execute(ins_string, snapshot.for_db_form())

    def add_many_status_checks(self, status_checks: typing.Iterable[StatusCheck]):
        ins_string = self._generate_insert_string_nt_class(StatusCheck)
        with self.connection:
            status_checks_for_db = (st.for_db_form() for st in status_checks)
            self.connection.executemany(ins_string, status_checks_for_db)

    def add_node_positions(self, node_positions: typing.Iterable[NodePosition]):
        ins_string = self._generate_insert_string_nt_class(NodePosition)
        with self.connection:
            self.connection.executemany(ins_string, node_positions)

    def get_snapshots(self) -> typing.Iterable[Snapshot]:
        with self.connection:
            rows = self.connection.execute("SELECT * FROM Snapshot ORDER BY iteration_num")
            db_forms = (Snapshot(*row) for row in rows)
            yield from (one_db_form.from_db() for one_db_form in db_forms)

    def set_stent_params(self, stent_params: "design.StentParams"):
        """Save the model parameters"""
        data_rows = list(stent_params.to_db_strings())
        ins_string = self._generate_insert_string_nt_class(ParamKeyValue)
        with self.connection:
            self.connection.executemany(ins_string, data_rows)

    def get_stent_params(self) -> "design.StentParams":
        """Get the model parameters"""
        with self.connection:
            rows = self.connection.execute("SELECT * FROM ParamKeyValue")
            l_rows = list(rows)

        return design.StentParams.from_db_strings(l_rows)

    def max_saved_iteration_num(self) -> int:
        with self.connection:
            rows = self.connection.execute("SELECT iteration_num FROM Snapshot ORDER BY iteration_num DESC LIMIT 1")
            l_rows = list(rows)

        if len(l_rows) == 0:
            return None

        else:
            return l_rows[0][0]

    def _single_snapshot_row(self, rows) -> typing.Optional[Snapshot]:
        l_rows = list(rows)
        if len(l_rows) == 0:
            return None

        else:
            # There can be multiple snapshots because of restarts, etc. But they should be the same.
            final_form = {Snapshot(*row).from_db() for row in rows}

            if len(final_form) > 1:
                raise ValueError("More than one different Snapshot.")

            return final_form.pop()

    def get_snapshot(self, i: int) -> Snapshot:
        with self.connection:
            rows = list(self.connection.execute("SELECT * FROM Snapshot WHERE iteration_num = ?", (i,)))
            to_return = self._single_snapshot_row(rows)

        if not to_return:
            raise ValueError(f"Snapshot {i} not found.")

        return to_return

    def get_most_recent_snapshot(self) -> typing.Optional[Snapshot]:
        with self.connection:
            rows = list(self.connection.execute("SELECT * FROM Snapshot ORDER BY iteration_num DESC LIMIT 1"))
            return self._single_snapshot_row(rows)

    def get_most_recent_design(self) -> typing.Optional["design.StentDesign"]:
        maybe_snapshot = self.get_most_recent_snapshot()
        if maybe_snapshot:
            stent_params = self.get_stent_params()
            design_space = stent_params.divs
            elem_num_to_idx = {iElem: idx for iElem, idx in design.generate_elem_indices(design_space)}
            return design.StentDesign(
                stent_params=stent_params,
                active_elements=frozenset( (elem_num_to_idx[iElem] for iElem in maybe_snapshot.active_elements)),
            )

    def get_status_checks(self, iter_greater_than: int) -> typing.Iterable[StatusCheck]:
        with self.connection:
            rows = self.connection.execute("SELECT * FROM StatusCheck WHERE iteration_num >= ? ORDER BY iteration_num, metric_name ", (iter_greater_than,))
            yield from (StatusCheck(*row).from_db() for row in rows)

    def get_node_positions(self) -> typing.Iterable[NodePosition]:
        with self.connection:
            rows = self.connection.execute("SELECT * FROM NodePosition ORDER BY iteration_num")
            yield from (NodePosition(*row) for row in rows)



def plot_history(hist_fn):
    with History(hist_fn) as hist:
        all_status_lines = list(hist.get_status_checks())


    def metric_name_first(st):
        return st.metric_name

    def iteration_num_first(st):
        return st.iteration_num

    all_status_lines.sort(key=metric_name_first)

    # Hacky colour cycle
    all_colours = list(plt.get_cmap('tab10').colors)

    for metric_name, iter_of_metrics in itertools.groupby(all_status_lines, metric_name_first):
        one_metric_statuses_by_iter_num = sorted(iter_of_metrics, key=iteration_num_first)
        max_val = max(st.metric_val for st in one_metric_statuses_by_iter_num)

        x_axis = []
        y_max = []
        y_mean = []
        if max_val:
            normed_metrics = [st._replace(metric_val=st.metric_val/max_val) for st in one_metric_statuses_by_iter_num]

        else:
            normed_metrics = one_metric_statuses_by_iter_num

        for iter_num, it_of_metrics_by_elem in itertools.groupby(normed_metrics, iteration_num_first):
            all_vals = [st.metric_val for st in it_of_metrics_by_elem]

            x_axis.append(iter_num)
            y_max.append(max(all_vals))
            y_mean.append(statistics.mean(all_vals))

        this_col = all_colours.pop(0)


        plt.plot(x_axis, y_max, color=this_col, linestyle='--', label=f"{metric_name} Max")
        plt.plot(x_axis, y_mean, color=this_col, label=f"{metric_name} Mean")

    plt.gcf().set_size_inches(12, 9)
    plt.gca().set_yscale('log')

    plt.legend()
    plt.show()


def make_fn_in_dir(working_dir: pathlib.Path, ext: str, iter_num: int):
    """e.g., iter_num=123 and ext=".inp" """
    intermediary = working_dir / f'It-{str(iter_num).rjust(6, "0")}.XXX'
    return intermediary.with_suffix(ext)

def make_history_db(working_dir: pathlib.Path) -> pathlib.Path:
    return working_dir / "History.db"


def history_write_read_test():
    with History(r"c:\temp\aaa23.db") as history:
        orig_stent_params = design.basic_stent_params
        history.set_stent_params(orig_stent_params)
        recreated_stent_params = history.get_stent_params()

        print(orig_stent_params == recreated_stent_params)


if __name__ == "__main__":
    history_write_read_test()

if False:
    plot_history(r"E:\Simulations\StentOpt\aba-70\History.db")


