# Keeps track of how the optimisation has been going.
import itertools
import sqlite3
import typing
import statistics
import collections
import matplotlib.pyplot as plt

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

class StatusCheck(typing.NamedTuple):
    iteration_num: int
    elem_num: int
    metric_name: str
    metric_val: float


_make_snapshot_table = """CREATE TABLE IF NOT EXISTS Snapshot(
iteration_num INTEGER,
filename TEXT,
active_elements TEXT
)"""

_make_status_check_table = """CREATE TABLE IF NOT EXISTS StatusCheck(
iteration_num INTEGER,
elem_num INTEGER,
metric_name TEXT,
metric_val REAL
)"""

_make_tables = [
    _make_snapshot_table,
    _make_status_check_table,
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
            self.connection.executemany(ins_string, status_checks)

    def get_snapshots(self) -> typing.Iterable[Snapshot]:
        with self.connection:
            rows = self.connection.execute("SELECT * FROM Snapshot ORDER BY iteration_num")
            db_forms = (Snapshot(*row) for row in rows)
            yield from (one_db_form.from_db() for one_db_form in db_forms)

    def get_status_checks(self) -> typing.Iterable[StatusCheck]:
        with self.connection:
            rows = self.connection.execute("SELECT * FROM StatusCheck ORDER BY iteration_num")
            yield from (StatusCheck(*row) for row in rows)



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

if __name__ == "__main__":
    plot_history(r"C:\Temp\aba\opt-42\History.db")















