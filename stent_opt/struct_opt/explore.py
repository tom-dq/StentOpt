
import pathlib
import typing

import holoviews
import panel
import numpy


from stent_opt.struct_opt import history

holoviews.extension('bokeh')

WORKING_DIR_TEMP = pathlib.Path(r"C:\TEMP\aba\AA-14")


def get_status_checks() -> typing.List["history.StatusCheck"]:
    history_db = history.make_history_db(WORKING_DIR_TEMP)
    with history.History(history_db) as hist:
        return list(hist.get_status_checks())


def main():
    status_checks = get_status_checks()
    scatter_data = [
        (st.iteration_num, st.metric_val) for st in status_checks
    ]
    scatter = holoviews.Scatter(scatter_data)
    panel.panel(scatter).show()



if __name__ == '__main__':
    main()