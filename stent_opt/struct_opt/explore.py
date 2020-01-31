import collections
import itertools
import pathlib
import typing

import holoviews
import panel
import numpy


from stent_opt.abaqus_model import base, element
from stent_opt.struct_opt import history, design

holoviews.extension('bokeh')

WORKING_DIR_TEMP = pathlib.Path(r"E:\Simulations\StentOpt\AA-23")


class ContourView(typing.NamedTuple):
    iteration_num: int
    metric_name: str
    deformed: bool


def get_status_checks() -> typing.List["history.StatusCheck"]:
    history_db = history.make_history_db(WORKING_DIR_TEMP)
    with history.History(history_db) as hist:
        return list(hist.get_status_checks())


def _build_contour_view_data(hist: history.History) -> typing.Iterable[typing.Tuple[ContourView, typing.Dict[int, float]]]:
    """Make the view info for a contour."""

    def make_contour_view(status_check: history.StatusCheck) -> ContourView:
        return ContourView(
            iteration_num=status_check.iteration_num,
            metric_name=status_check.metric_name,
            deformed=None,
        )

    for contour_view, sub_iter in itertools.groupby(hist.get_status_checks(), make_contour_view):
        elem_vals = {status_check.elem_num: status_check.metric_val for status_check in sub_iter}
        yield contour_view, elem_vals


def make_contour(
        node_pos: typing.Dict[int, base.XYZ],
        contour_view: ContourView,
        element_to_value: typing.Dict[element.Element, float],
) -> holoviews.Polygons:

    """Make a single Holoviews contour."""

    # TODO - try again with QuadMesh - maybe we can do some masking/nan/blah blah blah stuff...

    def gen_poly_data(elem, value):
        one_poly_data = {
            ('x', 'y'): [ (node_pos[iNode].x, node_pos[iNode].y) for iNode in elem.connection],
            'level': value}
        return one_poly_data

    poly_data = [gen_poly_data(elem, value) for elem, value in element_to_value.items()]

    polys = holoviews.Polygons(poly_data, vdims='level', group=contour_view.metric_name)
    polys.opts(color='level', aspect='equal', line_width=0.1, padding=0.1, width=2000, height=1200, invert_axes=True)

    return polys



def make_dashboard(working_dir: pathlib.Path):
    # Get the data from the

    history_db = history.make_history_db(working_dir)
    with history.History(history_db) as hist:
        stent_params = hist.get_stent_params()
        snapshots = {snap.iteration_num: snap for snap in hist.get_snapshots()}

        # Node undeformed position and connectivity
        node_pos_undeformed = {iNode: pos.to_xyz() for iNode, idx, pos in design.generate_nodes(stent_params)}
        node_pos_deformed_all_iters = collections.defaultdict(dict)
        for node_pos in hist.get_node_positions():
            node_pos_deformed_all_iters[node_pos.iteration_num][node_pos.node_num] = base.XYZ(x=node_pos.x, y=node_pos.y, z=node_pos.z)

        num_to_elem = {iElem: elem for idx, iElem, elem in design.generate_stent_part_elements(stent_params)}

        plot_dict = {}
        for contour_view_raw, elem_data in _build_contour_view_data(hist):

            for deformed, node_pos in (
                    (False, node_pos_undeformed),
                    (True, node_pos_deformed_all_iters[contour_view_raw.iteration_num])
            ):
                contour_view = contour_view_raw._replace(deformed=deformed)
                print(contour_view)

                # Some measures (like RegionGradient) can exist even if the element is inactive. Strip these out of the display.
                elem_to_value = {
                    num_to_elem[iElem]: val for
                    iElem, val in elem_data.items()
                    if iElem in snapshots[contour_view.iteration_num].active_elements}

                polys = make_contour(node_pos, contour_view, elem_to_value)
                plot_dict[contour_view] = polys


        hmap = holoviews.HoloMap(plot_dict, kdims=list(contour_view._fields))
        panel.panel(hmap).show()


def do_test():
    x = [4.2, 5, 6]
    y = [12,15]
    X, Y = numpy.meshgrid(x, y)
    Z = X + Y
    qmesh = holoviews.QuadMesh( (X, Y, Z))
    Z[0,1] = numpy.nan

    print(X)
    print(Y)
    print(Z)

    panel.panel(qmesh).show()


def do_test_poly():
    def make_poly(i):
        x= i
        y = 10+i
        return {
            ('x', 'y'): [(x, y), (x+0.4, y), (x+0.4, y+0.7), (x, y+0.8)],
            'level': i}

    poly_data = [make_poly(i) for i in range(5)]
    polys = holoviews.Polygons(poly_data, vdims='level')
    polys.opts(color='level', line_width=1, padding=0.1)
    panel.panel(polys).show()


def main():
    status_checks = get_status_checks()
    scatter_data = [
        (st.iteration_num, st.metric_val) for st in status_checks
    ]
    scatter = holoviews.Scatter(scatter_data)
    scatter.options(cmap='viridis')
    panel.panel(scatter).show()



if __name__ == '__main__':

    make_dashboard(WORKING_DIR_TEMP)