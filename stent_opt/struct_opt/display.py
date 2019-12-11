import collections
import typing

import matplotlib as mpl
import matplotlib.cm as cm
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import scipy as sp

from stent_opt.struct_opt import design, generation
from stent_opt.abaqus_model import base

def render_status(
        stent_design: design.StentDesign,
        node_position: typing.Dict[int, base.XYZ],
        components: typing.Iterable[generation.PrimaryRankingComponent]
):

    components = list(components)

    # Make sure there's only one (for now!)
    comp_types = {comp.comp_name for comp in components}
    if len(comp_types) != 1:
        raise ValueError(comp_types)

    comp_type = comp_types.pop()

    # Get all the faces of the stent elements
    def faces_and_values():
        elem_num_to_indices = {iElem: idx for iElem, idx in design.generate_elem_indices(stent_design.design_space)}
        elem_lookup = {elem_num_to_indices[comp.elem_id]: comp.value for comp in components}

        for one_elem in stent_design.active_elements:
            elem_connection = design.get_c3d8_connection(stent_design.design_space, one_elem)
            for one_face_idx in generation.FACES_OF_HEX:
                global_node_nums = [elem_connection[i] for i in one_face_idx]
                yield global_node_nums, elem_lookup[one_elem]

    # Find only the exposed faces.
    faces_paired = collections.defaultdict(list)
    for global_node_nums, value in faces_and_values():
        sorted_node_nums = tuple(sorted(global_node_nums))
        faces_paired[sorted_node_nums].append( (global_node_nums, value) )

    single_faces_and_vals = [one_list[0] for one_list in faces_paired.values() if len(one_list) == 1]


    # Render it
    poly_list = []
    for face_nodes, val in single_faces_and_vals:
        verts = [node_position[iNode] for iNode in face_nodes]

        poly_list.append( (verts, val))

    _show(poly_list)



def _show(poly_list):
    ax = a3.Axes3D(pl.figure())

    min_c = min(val for _, val in poly_list)
    max_c = max(val for _, val in poly_list)

    norm = mpl.colors.Normalize(vmin=min_c, vmax=max_c)
    cmap = cm.viridis

    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    for verts, val in poly_list:
        tri = a3.art3d.Poly3DCollection([verts])
        col = m.to_rgba(val)
        tri.set_color(col)
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    pl.show()

def render_thing():

    ax = a3.Axes3D(pl.figure())
    for i in range(1000):
        vtx = sp.rand(3,3)
        tri = a3.art3d.Poly3DCollection([vtx])
        tri.set_color(colors.rgb2hex(sp.rand(3)))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    pl.show()

if __name__ == "__main__":
    render_thing()