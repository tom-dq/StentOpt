import collections
import itertools
import math
import functools
import sys

from abaqus import *
from abaqusConstants import *

import sketch
import part

StentParams = collections.namedtuple('StentParams', 'length inner_radius thickness angle_deg n_div_z n_div_theta n_div_thickness')
PositionIndex = collections.namedtuple('PositionIndex', 'iR iTheta iZ')
PositionCyl = collections.namedtuple('PositionCyl', 'R Theta Z')
PositionXYZ = collections.namedtuple('PositionXYZ', 'X Y Z')

MODELNAME = 'Model A'
PARTNAME = 'Stent'
INSTANCENAME = 'Stent-1'
_full_circle_ = 2.0 * math.pi

myModel = mdb.Model(name=MODELNAME)
stent = myModel.Part(name=PARTNAME, dimensionality=THREE_D, type=DEFORMABLE_BODY)
stentInstance = mdb.models[MODELNAME].rootAssembly.Instance(name=INSTANCENAME, part=stent, dependent=False)



#STENT_CYL = STENT_CYL_FEATURE
#stent.generateMesh(regions=(stent.faces[0],))

def log(*args):
    out = [str(x) for x in args]
    print >> sys.__stdout__, '\t'.join(out)

def include_all(pos_cyl):
    return True
    
def include_some_test(pos_cyl):
    if abs(math.cos(pos_cyl.Theta - pos_cyl.Z/2.0)) < 0.5:
        return False
    else:
        return True

def include_decider_hollow_middle(stent_params, pos_cyl):

    dist_from_end = min(abs(pos_cyl.Z), abs(pos_cyl.Z - stent_params.length))
    segment_angle = math.radians(stent_params.angle_deg)
    ang_from_lower = abs(pos_cyl.Theta)
    ang_from_higher = abs(abs(pos_cyl.Theta) - segment_angle)
    dist_from_edges_rad = min(ang_from_lower, ang_from_higher)
    dist_from_edge_linear = pos_cyl.R * dist_from_edges_rad

    dist_from_edge = min(dist_from_end, dist_from_edge_linear)

    if dist_from_edge < 1.2:
        return True

    length_gap = 1.0
    return round(2 * pos_cyl.Z / length_gap) % 2 == 0


def make_stent(part, stent_params, include_decider):
    points_z = range(stent_params.n_div_z)
    points_theta = range(stent_params.n_div_theta)
    points_r = range(stent_params.n_div_thickness)

    max_angle_radians = math.radians(stent_params.angle_deg)
    is_full_circle = abs(stent_params.angle_deg - 360) < 1e-10

    def point_to_coords_cyl(pos_idx):
        z = pos_idx.iZ * stent_params.length / (1.0*(stent_params.n_div_z-1))
        theta = pos_idx.iTheta * (max_angle_radians / (stent_params.n_div_theta-1))
        r = stent_params.inner_radius + pos_idx.iR * (stent_params.thickness / (1.0*(stent_params.n_div_thickness-1)))
        return PositionCyl(R=r, Theta=theta, Z=z)
        
    def point_to_coords_xyz(pos_idx):
        pos_cyl = point_to_coords_cyl(pos_idx)
        global_x = pos_cyl.R * math.cos(pos_cyl.Theta)
        global_y = pos_cyl.R * math.sin(pos_cyl.Theta)
        global_z = pos_cyl.Z
        return PositionXYZ(X=global_x, Y=global_y, Z=global_z)
        
    # Utility functions to move indices...    
    def plus_theta(i_coords):
        iTheta = i_coords[1]
        if iTheta==stent_params.n_div_theta-1:
            return i_coords._replace(iTheta=0)

        else:
            return i_coords._replace(iTheta=i_coords.iTheta+1)
            
    def plus_z(i_coords):
        return i_coords._replace(iZ=i_coords.iZ+1)
        
    def plus_r(i_coords):
        return i_coords._replace(iR=i_coords.iR+1)
        
    def elem_connections(i_coords):
        node_pos_index = [
            i_coords,
            plus_theta(i_coords),
            plus_z(plus_theta(i_coords)),
            plus_z(i_coords),
            plus_r(i_coords),
            plus_theta(plus_r(i_coords)),
            plus_z(plus_theta(plus_r(i_coords))),
            plus_z(plus_r(i_coords))
            ]
        return node_pos_index
        
    def elem_centroid(pos_index):
        def average(l):
            return sum(l) / len(l)
            
        def average_angle_boundary(l):
            # Average across a 2*pi jump
            def min_diff(a,b):
                poss = [a-b, a-b+max_angle_radians, a-b-max_angle_radians]
                return min(poss)
                
            first, rest = l[0], l[1:]
            
            diffs = [min_diff(first, x) for x in rest]
            totals = [first] + [first+x for x in diffs]
            return average(totals)
        
        all_coords =[point_to_coords_cyl(iPos) for iPos in elem_connections(pos_index)]
        all_r, all_theta, all_z = zip(*all_coords)
        return PositionCyl(R=average(all_r), Theta=average_angle_boundary(all_theta), Z=average(all_z))
        
        
    # First element pass - see which nodes are used.
    used_node_index = set()
    used_element_index = list()
    if is_full_circle:
        angle_points = points_theta

    else:
        angle_points = points_theta[0:-1]

    for iR, iTheta, iZ in itertools.product(points_r[0:-1], angle_points, points_z[0:-1]):
        elem_index_coords = PositionIndex(iR=iR, iTheta=iTheta, iZ=iZ)
        elem_cent = elem_centroid(elem_index_coords)
        if include_decider(elem_cent):
            used_element_index.append(elem_index_coords)
            for each_node_idx in elem_connections(elem_index_coords):
                used_node_index.add(each_node_idx)
        
    # Second pass - create the nodes.
    nodes = {}
    for node_pos_index in sorted(used_node_index):
        coords_xyz = point_to_coords_xyz(node_pos_index)
        nodes[node_pos_index] = part.Node(coordinates=coords_xyz)
        
    # Third pass - create the elements.
    for elem_index_coords in used_element_index:
        these_nodes = [nodes[local_node] for local_node in elem_connections(elem_index_coords)]
        part.Element(nodes=these_nodes, elemShape=HEX8)
    


        

#stent1=mdb.models[MODELNAME].rootAssembly.instances[INSTANCENAME]


if __name__ == "__main__":
    stent_params = StentParams(length=20.0, inner_radius=2.0, angle_deg=120, thickness=0.5, n_div_z=20, n_div_theta=18,
                               n_div_thickness=2)

    include_decider = functools.partial(include_decider_hollow_middle, stent_params)

    make_stent(stent, stent_params, include_decider)

    real_part = mdb.models[MODELNAME].parts[PARTNAME].PartFromMesh(name='Stent-Real', copySets=True)
    STENT_CYL_FEATURE = real_part.DatumCsysByThreePoints(name='cylC',coordSysType=CYLINDRICAL, origin=(0,0,0),point1=(0.0, 0.0, 1.0), point2=(1.0, 0.0, 0.0) )

    mdb.saveAs(pathName=r'E:\Simulations\StentOpt\From-Python-J')

    # Run with this on the command line:
    # abaqus cae noGUI=basics.py

