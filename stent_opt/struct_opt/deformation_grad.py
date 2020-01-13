# Utilities for separating the rigid body and deformation gradients of some point cloud.

import typing

import numpy
import scipy

from stent_opt.abaqus_model.base import XYZ

T_NodeKey = typing.TypeVar("T_NodeKey")
T_NodeMap = typing.Dict[T_NodeKey, XYZ]

class RigidMotion(typing.NamedTuple):
    R: numpy.array  # Rotation matrix, 3x3
    t: numpy.array  # Deformation vector.


def _get_points_array(key_order, nodes: T_NodeMap) -> numpy.array:
    points_in_order = (nodes[k] for k in key_order)
    points_raw = tuple(sorted(points_in_order))
    points_array = numpy.array(points_raw)
    return points_array


def _get_node_map(key_order, points: numpy.array) -> T_NodeMap:
    out_points = {}
    for idx, k in enumerate(key_order):
        x, y, z = [float(_x) for _x in points[idx, :]]
        out_points[k] = XYZ(x, y, z)

    return out_points

def get_rigid_motion(orig: T_NodeMap, deformed: T_NodeMap) -> RigidMotion:
    # Assume all points are weighted equally, and put them in the same order.
    node_keys = tuple(sorted(set.union(set(orig.keys()), set(deformed.keys()))))

    p_orig = _get_points_array(node_keys, orig)
    q_deformed = _get_points_array(node_keys, deformed)

    return _get_rigid_motion(p_orig, q_deformed)


def _get_rigid_motion(p_orig: numpy.array, q_deformed: numpy.array) -> RigidMotion:
    """Least-Squares Rigid Motion Using SVD
    Olga Sorkine-Hornung and Michael Rabinovich
    https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    """

    d = 3

    # Step 1 - get the centroids
    p_bar = numpy.mean(p_orig, axis=0)
    q_bar = numpy.mean(q_deformed, axis=0)

    # Step 2 - centred vectors
    x = p_orig - p_bar
    y = q_deformed - q_bar

    # Step 3 - compute the covariance matrix
    S = x.T @ y

    # Step 4 - SVD
    u, s, vh = numpy.linalg.svd(S, full_matrices=True)

    middle_bit = numpy.eye(d)
    det_vut = numpy.linalg.det(vh @ u.T)
    middle_bit[d-1, d-1] = det_vut
    R = vh @ middle_bit @ u.T

    t = q_bar - R @ p_bar

    return RigidMotion(R=R, t=t)


def apply_rigid_motion(rigid_motion: RigidMotion, orig: T_NodeMap) -> T_NodeMap:
    """Apply the rigid body motion to some undeformed nodes."""

    node_keys = tuple(sorted(orig.keys()))
    p_orig = _get_points_array(node_keys, orig)
    p_def = _apply_rigid_motion(rigid_motion, p_orig)
    return _get_node_map(node_keys, p_def)


def _apply_rigid_motion(rigid_motion: RigidMotion, p_orig: numpy.array) -> numpy.array:
    rot_only = rigid_motion.R @ p_orig.T
    p_def = rot_only.T + rigid_motion.t
    return p_def


def nodal_deformation(nominal_length: float, orig: T_NodeMap, deformed: T_NodeMap) -> float:
    """Gets some normalised deformation measure for the subset of nodes overall."""

    node_keys = tuple(sorted(set.union(set(orig.keys()), set(deformed.keys()))))

    p_orig = _get_points_array(node_keys, orig)
    q_deformed = _get_points_array(node_keys, deformed)

    rigid_motion = _get_rigid_motion(p_orig, q_deformed)
    p_with_rigid = _apply_rigid_motion(rigid_motion, p_orig)

    deformation_only = q_deformed - p_with_rigid
    norms = numpy.linalg.norm(deformation_only, axis=1)
    if len(norms) != len(orig):
        raise ValueError("Looks like Tom got an index/axis wrong someplace...")

    return float(numpy.mean(norms) / nominal_length)


if __name__ == "__main__":
    orig_points = {
        "a": XYZ(0.0, 0.0, 0.0),
        "b": XYZ(1.0, 0.0, 0.0),
        "c": XYZ(1.0, 1.0, 0.0),
        "d": XYZ(0.0, 1.0, 0.0),
        "e": XYZ(0.5, 0.5, 0.0),
    }

    move_in_z = {
        "a": XYZ(0.0, 0.0, 1.0),
        "b": XYZ(1.0, 0.0, 1.0),
        "c": XYZ(1.0, 1.0, 1.0),
        "d": XYZ(0.0, 1.0, 1.0),
        "e": XYZ(0.5, 0.5, 1.0),
    }

    strain_in_x = {
        "a": XYZ(0.0, 0.0, 0.0),
        "b": XYZ(2.0, 0.0, 0.0),
        "c": XYZ(2.0, 1.0, 0.0),
        "d": XYZ(0.0, 1.0, 0.0),
        "e": XYZ(1.0, 0.5, 0.0),
    }

    rotate_90_deg = {
        "a": XYZ(0.0, 0.0, 0.0),
        "b": XYZ(0.0, 1.0, 0.0),
        "c": XYZ(-1.0, 1.0, 0.0),
        "d": XYZ(-1.0, 0.0, 0.0),
        "e": XYZ(-0.5, 0.5, 0.0),
    }

    print(nodal_deformation(1.0, orig_points, move_in_z))
    print(nodal_deformation(1.0, orig_points, strain_in_x))
    print(nodal_deformation(1.0, orig_points, rotate_90_deg))




















