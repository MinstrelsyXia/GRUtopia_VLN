import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

from grutopia.core.util.log import log

def euler_angles_to_quat(angles, degrees=False):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion.

    Args:
        angles (list or np.array): Euler angles [roll, pitch, yaw] in degrees.

    Returns:
        np.array: Quaternion [x, y, z, w].
    """
    r = R.from_euler('xyz', angles, degrees=degrees)
    quat = r.as_quat()
    return [quat[3], quat[0], quat[1], quat[2]]

def quat_to_euler_angles(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat (list or np.array): Quaternion [x, y, z, w].

    Returns:
        np.array: Euler angles [roll, pitch, yaw] in degrees.
    """
    r = R.from_quat(quat)
    angles = r.as_euler('xyz', degrees=True)
    return angles

def compute_rel_orientations(prev_position, current_position, return_quat=False):
    """
    Compute the relative orientation between two positions.

    Args:
        prev_position (np.array): Previous position [x, y, z].
        current_position (np.array): Current position [x, y, z].

    Returns:
        np.array: Relative orientation [roll, pitch, yaw] in degrees.
    """
    # Compute the relative orientation between the two positions
    current_position = np.array(current_position) if isinstance(current_position, list) else current_position
    prev_position = np.array(prev_position) if isinstance(prev_position, list) else prev_position
    diff = current_position - prev_position
    yaw = np.arctan2(diff[1], diff[0]) * 180 / np.pi
    if return_quat:
        return np.array(euler_angles_to_quat([0, 0, yaw]))
    else:
        return np.array([0, 0, yaw])

def dict_to_namespace(d):
    ns = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            value = dict_to_namespace(value)
        setattr(ns, key, value)
    return ns
