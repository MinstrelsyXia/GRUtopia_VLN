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
        np.array: Quaternion [w, x, y, z].
    """
    r = R.from_euler('xyz', angles, degrees=degrees)
    quat = r.as_quat()
    return [quat[3], quat[0], quat[1], quat[2]]

def quat_to_euler_angles(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat (list or np.array): Quaternion [w, x, y, z].

    Returns:
        np.array: Euler angles [roll, pitch, yaw] in degrees.
    """
    reordered_quat = [quat[1], quat[2], quat[3], quat[0]]
    r = R.from_quat(reordered_quat)
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

def get_diff_beween_two_quat(w1,w2):
    a1 = quat_to_euler_angles(w1)
    a2 = quat_to_euler_angles(w2)
    diff = (diff + np.pi) % (2 * np.pi) - np.pi  # 将差值归一化到 -π 到 π
    return np.linalg.norm(diff)



def dict_to_namespace(d):
    ns = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            value = dict_to_namespace(value)
        setattr(ns, key, value)
    return ns



def get_dummy_2d_grid(width,height):
    # Generate a meshgrid of pixel coordinates
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)

    # Flatten the meshgrid arrays to correspond to the flattened depth map
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    # Combine the flattened x and y coordinates into a 2D array of points
    points_2d = np.vstack((xx_flat, yy_flat)).T  # Shape will be (N, 2), where N = height * width
    return points_2d

def downsample_pc(pc, depth_sample_rate):
    '''
    INput: points:(N,3); rate:downsample rate:int
    Output: downsampled_points:(N/rate,3)
    '''
    # np.random.seed(42)
    shuffle_mask = np.arange(pc.shape[0])
    np.random.shuffle(shuffle_mask)
    shuffle_mask = shuffle_mask[::depth_sample_rate]
    pc = pc[shuffle_mask,:]
    return pc

import open3d as o3d
def save_point_cloud_image(pcd, save_path="point_cloud.jpg"):
    # 设置无头渲染
    vis = o3d.visualization.Visualizer()
    vis.create_window()  # 创建一个不可见的窗口
    ctr = vis.get_view_control()

    # 设定特定的视角
    ctr.set_front([0, 0, -1])  # 设置相机朝向正面
    ctr.set_lookat([0, 0, 0])  # 设置相机目标点为原点
    ctr.set_up([0, 0, 1])   
    # 创建点云对象
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # 捕获当前视图并保存为图像
    vis.capture_screen_image(save_path)
    vis.destroy_window()
              
def visualize_pc(pcd,headless,save_path = 'pc.jpg'):
    '''
    pcd:     after:    pcd_global = o3d.geometry.PointCloud()
    pcd_global.points = o3d.utility.Vector3dVector(points_3d)
    '''
    if headless==True:
        save_point_cloud_image(pcd,save_path=save_path)
        return
    else:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1.0, origin=[0, 0, 0]) 
        o3d.io.write_point_cloud("point_cloud.pcd", pcd)
        o3d.io.write_triangle_mesh("coordinate_frame.ply", coordinate_frame)
        return