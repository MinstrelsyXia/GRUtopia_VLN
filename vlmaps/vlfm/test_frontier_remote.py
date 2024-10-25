# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# python
import math
import typing

import numpy as np


# internal global constants
_POLE_LIMIT = 1.0 - 1e-6


def rot_matrix_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Quaternion.

    Args:
        mat (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray: quaternion (w, x, y, z).
    """
    if mat.shape == (3, 3):
        tmp = np.eye(4)
        tmp[0:3, 0:3] = mat
        mat = tmp

    q = np.empty((4,), dtype=np.float64)
    t = np.trace(mat)
    if t > mat[3, 3]:
        q[0] = t
        q[3] = mat[1, 0] - mat[0, 1]
        q[2] = mat[0, 2] - mat[2, 0]
        q[1] = mat[2, 1] - mat[1, 2]
    else:
        i, j, k = 0, 1, 2
        if mat[1, 1] > mat[0, 0]:
            i, j, k = 1, 2, 0
        if mat[2, 2] > mat[i, i]:
            i, j, k = 2, 0, 1
        t = mat[i, i] - (mat[j, j] + mat[k, k]) + mat[3, 3]
        q[i + 1] = t
        q[j + 1] = mat[i, j] + mat[j, i]
        q[k + 1] = mat[k, i] + mat[i, k]
        q[0] = mat[k, j] - mat[j, k]
    q *= 0.5 / np.sqrt(t * mat[3, 3])
    return q


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert input quaternion to rotation matrix.

    Args:
        quat (np.ndarray): Input quaternion (w, x, y, z).

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    q = np.array(quat, dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < 1e-10:
        return np.identity(3)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(
        (
            (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
            (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
            (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
        ),
        dtype=np.float64,
    )


def matrix_to_euler_angles(mat: np.ndarray, degrees: bool = False, extrinsic: bool = True) -> np.ndarray:
    """Convert rotation matrix to Euler XYZ extrinsic or intrinsic angles.

    Args:
        mat (np.ndarray): A 3x3 rotation matrix.
        degrees (bool, optional): Whether returned angles should be in degrees.
        extrinsic (bool, optional): True if the rotation matrix follows the extrinsic matrix
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic matrix conventions (equivalent to XYZ ordering).
                   Defaults to True.

    Returns:
        np.ndarray: Euler XYZ angles (intrinsic form) if extrinsic is False and Euler XYZ angles (extrinsic form) if extrinsic is True.
    """
    if extrinsic:
        if mat[2, 0] > _POLE_LIMIT:
            roll = np.arctan2(mat[0, 1], mat[0, 2])
            pitch = -np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])

        if mat[2, 0] < -_POLE_LIMIT:
            roll = np.arctan2(mat[0, 1], mat[0, 2])
            pitch = np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])

        roll = np.arctan2(mat[2, 1], mat[2, 2])
        pitch = -np.arcsin(mat[2, 0])
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
        if degrees:
            roll = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)
        return np.array([roll, pitch, yaw])
    else:
        if mat[0, 2] > _POLE_LIMIT:
            roll = np.arctan2(mat[1, 0], mat[1, 1])
            pitch = np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])

        if mat[0, 2] < -_POLE_LIMIT:
            roll = np.arctan2(mat[1, 0], mat[1, 1])
            pitch = -np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])
        roll = -math.atan2(mat[1, 2], mat[2, 2])
        pitch = math.asin(mat[0, 2])
        yaw = -math.atan2(mat[0, 1], mat[0, 0])

        if degrees:
            roll = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)
        return np.array([roll, pitch, yaw])


def euler_to_rot_matrix(euler_angles: np.ndarray, degrees: bool = False, extrinsic: bool = True) -> np.ndarray:
    """Convert Euler XYZ or ZYX angles to rotation matrix.

    Args:
        euler_angles (np.ndarray): Euler angles.
        degrees (bool, optional): Whether passed angles are in degrees.
        extrinsic (bool, optional): True if the euler angles follows the extrinsic angles
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic angles conventions (equivalent to XYZ ordering).
                   Defaults to True.

    Returns:
        np.ndarray:  A 3x3 rotation matrix in its extrinsic or intrinsic form depends on the extrinsic argument.
    """
    if extrinsic:
        yaw, pitch, roll = euler_angles
    else:
        roll, pitch, yaw = euler_angles
    if degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
    cr = np.cos(roll)
    sr = np.sin(roll)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    if extrinsic:
        return np.array(
            [
                [(cp * cr), ((cr * sp * sy) - (cy * sr)), ((cr * cy * sp) + (sr * sy))],
                [(cp * sr), ((cy * cr) + (sr * sp * sy)), ((cy * sp * sr) - (cr * sy))],
                [-sp, (cp * sy), (cy * cp)],
            ]
        )
    else:
        return np.array(
            [
                [(cp * cy), (-cp * sy), sp],
                [((cy * sr * sp) + (cr * sy)), ((cr * cy) - (sr * sp * sy)), (-cp * sr)],
                [((-cr * cy * sp) + (sr * sy)), ((cy * sr) + (cr * sp * sy)), (cr * cp)],
            ]
        )


def quat_to_euler_angles(quat: np.ndarray, degrees: bool = False, extrinsic: bool = True) -> np.ndarray:
    """Convert input quaternion to Euler XYZ or ZYX angles.

    Args:
        quat (np.ndarray): Input quaternion (w, x, y, z).
        degrees (bool, optional): Whether returned angles should be in degrees. Defaults to False.
        extrinsic (bool, optional): True if the euler angles follows the extrinsic angles
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic angles conventions (equivalent to XYZ ordering).
                   Defaults to True.


    Returns:
        np.ndarray: Euler XYZ angles (intrinsic form) if extrinsic is False and Euler XYZ angles (extrinsic form) if extrinsic is True.
    """
    return matrix_to_euler_angles(quat_to_rot_matrix(quat), degrees=degrees, extrinsic=extrinsic)


def euler_angles_to_quat(euler_angles: np.ndarray, degrees: bool = False, extrinsic: bool = True) -> np.ndarray:
    """Convert Euler angles to quaternion.

    Args:
        euler_angles (np.ndarray):  Euler XYZ angles.
        degrees (bool, optional): Whether input angles are in degrees. Defaults to False.
        extrinsic (bool, optional): True if the euler angles follows the extrinsic angles
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic angles conventions (equivalent to XYZ ordering).
                   Defaults to True.

    Returns:
        np.ndarray: quaternion (w, x, y, z).
    """
    mat = np.array(euler_to_rot_matrix(euler_angles, degrees=degrees, extrinsic=extrinsic))
    return rot_matrix_to_quat(mat)


import os
from vlmaps.vlfm.obstacle_map import ObstacleMap


main_dir = "/ssd/xiaxinyuan/code/w61-grutopia/logs/sample_episodes_safe/s8pcmisQ38h/id_37"
pose = np.loadtxt(main_dir + "/poses.txt")
save_dir = "/ssd/xiaxinyuan/code/w61-grutopia/logs/sample_episodes_safe_2/s8pcmisQ38h/id_37"
pcd_save_dir = save_dir + '/pcd'
pcd_files = [f for f in os.listdir(pcd_save_dir) if f.endswith(".npy")]


my_map = ObstacleMap(
    min_height= 0.1,
    max_height= 1.7,
    agent_radius=0.25,
    pixels_per_meter=10,
    log_image_dir=save_dir,
    dilate_iters= 1
)

k = 0 

def wrap_heading(heading):
    """Ensures input heading is between -180 an 180; can be float or np.ndarray"""
    return (heading + np.pi) % (2 * np.pi) - np.pi

from vlmaps.vlmaps.navigator.navigator import Navigator
from vln.src.local_nav.path_planner import AStarPlanner
import yaml
import time
# navigator_type = 'astar'
navigator_type = 'visgraph'


if navigator_type == 'astar':

    class Args:
        class LLMS:
            def __init__(self, map_grid_num):
                self.map_grid_num = map_grid_num

        def __init__(self, map_grid_num):
            self.llms = self.LLMS(map_grid_num)
    args = Args(map_grid_num=30)
    my_nav = AStarPlanner(args= args, map_width = 1000, map_height = 1000,show_animation=True, verbose = True,max_step=1000)
    goal = (pose[-1,0],pose[-1,1])
    for k in range(len(pcd_files)):
        start_time = time.time()
        # from update obstacle map
        pcd = np.load(pcd_save_dir + '/'+pcd_files[k])
        camera_position = pose[k,:3]
        camera_rotation = pose[k,3:]
        camera_yaw = quat_to_euler_angles(camera_rotation)
        pcd_filtered = pcd[(-0.9 < (camera_position[2]-pcd[:,2])) & ((camera_position[2]-pcd[:,2]) < 1.0)]

        distances = np.linalg.norm(pcd_filtered - camera_position, axis=1)
        
        # 找出所有距离大于阈值的点
        threshold = 0.3
        mask = distances >= threshold
        
        # 使用布尔索引选择满足条件的点
        pcd_new = pcd_filtered[mask]

        np.save('tmp/dist.npy',distances)
        # 打印警告信息（可选，取代逐点判断）
        if np.any(distances < threshold):
            print('origin exists')

        print(np.min((camera_position[2]-pcd[:,2])),np.max((camera_position[2]-pcd[:,2])))
        # camera_pos<pcd: lose constraint; camera_pos >pcd: 
        navigable_map_visual = my_map.update_map_with_pc(
            pc= pcd_new,
            camera_position = camera_position,
            camera_orientation= camera_yaw+np.pi/2,
            max_depth = 11,
            topdown_fov= 60.0/180.0*np.pi,
            step = k,
            verbose=True
        )
        #! camera_yaw+np.pi/2
        rows, cols = np.where(my_map._navigable_map == 0)
        min_row = np.max(np.min(rows)-1,0)
        min_col = np.max(np.min(cols)-1,0)

        start = my_map._xy_to_px(np.array([[camera_position[0],camera_position[1]]]))[0]
        
        # from get_frontier:
        frontiers = my_map.frontiers # array of waypoints
        if len(frontiers) == 0:
            frontiers = np.array([my_map.get_random_free_point()])[0]
            print("error")
        else:
            # randomly pick a frontier:
            # frontier in world coord
            num = frontiers.shape[0]
            idx = np.random.randint(0,num)
            pos = frontiers[idx]
            
            # goal = my_map._xy_to_px(np.array([[pos[0],pos[1]]]))[0]
            goal_xy = my_map._xy_to_px(np.array([[pos[0],pos[1]]]))[0]


        # goal_xy = my_map._xy_to_px(np.array([goal]))[0]
        # path = my_nav.plan_to([start[1],start[0]], [goal_xy[1],goal_xy[0]], vis = True,navigable_map_visual=my_map.nav_map_visual)
        path, find_flag = my_nav.planning(start[0], start[1], goal_xy[0], goal_xy[1], img_save_path = 'tmp/astar_planning.jpg',obs_map = (1-my_map._navigable_map.T==0))

        if find_flag == False:
            continue
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Iteration {k} took {elapsed_time:.2f} seconds")

        # path = np.array(path)
        # path = [path[:,1], path[:,0]]


else:
    my_nav = Navigator()

    goal = (pose[-1,0],pose[-1,1])
    for k in range(len(pcd_files)):
        start_time = time.time()
        # from update obstacle map
        pcd = np.load(pcd_save_dir + '/'+pcd_files[k])
        camera_position = pose[k,:3]
        camera_rotation = pose[k,3:]
        camera_yaw = quat_to_euler_angles(camera_rotation)
        pcd_filtered = pcd[(-0.9 < (camera_position[2]-pcd[:,2])) & ((camera_position[2]-pcd[:,2]) < 1.0)]

        distances = np.linalg.norm(pcd_filtered - camera_position, axis=1)
        
        # 找出所有距离大于阈值的点
        threshold = 0.001
        mask = distances >= threshold
        
        # 使用布尔索引选择满足条件的点
        pcd_new = pcd_filtered[mask]

        # # 打印警告信息（可选，取代逐点判断）
        # if np.any(distances < threshold):
        #     print('origin exists')

        print(np.min((camera_position[2]-pcd[:,2])),np.max((camera_position[2]-pcd[:,2])))
        # camera_pos<pcd: lose constraint; camera_pos >pcd: 
        navigable_map_visual = my_map.update_map_with_pc(
            pc= pcd_new,
            camera_position = camera_position,
            camera_orientation= camera_yaw+np.pi/2,
            max_depth = 11,
            topdown_fov= 60.0/180.0*np.pi,
            step = k,
            verbose=True
        )
        #! camera_yaw+np.pi/2
        rows, cols = np.where(my_map._navigable_map == 0)
        min_row = np.max(np.min(rows)-1,0)
        min_col = np.max(np.min(cols)-1,0)
        my_nav.build_visgraph(my_map._navigable_map,
                            rowmin = 0,
                            colmin = 0,
                            vis = True)
        start = my_map._xy_to_px(np.array([[camera_position[0],camera_position[1]]]))[0]
        
        # from get_frontier:
        frontiers = my_map.frontiers # array of waypoints
        if len(frontiers) == 0:
            frontiers = np.array([my_map.get_random_free_point()])[0]
            print("error")
        else:
            # randomly pick a frontier:
            # frontier in world coord
            num = frontiers.shape[0]
            idx = np.random.randint(0,num)
            pos = frontiers[idx]
            
            # goal = my_map._xy_to_px(np.array([[pos[0],pos[1]]]))[0]
            goal_xy = my_map._xy_to_px(np.array([[pos[0],pos[1]]]))[0]


        # goal_xy = my_map._xy_to_px(np.array([goal]))[0]
        paths = my_nav.plan_to([start[1],start[0]], [goal_xy[1],goal_xy[0]], vis = True,navigable_map_visual=my_map.nav_map_visual)

        paths = np.array(paths)
        paths = np.array([paths[:,1], paths[:,0]]).T

        paths_3d = []
        for path in paths:
            xy = my_map._px_to_xy(np.array([[path[0], path[1]]]))[0]
            paths_3d.append([xy[0],xy[1],1]) # fix height
        paths_3d = np.array(paths_3d)
        print(f"moving from {start} to {goal_xy} on {paths}")
        print(f'moving from {camera_position} to {pos} on {paths_3d}')
        # # test angle:
        # target_rotation = np.arctan2(goal_xy[1] - start[1], goal_xy[0] - start[0]) / np.pi * 180
        
        # yaw = my_map._get_current_angle_on_map(camera_yaw[2] + np.pi / 2) #! camera_yaw[2] + np.pi / 2

        # print("camera yaw, target rotation", yaw,target_rotation)
        # print(((yaw -target_rotation+180) % 360-180))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Iteration {k} took {elapsed_time:.2f} seconds")



        # visualize the angle 
