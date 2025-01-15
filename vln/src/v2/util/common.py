import numpy as np
from scipy.ndimage import binary_dilation
from grutopia.core.util.log import log
import math

def create_robot_mask(
    topdown_global_map_camera,
    mask_size=20
):
    height, width = topdown_global_map_camera._camera._resolution
    center_x, center_y = width // 2, height// 2
    # Calculate the top-left and bottom-right coordinates
    half_size = mask_size // 2
    top_left_x = center_x - half_size
    top_left_y = center_y - half_size
    bottom_right_x = center_x + half_size
    bottom_right_y = center_y + half_size

    # Create the mask
    robot_mask = np.zeros((width, height), dtype=np.uint8)
    robot_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1
    return robot_mask

def create_dilation_structure(voxel_size, radius):
    """
    Creates a dilation structure based on the robot's radius.
    """
    radius_cells = int(np.ceil(radius / voxel_size))
    # Create a structuring element for dilation (a disk of the robot's radius)
    dilation_structure = np.zeros((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
    cy, cx = radius_cells, radius_cells
    for y in range(2 * radius_cells + 1):
        for x in range(2 * radius_cells + 1):
            if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius_cells:
                dilation_structure[y, x] = True
    return dilation_structure

def freemap_to_accupancy_map(
    topdown_global_map_camera,
    freemap, 
    dilation_iterations=0,
    voxel_size=0.1,
    agent_radius=0.25,
):
    height, width = topdown_global_map_camera._camera._resolution
    occupancy_map = np.zeros((width, height))
    occupancy_map[freemap == 1] = 2
    occupancy_map[freemap == 0] = 255
    if dilation_iterations > 0:
        dilation_structure = create_dilation_structure(voxel_size, agent_radius)
        for i in range(1, dilation_iterations):
            ob_mask = np.logical_and(occupancy_map!=0, occupancy_map!=2)
            expanded_ob_mask = binary_dilation(ob_mask, structure=dilation_structure, iterations=1)
            occupancy_map[expanded_ob_mask&(np.logical_or(occupancy_map==0,occupancy_map==2))] = 255 - i*10
    return occupancy_map

def check_robot_fall(robot_position, robot_rotation, robots_bottom_z, pitch_threshold=35, roll_threshold=15, height_threshold=0.5):
    from omni.isaac.core.utils.rotations import quat_to_euler_angles
    roll, pitch, yaw = quat_to_euler_angles(robot_rotation, degrees=True)
    # Check if the pitch or roll exceeds the thresholds
    if abs(pitch) > pitch_threshold or abs(roll) > roll_threshold:
        is_fall = True
        log.info(f"Robot falls down!!!")
        log.info(f"Current Position: {robot_position}, Orientation: {roll, pitch, yaw}")
    else:
        is_fall = False
    
    # Check if the height between the robot base and the robot ankle is smaller than a threshold
    robot_ankle_z = robots_bottom_z
    robot_base_z = robot_position[2]
    if robot_base_z - robot_ankle_z < height_threshold:
        is_fall = True
        log.info(f"Robot falls down!!!")
        log.info(f"Current Position: {robot_position}, Orientation: {roll, pitch, yaw}")
    return is_fall

def describe_action(action):
    if action == 1:
        return "向前走0.25米"
    elif action == 2:
        return "左转15°"
    elif action == 3:
        return "右转15°"
    else:
        return "结束"

def get_action_state(obs, action_name):
    for env_idx, (task_name, task) in enumerate(obs.items()):
        for robot_name, robot in task.items():
            action_state = robot[action_name]
            return action_state['finished']
    return False

def check_is_on_track(
    robot_position,
    robot_rotation,
    action,
    action_index,
    real_points,
):
    if action == 1:
        distance = np.linalg.norm(robot_position[:2] - real_points[action_index][:2])
        if distance > 0.5:
            log.info(f"[distance:{round(distance, 2)} > 0.5 ] replanning")
            return False
    else:
        from omni.isaac.core.utils.rotations import quat_to_euler_angles
        _, _, real_yaw = quat_to_euler_angles(robot_rotation)
        yaw_diff = abs(real_yaw - real_points[action_index])
        if yaw_diff > math.pi / 6:
            log.info(f"[yaw_diff: {round(yaw_diff * (180 / math.pi))} 度 > 30 度] replanning")
            return False
    return True