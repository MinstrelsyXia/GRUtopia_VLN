import numpy as np
import os
from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from vln import PROJECT_ROOT_PATH
from vln.src.v2.util.common import (
    check_robot_fall, 
    load_data,
    load_scene_usd,
    create_robot_mask,
    freemap_to_accupancy_map,
)
from vln.src.v2.util.path_plan import (
    world_to_pixel,
    vis_nav_path,
)
from datetime import datetime

def calc_xy_index(position, min_pos, resolution):
    return round((position - min_pos) / resolution)

def get_global_map(
    robot,
    robot_name,
    robot_height,
    dilation_iterations=0,
    voxel_size=0.1,
    agent_radius=0.25,
):  
    topdown_global_map_camera = robot.sensors['topdown_camera_500']
    # 获取 free_map
    min_height = robot_height
    max_height = robot_height + 0.8
    data_info = topdown_global_map_camera.get_data()
    depth = np.array(data_info["depth"])
    flat_surface_mask = np.ones_like(depth, dtype=bool)
    if robot_name == 'h1':
        depth_mask = ((depth >= min_height) & (depth < max_height)) | ((depth <= 0.5) & (depth > 0.02))
    elif robot_name == 'aliengo':
        base_height = robot.get_robot_base().get_world_pose()[0][2]
        foot_height = robot.get_ankle_height()
        min_height = base_height - foot_height + 0.05
        depth_mask = ((depth >= min_height) & (depth < max_height))
    robot_mask = create_robot_mask(topdown_global_map_camera)
    free_map = np.zeros_like(depth, dtype=int)
    free_map[flat_surface_mask & depth_mask] = 1
    free_map[robot_mask == 1] = 1
    accupancy_map = freemap_to_accupancy_map(
        topdown_global_map_camera=topdown_global_map_camera,
        freemap=free_map, 
        dilation_iterations=dilation_iterations,
        voxel_size=voxel_size,
        agent_radius=agent_radius,
    )
    return accupancy_map


# 需要处理的 path_id 列表
path_id_list = [
    319
]

headless=False
split='train'
the_scan ='E9uDoFAP3SH'
project_path = PROJECT_ROOT_PATH
base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
mp3d_data_dir = f"{project_path}/../Matterport3D/data/v1/scans"
aperture=200
robot_name = 'aliengo' # h1 / aliengo
if robot_name == "aliengo":
    robot_offset = np.array([0.   , 0.   , 0.50])
else:
    robot_offset = np.array([0.   , 0.   , 1.05])
# 获取数据
data_map = load_data(base_data_dir,split,True,True,)

filtered_path_list = []
for scan_id, path_list in data_map.items():
    if scan_id != the_scan:
        continue
    for one_path in path_list:
        path_id = one_path['trajectory_id']
        if path_id not in path_id_list:
            continue
        one_path["start_position"] += robot_offset
        for i, _ in enumerate(one_path["reference_path"]):
            one_path["reference_path"][i] += robot_offset
        filtered_path_list.append(one_path)

# 加载场景
sim_cfg_file = f'{project_path}/vln/configs/sim_cfg_policy_{robot_name}_eval.yaml'
sim_config = SimulatorConfig(sim_cfg_file)
fall_height_threshold = sim_config.config_dict['tasks'][0]['robots'][0]['fall_height_threshold']
robot_height = sim_config.config_dict['tasks'][0]['robots'][0]['robot_height']
scene_asset_path = load_scene_usd(mp3d_data_dir, the_scan)
sim_config.config.tasks[0].scene_asset_path = scene_asset_path
path_zero=filtered_path_list[0]
start_position = np.array(path_zero["start_position"])
start_rotation = np.array(path_zero["start_rotation"])
sim_config.config.tasks[0].robots[0].position = start_position
sim_config.config.tasks[0].robots[0].orientation = start_rotation
env = BaseEnv(sim_config, headless=headless, webrtc=False)
from omni.kit.actions.core import get_action_registry
get_action_registry().get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera").execute()

the_task = env._runner.current_tasks[list(env._runner.current_tasks.keys())[0]]
the_robot = the_task.robots[list(the_task.robots.keys())[0]]
the_isaac_robot = the_robot.isaac_robot

for path in filtered_path_list:
    the_path_id = path['trajectory_id']
    # 设置机器人位置
    start_position = np.array(path["start_position"])
    start_rotation = np.array(path["start_rotation"])
    

    the_task.set_single_robot_poses_without_offset(start_position, start_rotation)
    the_isaac_robot.set_world_velocity(np.zeros(6))
    the_isaac_robot.set_joint_velocities(np.zeros(len(the_isaac_robot.dof_names)))
    the_isaac_robot.set_joint_positions(np.zeros(len(the_isaac_robot.dof_names)))
    the_isaac_robot.set_joint_efforts(np.zeros(len(the_isaac_robot.dof_names)))

    reference_path = path['reference_path']
    reference_path = reference_path[1:]
    for index in range(len(reference_path) - 1):
        start_position = reference_path[index]
        end_position = reference_path[index + 1]
        print(f"start_position: {start_position}")
        print(f"end_position: {start_position}")
        the_task.set_single_robot_poses_without_offset(start_position, start_rotation)
        the_isaac_robot.set_world_velocity(np.zeros(6))
        the_isaac_robot.set_joint_velocities(np.zeros(len(the_isaac_robot.dof_names)))
        the_isaac_robot.set_joint_positions(np.zeros(len(the_isaac_robot.dof_names)))
        the_isaac_robot.set_joint_efforts(np.zeros(len(the_isaac_robot.dof_names)))
        robot_pos = the_task.get_robot_poses_without_offset()[0]

        for _ in range(240):
            env.step(actions=[{robot_name:{'stand_still': []}}], add_rgb_subframes=False, render=False)

        global_map = get_global_map(
            the_robot,
            robot_name,
            robot_height,
            dilation_iterations=2,
        )
        topdown_global_map_camera = the_robot.sensors['topdown_camera_500']
        height, width = topdown_global_map_camera._camera._resolution
        camera_pose = topdown_global_map_camera.get_world_pose()[0] - the_task._offset
        robot_position, robot_rotation = the_task.get_robot_poses_without_offset()
        from omni.isaac.core.utils.rotations import quat_to_euler_angles
        _, _, yaw = quat_to_euler_angles(robot_rotation)
        start_pixel = world_to_pixel(robot_position,camera_pose,aperture,width,height)
        goal_pixel = world_to_pixel(end_position,camera_pose,aperture,width,height)

        start_node = [
            calc_xy_index(start_pixel[0],0,1),
            calc_xy_index(start_pixel[1],0,1),
        ]

        end_node = [
            calc_xy_index(goal_pixel[0],0,1),
            calc_xy_index(goal_pixel[1],0,1),
        ]
        print(f"start_node: {start_node}")
        print(f"end_node: {end_node}")


        if global_map[end_node[0],end_node[1]] == 255:
            print("goal_in_obstacle")
        file_name = f"path_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg" 
        vis_nav_path(
            start_node, 
            end_node, 
            [], 
            global_map, 
            img_save_path=os.path.join(f'{project_path}/test/zhaohui/', file_name)
        )
        while True:
            env.step(actions=[{robot_name:{'stand_still': []}}], add_rgb_subframes=False, render=False)
    
if(hasattr(env, 'simulation_app')):
    env.simulation_app.close()