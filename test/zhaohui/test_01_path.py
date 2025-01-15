from vln.src.dataset.data_utils_multi_env import load_gather_data,load_scene_usd
from grutopia.core.env import BaseEnv
from grutopia.core.config import SimulatorConfig
import numpy as np
from vln.src.local_nav.path_planner import AStarPlanner
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
# from vln.src.discrete.utils import AStarDiscretePlanner
import time

def world_to_pixel(world_pose, camera_pose,aperture,width,height):
    cx, cy = camera_pose[0]*10/aperture*width, -camera_pose[1]*10/aperture*height

    X, Y = world_pose[0]*10/aperture*width, -world_pose[1]*10/aperture*height
    pixel_x = width - (X - cx + width/2)
    pixel_y = Y - cy + height/2

    return [pixel_x, pixel_y]

def vis_nav_path(start_pixel, goal_pixel, points, occupancy_map, img_save_path='path_planning.jpg'):
    cmap = mcolors.ListedColormap(['white', 'green', 'gray', 'black'])
    bounds = [0, 1, 3, 254, 256]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=(10, 10))
    # plt.imshow(occupancy_map, cmap='binary', origin='lower')
    plt.imshow(occupancy_map, cmap=cmap, norm=norm, origin='upper')

    # Plot start and goal points
    plt.plot(start_pixel[1], start_pixel[0], 'ro', markersize=6, label='Start')
    plt.plot(goal_pixel[1], goal_pixel[0], 'bo', markersize=6, label='Goal')

    # Plot the path
    if len(points) > 0:
        path = np.array(points)
        plt.plot(path[:, 1], path[:, 0], 'xb-', linewidth=1, markersize=5, label='Path')

    # Customize the plot
    plt.title('Path planning')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.colorbar(label='Occupancy (0: Free, 1: Occupied)')

    # Save the plot
    plt.savefig(img_save_path, pad_inches=0, bbox_inches='tight', dpi=100)
    print(f"Saved path planning visualization to {img_save_path}")
    plt.close()

class GlobalTopdownconfig:
    def __init__(self):
        self.width=500
        self.height=500
        self.aperture=200
        self.camera_transform_height=0.8
        self.voxel_size=0.1
class Maps:
    def __init__(self):
        self.dilation_iterations = 2
        self.add_dilation = True
        self.global_topdown_config = GlobalTopdownconfig()
        self.agent_radius = 0.25
class Datasets:
    def __init__(self):
        self.mp3d_data_dir = '/ssd/share/Matterport3D/data/v1/scans'
        self.base_data_dir = '/ssd/share/VLN/VLNCE/R2R_VLNCE_v1-3_corrected'
class Settings:
    def __init__(self):
        self.use_llm = True
        self.max_step = 25000
        self.sample_camera_list = ['pano_camera_0']
class Planners:
    def __init__(self):
        self.stair_sample_step = 2
        self.a_star_max_iter = 50000
class Args:
    def __init__(self):
        self.maps = Maps()
        self.log_image_dir = '/ssd/zhaohui/workspace/w61_grutopia_1118/test/zhaohui/images'
        self.datasets = Datasets()
        self.windows_head = False
        self.settings = Settings()
        self.planners = Planners()
        self.headless = True
        self.sample_episode_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_episodes")
        self.name='test_06_discrete'
        self.save_path_planning=False

project_path = "/ssd/zhaohui/workspace/w61_grutopia_0102"
args = Args()
the_scan ='X7HyMhZNoso'
the_path_id_list = [1182]
split='val_unseen'
headless=True
sim_cfg_file = f'{project_path}/vln/configs/sample_episodes_sim_cfg.yaml'
sim_config = SimulatorConfig(sim_cfg_file)
aperture=500
width=500
height=500

#加载数据
scan_map, _ = load_gather_data(args, split, filter_same_trajectory=True, filter_stairs=True)
sample_path_list = []
robot_offset = np.array([0.   , 0.   , 1.05])
for scan_id, path_list in scan_map.items():
    if scan_id != the_scan:
        continue
    for one_path in path_list:
        if one_path['trajectory_id'] not in  the_path_id_list:
            continue
        one_path["start_position"] += robot_offset
        for i, _ in enumerate(one_path["reference_path"]):
            one_path["reference_path"][i] += robot_offset
        sample_path_list.append(one_path)

       
path_zero = sample_path_list[0]

#获取 occupancy_map
scene_asset_path = load_scene_usd(args, the_scan)
sim_config.config.tasks[0].scene_asset_path = scene_asset_path
start_position = np.array(path_zero["start_position"])
start_rotation = np.array(path_zero["start_rotation"])
sim_config.config.tasks[0].robots[0].position = start_position
sim_config.config.tasks[0].robots[0].orientation = start_rotation
env = BaseEnv(sim_config, headless=headless, webrtc=False)
task = env._runner.current_tasks[list(env._runner.current_tasks.keys())[0]]
the_robot = task.robots[list(task.robots.keys())[0]]
the_isaac_robot = the_robot.isaac_robot

from vln.src.local_nav.camera_occupancy_map import CamOccupancyMap
from vln.src.local_nav.global_topdown_map import GlobalTopdownMap
from omni.isaac.core.utils.rotations import quat_to_euler_angles
topdown_map = GlobalTopdownMap(args,the_scan)
occupancy_map = CamOccupancyMap(args, the_robot.sensors['topdown_camera_500'])

# path_planner_old = AStarPlanner(
#     args=args,
#     map_width=width,
#     map_height=height,
#     max_step=50000,
#     windows_head=False,
#     for_llm=False,
#     verbose=False
# )
# path_planner_new = AStarDiscretePlanner(
#     map_width = width,
#     map_height= height,
#     aperture = aperture,
#     step_unit_meter = 0.25,
#     angle_unit=15,
#     max_step=50000,
# )

for the_data in sample_path_list:
    trajectory_id = the_data['trajectory_id']
    reference_path = the_data['reference_path']
    for index in range(len(reference_path) - 1):
        start_position = reference_path[index]
        end_position = reference_path[index + 1]
        task.set_single_robot_poses_without_offset(start_position, start_rotation)
        the_isaac_robot.set_world_velocity(np.zeros(6))
        the_isaac_robot.set_joint_velocities(np.zeros(len(the_isaac_robot.dof_names)))
        the_isaac_robot.set_joint_positions(np.zeros(len(the_isaac_robot.dof_names)))
        the_isaac_robot.set_joint_efforts(np.zeros(len(the_isaac_robot.dof_names)))
        robot_pos = task.get_robot_poses_without_offset()[0]
        occupancy_map.set_world_pose(robot_pos)
        
        for _ in range(80):
            env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=False, render=False)
        robot_position, robot_rotation = the_isaac_robot.get_world_pose()
        camera_pose = occupancy_map.topdown_camera.get_world_pose()[0] - task._offset
        freemap, _ = occupancy_map.get_global_free_map(robot_pos=robot_position, robot_height=1.55, update_camera_pose=False, verbose=False)
        topdown_map.update_map(freemap, camera_pose, verbose=False, env_idx=0, update_map=True)
        _, _, yaw = quat_to_euler_angles(robot_rotation)
        occupancy_map_, _ = topdown_map.get_map(robot_position, return_camera_pose=True)
        
        start_pixel = world_to_pixel(start_position,camera_pose,aperture,width,height)
        goal_pixel = world_to_pixel(end_position,camera_pose,aperture,width,height)

        file_name = f"{trajectory_id}.jpg" 
        vis_nav_path(
            start_pixel, 
            goal_pixel, 
            [], 
            occupancy_map_, 
            img_save_path=os.path.join(f'{project_path}/test/zhaohui/', file_name)
        )
        break

env.simulation_app.close()