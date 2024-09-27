# import sys
# sys.path.append('GRUtopia/grutopia_extension/agents/social_navigation_agent')
# import os
# import numpy as np
# import yaml
# import pickle
# from PIL import Image
# import jsonlines
# from modules.mapping.obstacle_map import ObstacleMap
# from GRUtopia.grutopia_extension.agents.social_navigation_agent.agent_utils.geometry_utils import get_intrinsic_matrix, extract_camera_pos_zyxrot

# data_path = '/home/huangwensi/wensi/GRUtopia/images' 
# with open('/home/huangwensi/isaac-sim-4.0.0/GRUtopia/grutopia_extension/agents/social_navigation_agent/memory_config.yaml', "r") as file:
#     memory_config = yaml.load(file, Loader=yaml.FullLoader)
# with open(os.path.join(data_path, 'camera_params.pkl'), 'rb') as f:
#     camera_params = pickle.load(f)

# in_matrix = get_intrinsic_matrix(camera_params)

# min_depth = memory_config['obstacle_map'].pop('min_depth')
# max_depth = memory_config['obstacle_map'].pop('max_depth')
# obstacle_map = ObstacleMap(**memory_config['obstacle_map'])



import os,sys,re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSAC_SIM_DIR = os.path.join(os.path.dirname(ROOT_DIR), "isaac-sim-4.0.0")
sys.path.append(ISSAC_SIM_DIR)

import isaacsim
from omni.isaac.kit import SimulationApp
# from isaacsim import SimulationApp

simulation_app = SimulationApp({'headless': True, 'anti_aliasing': 0, 'renderer': 'RayTracing'}) # !!!

from omni.isaac.core import World
from omni.isaac.sensor import Camera
import numpy as np
from omni.isaac.lab.app import AppLauncher
import argparse
import open3d as o3d

import carb
from pxr import Sdf, Usd, UsdGeom, Vt
# # 命令行参数解析
# parser = argparse.ArgumentParser(description="This script demonstrates different dexterous hands.")
# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app
my_world = World(stage_units_in_meters=1.0)

# 自定义 Camera 类
R_U_TRANSFORM = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
class my_Camera(Camera):
    def __init__(self, prim_path, resolution):
        super().__init__(prim_path=prim_path, resolution=resolution)
        
    def get_intrinsics_matrix(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: the intrinsics of the camera (used for calibration)
        """
        return self._backend_utils.create_tensor_from_list(
           [[554.25616,   0.     , 320.     ],
            [  0.     , 554.25616, 240.     ],
            [  0.     ,   0.     ,   1.     ]],dtype="float32", device=self._device)
    

    def obtain_world_w_cam_u_T(self):
        self.world_w_cam_u_T = UsdGeom.Imageable(self.prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return self.world_w_cam_u_T
# 初始化相机实例
camera = my_Camera(
        prim_path="/World/camera",
        resolution=(640, 480) # (640,480)
    )
camera.set_projection_type('pinhole')

# my_world.scene.add_default_ground_plane()
# my_world.reset()
camera.initialize()

# 载入姿态和深度图数据
main_dir = "/ssd/xiaxinyuan/code/w61-grutopia/logs/sample_episodes_safe/s8pcmisQ38h/id_37"
pose = np.loadtxt(main_dir + "/poses.txt")
save_dir = "/ssd/xiaxinyuan/code/w61-grutopia/logs/sample_episodes/s8pcmisQ38h/id_37"
pcd_save_dir = save_dir + '/pcd'

i = 0
depth_dir = os.path.join(main_dir, "depth")
depth_files = [f for f in os.listdir(depth_dir) if f.endswith(".npy")]
# 提取文件名中的步数并按数字排序
depth_files = sorted(depth_files, key=lambda x: int(re.search(r'(\d+)', x).group()))

rgb_dir = os.path.join(main_dir, "rgb")
rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(".png")]

# 读取第一个深度图文件以获取图像大小
depth_map = np.load(os.path.join(depth_dir, depth_files[0]))
k = 0

PCD_GLOBAL = o3d.geometry.PointCloud()
from vlmaps.application_my.utils import get_dummy_2d_grid, downsample_pc, visualize_pc
def get_pc(camera,depth):
    global PCD_GLOBAL
    grid_2d =  get_dummy_2d_grid(depth.shape[1],depth.shape[0])
    pc = camera.get_world_points_from_image_coords(grid_2d, depth.flatten())
    pc_downsampled = downsample_pc(pc, 150)
    pcd_global = o3d.geometry.PointCloud()
    pcd_global.points = o3d.utility.Vector3dVector(pc_downsampled)
    PCD_GLOBAL+=pcd_global
    # visualize_pc(PCD_GLOBAL,headless=False, save_path = "1.jpg")
    return pc_downsampled

from vlmaps.vlfm.obstacle_map import ObstacleMap

my_map = ObstacleMap(
    min_height= 0.1,
    max_height= 1.7,
    agent_radius=0.25,
    pixels_per_meter=10,
    log_image_dir=save_dir
)
from omni.isaac.core.utils.rotations import quat_to_euler_angles
while simulation_app.is_running():
    my_world.step()
    i+=1
    print(i)
    if i % 10 ==0:
        if(k>=len(depth_files)):
            break
        camera.set_world_pose(pose[k, :3], pose[k, 3:])
        camera_position = camera.get_world_pose()[0]
        camera_rotation = camera.get_world_pose()[1]
        camera_yaw = quat_to_euler_angles(camera_rotation)
        depth_map = np.load(os.path.join(depth_dir, depth_files[k]))
        
        k+=1
        pcd = get_pc(camera,depth_map) # [N,3]
        # np.save(os.path.join(pcd_save_dir, f'pc_{k}.npy'), pcd)
        pcd_filtered = pcd[np.abs(pcd[:,2] - camera_position[2]) < 0.6]
        my_map.update_map_with_pc(
            pc= pcd_filtered,
            camera_position = camera_position,
            camera_orientation= camera_yaw,
            max_depth = 11,
            topdown_fov= 60.0/180.0*np.pi,
            step = k,
            verbose=True
        )

simulation_app.close()








# for step_time in range(500, 1300, 100):
#     depth = np.load(os.path.join(data_path, f'depth_{step_time}.npy'))
#     camera_transform = np.load(os.path.join(data_path, f'camera_transform_{step_time}.npy'))
#     position = np.load(os.path.join(data_path, f'robot_pos_{step_time}.npy'))
#     orientation = np.load(os.path.join(data_path, f'robot_orient_{step_time}.npy'))


#     camera_in = in_matrix
#     topdown_fov = 2 * np.arctan(camera_in[0, 2] / camera_in[0, 0])
#     camera_position, camera_rotation = extract_camera_pos_zyxrot(camera_transform)
#     obstacle_map.update_map(
#                 depth,
#                 camera_in,
#                 camera_transform,
#                 min_depth,
#                 max_depth,
#                 topdown_fov,
#                 verbose=True
#             )