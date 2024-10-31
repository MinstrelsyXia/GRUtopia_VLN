      
# from grutopia.core.config import SimulatorConfig
# from grutopia.core.env import BaseEnv
# from grutopia.core.util.container import is_in_container

# # file_path = './GRUtopia/demo/configs/h1_house_mp3d.yaml'
# # file_path = './GRUtopia/demo/configs/h1_house.yaml'
# file_path = '/ssd/xiaxinyuan/code/w61-grutopia/demo/configs/h1_camera_test.yaml'
# sim_config = SimulatorConfig(file_path)

# headless = True
# webrtc = False

# # if is_in_container():
# #     headless = True
# #     webrtc = True

# print(f'headless: {headless}')

# env = BaseEnv(sim_config, headless=True, webrtc=webrtc)

# task_name = env.config.tasks[0].name
# robot_name = env.config.tasks[0].robots[0].name


# camera = env._runner.current_tasks[task_name].robots[robot_name].sensors['pano_camera_0']._camera
import os,sys

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

# # 命令行参数解析
# parser = argparse.ArgumentParser(description="This script demonstrates different dexterous hands.")
# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app
my_world = World(stage_units_in_meters=1.0)

# 自定义 Camera 类
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

# 初始化相机实例
camera = my_Camera(prim_path="/World/camera", resolution=(256, 256))
# camera.set_projection_type('pinhole')

# my_world.scene.add_default_ground_plane()
# my_world.reset()
camera.initialize()

# 载入姿态和深度图数据
pose = np.loadtxt("/ssd/wangliuyi/code/w61_grutopia_new/logs/sample_episodes_R2R_20240908/train/1LXtFkjw3qL/id_67/poses.txt")
depth_map1 = np.load("/ssd/wangliuyi/code/w61_grutopia_new/logs/sample_episodes_R2R_20240908/train/1LXtFkjw3qL/id_67/pano_camera_0_depth_step_279.npy")
depth_map2 = np.load("/ssd/wangliuyi/code/w61_grutopia_new/logs/sample_episodes_R2R_20240908/train/1LXtFkjw3qL/id_67/pano_camera_0_depth_step_318.npy")
depth_map3 = np.load("/ssd/wangliuyi/code/w61_grutopia_new/logs/sample_episodes_R2R_20240908/train/1LXtFkjw3qL/id_67/pano_camera_0_depth_step_357.npy")
depth_map_list = [depth_map1, depth_map2, depth_map3]
width, height = depth_map1.shape

# 生成像素坐标的网格
x = np.arange(width)
y = np.arange(height)
xx, yy = np.meshgrid(x, y)
xx_flat = xx.flatten()
yy_flat = yy.flatten()
points_2d = np.vstack((xx_flat, yy_flat)).T  # (N, 2)

actions = {'h1': {'move_with_keyboard': []}}
global_pcd = o3d.geometry.PointCloud()

# 点云下采样函数
def downsample_pc(pc, depth_sample_rate):
    shuffle_mask = np.arange(pc.shape[0])
    np.random.shuffle(shuffle_mask)
    shuffle_mask = shuffle_mask[::depth_sample_rate]
    pc = pc[shuffle_mask, :]
    return pc

# 保存点云图像函数
def save_point_cloud_image(pcd, save_path="point_cloud.jpg"):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 0, 1])
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    vis.destroy_window()

# 点云可视化函数
def visualize_pc(pcd, headless, save_path='pc.jpg'):
    if headless:
        save_point_cloud_image(pcd, save_path=save_path)
    else:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, coordinate_frame])
        o3d.io.write_point_cloud("point_cloud.pcd", pcd)
        o3d.io.write_triangle_mesh("coordinate_frame.ply", coordinate_frame)
# 仿真主循环
i = 0
headless = False
step_interval = 50
pc_interval = 20
depth_length = 3
current_depth = 0
set_step = step_interval
while simulation_app.is_running():
    i += 1

    if i%step_interval == 0:
        camera.set_world_pose(pose[current_depth, :3], pose[current_depth, 3:])
        set_step = i
    
    if i%(set_step+pc_interval) == 0:
        depth_map = depth_map_list[current_depth]
        depth_map = depth_map.flatten()
        try:
            points_3d = camera.get_world_points_from_image_coords(points_2d, depth_map)
            points_3d_downsampled = downsample_pc(points_3d, 100)
            pcd_global = o3d.geometry.PointCloud()
            pcd_global.points = o3d.utility.Vector3dVector(points_3d_downsampled)
            global_pcd += pcd_global
            visualize_pc(pcd_global, headless, f"logs/pc/{current_depth}.jpg")
            current_depth += 1
            if current_depth == depth_length:
                current_depth = 0
        except Exception:
            print("Error!")
            continue
    
    my_world.step()
    
    # my_world.step()

    # if i % 100 == 0:
    #     print(i)

    # # 阶段1
    # camera.set_world_pose(pose[0, :3], pose[0, 3:])
    # my_world.step()
    
    # depth_map1 = depth_map1.flatten() # (N,)
    # points_3d = camera.get_world_points_from_image_coords(points_2d, depth_map1)
    # points_3d_downsampled = downsample_pc(points_3d, 100)
    # pcd_global = o3d.geometry.PointCloud()
    # pcd_global.points = o3d.utility.Vector3dVector(points_3d_downsampled)
    # global_pcd += pcd_global
    # visualize_pc(pcd_global, headless, "1.jpg")

    # # 阶段2
    # camera.set_world_pose(pose[1, :3], pose[1, 3:])
    # depth_map2 = depth_map2.flatten() # (N,)
    # points_3d = camera.get_world_points_from_image_coords(points_2d, depth_map2)
    # points_3d_downsampled = downsample_pc(points_3d, 100)
    # pcd_global = o3d.geometry.PointCloud()
    # pcd_global.points = o3d.utility.Vector3dVector(points_3d_downsampled)
    # global_pcd += pcd_global
    # visualize_pc(pcd_global, headless, "2.jpg")
    # visualize_pc(global_pcd, headless, "3.jpg")

    # # 阶段3
    # camera.set_world_pose(pose[4, :3], pose[4, 3:])
    # depth_map3 = depth_map3.flatten() # (N,)
    # points_3d = camera.get_world_points_from_image_coords(points_2d, depth_map3)
    # points_3d_downsampled = downsample_pc(points_3d, 100)
    # pcd_global = o3d.geometry.PointCloud()
    # pcd_global.points = o3d.utility.Vector3dVector(points_3d_downsampled)
    # global_pcd += pcd_global
    # visualize_pc(pcd_global, headless, "4.jpg")
    # visualize_pc(global_pcd, headless, "5.jpg")

simulation_app.close()


    