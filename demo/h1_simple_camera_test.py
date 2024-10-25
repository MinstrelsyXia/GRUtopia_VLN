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
import os,sys,re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSAC_SIM_DIR = os.path.join(os.path.dirname(ROOT_DIR), "isaac-sim-4.0.0")
sys.path.append(ISSAC_SIM_DIR)

import isaacsim
from omni.isaac.kit import SimulationApp
# from isaacsim import SimulationApp

simulation_app = SimulationApp({'headless': True, 'anti_aliasing': 0, 'renderer': 'RayTracing', 'multi_gpu': False}) # !!!

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

# depth_map1 = np.load(main_dir + "/depth/pano_camera_0_depth_step_8.npy")
# depth_map2 = np.load(main_dir + "/depth/pano_camera_0_depth_step_11.npy")
# depth_map3 = np.load(main_dir + "/depth/pano_camera_0_depth_step_14.npy")
# height, width = depth_map1.shape

# # 生成像素坐标的网格
# x = np.arange(width)
# y = np.arange(height)
# xx, yy = np.meshgrid(x, y)
# xx_flat = xx.flatten()
# yy_flat = yy.flatten()
# points_2d = np.vstack((xx_flat, yy_flat)).T  # (N, 2)

# actions = {'h1': {'move_with_keyboard': []}}
# global_pcd = o3d.geometry.PointCloud()

# # 点云下采样函数
# def downsample_pc(pc, depth_sample_rate):
#     shuffle_mask = np.arange(pc.shape[0])
#     np.random.shuffle(shuffle_mask)
#     shuffle_mask = shuffle_mask[::depth_sample_rate]
#     pc = pc[shuffle_mask, :]
#     return pc

# # 保存点云图像函数
# def save_point_cloud_image(pcd, save_path="point_cloud.jpg"):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     ctr = vis.get_view_control()
#     ctr.set_front([0, 0, -1])
#     ctr.set_lookat([0, 0, 0])
#     ctr.set_up([0, 0, 1])
#     vis.add_geometry(pcd)
#     vis.update_geometry(pcd)
#     vis.poll_events()
#     vis.update_renderer()
#     vis.capture_screen_image(save_path)
#     vis.destroy_window()

# # 点云可视化函数
# def visualize_pc(pcd, headless, save_path='pc.jpg'):
#     if headless:
#         save_point_cloud_image(pcd, save_path=save_path)
#     else:
#         coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
#         # o3d.visualization.draw_geometries([pcd, coordinate_frame])
#         o3d.io.write_point_cloud("point_cloud.pcd", pcd)
#         o3d.io.write_triangle_mesh("coordinate_frame.ply", coordinate_frame)
# # 仿真主循环
# i = 0
# headless = False
# while simulation_app.is_running():
#     i += 1
#     print(i)
#     my_world.step()
#     world_w_cam_u_T = camera.obtain_world_w_cam_u_T()
#     print(world_w_cam_u_T)
#     if i == 50:
#         camera.set_world_pose(pose[0, :3], pose[0, 3:])
#         print(pose)
#         print(camera.get_world_pose())
    
#     if i ==100:
#         depth_map1 = depth_map1.flatten() # (N,)
#         points_3d = camera.get_world_points_from_image_coords(points_2d, depth_map1)
#         points_3d_downsampled = downsample_pc(points_3d, 100)
#         pcd_global = o3d.geometry.PointCloud()
#         pcd_global.points = o3d.utility.Vector3dVector(points_3d_downsampled)
#         global_pcd += pcd_global
#         visualize_pc(pcd_global, headless, "1.jpg")

#     if i == 150 :
#         camera.set_world_pose(pose[1, :3], pose[1, 3:])
#         print(pose)
#         print(camera.get_world_pose())

#     if i ==200:
#         depth_map2 = depth_map2.flatten() # (N,)
#         points_3d = camera.get_world_points_from_image_coords(points_2d, depth_map2)
#         points_3d_downsampled = downsample_pc(points_3d, 100)
#         pcd_global = o3d.geometry.PointCloud()
#         pcd_global.points = o3d.utility.Vector3dVector(points_3d_downsampled)
#         global_pcd += pcd_global
#         visualize_pc(pcd_global, headless, "2.jpg")
#         visualize_pc(global_pcd, headless, "3.jpg")

#     if i == 250:
#         camera.set_world_pose(pose[2, :3], pose[2, 3:])
#         print(pose)
#         print(camera.get_world_pose())
    
#     if i == 300:
#         depth_map3 = depth_map3.flatten() # (N,)
#         points_3d = camera.get_world_points_from_image_coords(points_2d, depth_map3)
#         points_3d_downsampled = downsample_pc(points_3d, 100)
#         pcd_global = o3d.geometry.PointCloud()
#         pcd_global.points = o3d.utility.Vector3dVector(points_3d_downsampled)
#         global_pcd += pcd_global
#         visualize_pc(pcd_global, headless, "4.jpg")
#         visualize_pc(global_pcd, headless, "5.jpg")
#         break
# simulation_app.close()

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



PCD_GLOBAL = o3d.geometry.PointCloud()


def test_pc(camera,depth):
    global PCD_GLOBAL
    grid_2d =  get_dummy_2d_grid(depth.shape[1],depth.shape[0])
    pc = camera.get_world_points_from_image_coords(grid_2d, depth.flatten())
    pc_downsampled = downsample_pc(pc, 150)
    pcd_global = o3d.geometry.PointCloud()
    pcd_global.points = o3d.utility.Vector3dVector(pc_downsampled)
    PCD_GLOBAL+=pcd_global
    visualize_pc(PCD_GLOBAL,headless=False, save_path = "1.jpg")

headless = False
i = 0
depth_dir = os.path.join(main_dir, "depth")
depth_files = [f for f in os.listdir(depth_dir) if f.endswith(".npy")]
# 提取文件名中的步数并按数字排序
depth_files = sorted(depth_files, key=lambda x: int(re.search(r'(\d+)', x).group()))

# 读取第一个深度图文件以获取图像大小
depth_map = np.load(os.path.join(depth_dir, depth_files[0]))
k = 0
while simulation_app.is_running():
    my_world.step()
    i+=1
    print(i)
    if i % 10 ==0:
        if(k>=len(depth_files)):
            break
        camera.set_world_pose(pose[k, :3], pose[k, 3:])
        depth_map = np.load(os.path.join(depth_dir, depth_files[k]))
        k+=1
        test_pc(camera,depth_map)

simulation_app.close()