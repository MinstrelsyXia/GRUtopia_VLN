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
# from omni.isaac.lab.app import AppLauncher
import argparse
import open3d as o3d

import carb
from pxr import Sdf, Usd, UsdGeom, Vt
import omni.replicator.core as rep
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

class FineCamera(Camera):

    def get_render_product(self):
        return self._render_product

    def get_view_matrix_ros(self):
        """3D points in World Frame -> 3D points in Camera Ros Frame

        Returns:
            np.ndarray: the view matrix that transforms 3d points in the world frame to 3d points in the camera axes
                        with ros camera convention.
        """
        R_U_TRANSFORM = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        width, height = self.get_resolution()
        rp = rep.create.render_product(self.prim_path, resolution=(width, height))
        _camera_params = rep.annotators.get('CameraParams')
        _camera_params.attach(rp)
        camera_params = _camera_params.get_data()
        try:
            world_w_cam_u_T = self._backend_utils.transpose_2d(
                self._backend_utils.convert(
                    np.linalg.inv(camera_params['cameraViewTransform'].reshape(4, 4)),
                    dtype='float32',
                    device=self._device,
                    indexed=True,
                ))
        except np.linalg.LinAlgError:
            world_w_cam_u_T = self._backend_utils.transpose_2d(
                self._backend_utils.convert(
                    UsdGeom.Imageable(self.prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                    dtype='float32',
                    device=self._device,
                    indexed=True,
                ))
        r_u_transform_converted = self._backend_utils.convert(R_U_TRANSFORM,
                                                              dtype='float32',
                                                              device=self._device,
                                                              indexed=True)
        return self._backend_utils.matmul(r_u_transform_converted, self._backend_utils.inverse(world_w_cam_u_T))
    
    def get_pointcloud(self,depth) -> np.ndarray:
        im_height, im_width = depth.shape[0], depth.shape[1]

        ww = np.linspace(0, im_width - 1, im_width)
        hh = np.linspace(0, im_height - 1, im_height)
        xmap, ymap = np.meshgrid(ww, hh)

        points_2d = np.column_stack((xmap.ravel(), ymap.ravel()))

        # Directly use this function from the camera class to do this.
        pointcloud = self.get_world_points_from_image_coords(points_2d, depth.flatten())

        return pointcloud

camera_type = "fine_camera"
if camera_type == "my_camera":
    camera = my_Camera(
            prim_path="/World/camera",
            resolution=(640, 480) # (640,480)
        )
    camera.set_projection_type('pinhole')

    # my_world.scene.add_default_ground_plane()
    # my_world.reset()
    camera.initialize()
if camera_type == "fine_camera":
    camera = FineCamera(
            prim_path="/World/camera",
            resolution=(640, 480) # (640,480)
        )
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()


# 载入姿态和深度图数据
# main_dir = "sample_episodes_safe/s8pcmisQ38h/id_37"
main_dir = "/g0433_data/xiaxinyuan/code/w61-grutopia/logs/sample_episodes_safe/s8pcmisQ38h/id_2606"
pose = np.loadtxt(main_dir + "/poses.txt")

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


def test_pc(camera,depth,camera_type='my_camera'):
    global PCD_GLOBAL
    if camera_type == 'my_camera':
        grid_2d =  get_dummy_2d_grid(depth.shape[1],depth.shape[0])
        pc = camera.get_world_points_from_image_coords(grid_2d, depth.flatten())

    else:
        pc = camera.get_pointcloud(depth)
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
        test_pc(camera,depth_map,camera_type)

simulation_app.close()