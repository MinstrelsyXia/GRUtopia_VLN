from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container

# file_path = './GRUtopia/demo/configs/h1_house_mp3d.yaml'
# file_path = './GRUtopia/demo/configs/h1_house.yaml'
file_path = '/ssd/xiaxinyuan/code/w61-grutopia/demo/configs/h1_camera_test.yaml'
sim_config = SimulatorConfig(file_path)

headless = False
webrtc = False

# if is_in_container():
#     headless = True
#     webrtc = True

print(f'headless: {headless}')

env = BaseEnv(sim_config, headless=True, webrtc=webrtc)

task_name = env.config.tasks[0].name
robot_name = env.config.tasks[0].robots[0].name


# camera = env._runner.current_tasks[task_name].robots[robot_name].sensors['pano_camera_0']._camera

from omni.isaac.sensor import Camera
import numpy as np

class my_Camera(Camera):
    def __init__(self, prim_path, resolution):
        super().__init__(prim_path=prim_path, resolution=resolution)
        
    def get_intrinsics_matrix(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: the intrinsics of the camera (used for calibration)
        """
        # if "pinhole" not in self.get_projection_type():
        #     raise Exception("pinhole projection type is not set to be able to use get_intrinsics_matrix method.")
        # focal_length = self.get_focal_length()
        # horizontal_aperture = self.get_horizontal_aperture()
        # vertical_aperture = self.get_vertical_aperture()
        # (width, height) = self.get_resolution()
        # fx = width * focal_length / horizontal_aperture
        # fy = height * focal_length / vertical_aperture
        # cx = width * 0.5
        # cy = height * 0.5
        # return self._backend_utils.create_tensor_from_list(
        #     [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype="float32", device=self._device
        # )
        return self._backend_utils.create_tensor_from_list(
           [[554.25616,   0.     , 320.     ],
            [  0.     , 554.25616, 240.     ],
            [  0.     ,   0.     ,   1.     ]],dtype="float32", device=self._device)

camera = my_Camera(
        prim_path="/World/camera",
        resolution=(640, 480) # (640,480)
    )
camera.set_projection_type('pinhole')

import numpy as np
import open3d as o3d
pose = np.loadtxt("/ssd/xiaxinyuan/code/w61-grutopia/logs/id_5278/poses.txt")
depth_map1 = np.load("/ssd/xiaxinyuan/code/w61-grutopia/logs/id_5278/pano_camera_0_depth_step_24.npy")
depth_map2 = np.load("/ssd/xiaxinyuan/code/w61-grutopia/logs/id_5278/pano_camera_0_depth_step_63.npy")
depth_map3 = np.load("/ssd/xiaxinyuan/code/w61-grutopia/logs/id_5278/pano_camera_0_depth_step_102.npy")
height, width = depth_map1.shape

# Generate a meshgrid of pixel coordinates
x = np.arange(width)
y = np.arange(height)
xx, yy = np.meshgrid(x, y)

# Flatten the meshgrid arrays to correspond to the flattened depth map
xx_flat = xx.flatten()
yy_flat = yy.flatten()

# Combine the flattened x and y coordinates into a 2D array of points
points_2d = np.vstack((xx_flat, yy_flat)).T  # Shape will be (N, 2), where N = height * width
depth_map1=depth_map1.flatten() # (N,)
depth_map2 = depth_map2.flatten()

i = 0

actions = {'h1': {'move_with_keyboard': []}}

global_pcd = o3d.geometry.PointCloud()
def downsample_pc(pc, depth_sample_rate):
    '''
    INput: points:(N,3); rate:downsample rate:int
    Output: downsampled_points:(N/rate,3)
    '''
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
        o3d.visualization.draw_geometries([pcd,coordinate_frame])
        return

while env.simulation_app.is_running():
    # i += 1
    env_actions = []
    env_actions.append(actions)
    obs = env.step(actions=env_actions)

    # if i % 100 == 0:
    #     print(i)
    camera.set_world_pose(pose[0,:3],pose[0,3:])
    points_3d = camera.get_world_points_from_image_coords(points_2d, depth_map1) # (N,3)
    points_3d_downsampled = downsample_pc(points_3d, 100)

    pcd_global = o3d.geometry.PointCloud()
    pcd_global.points = o3d.utility.Vector3dVector(points_3d_downsampled)
    global_pcd+=pcd_global
    visualize_pc(pcd_global,headless,"1.jpg")

    points_2d = camera.get_image_coords_from_world_points(
                points_3d) # (N,3)
    
    # phase 2
    points_2d = np.vstack((xx_flat, yy_flat)).T
    camera.set_world_pose(pose[1,:3],pose[1,3:])
    points_3d = camera.get_world_points_from_image_coords(points_2d, depth_map2) # (N,3)
    points_3d_downsampled = downsample_pc(points_3d, 100)

    pcd_global = o3d.geometry.PointCloud()
    pcd_global.points = o3d.utility.Vector3dVector(points_3d_downsampled)
    global_pcd+=pcd_global  
    visualize_pc(pcd_global,headless,"2.jpg")   
    visualize_pc(global_pcd,headless,"3.jpg")
    points_2d = camera.get_image_coords_from_world_points(
                points_3d) # (N,3)
    
    # phase 3
    points_2d = np.vstack((xx_flat, yy_flat)).T
    camera.set_world_pose(pose[4,:3],pose[4,3:])

    points_3d = camera.get_world_points_from_image_coords(points_2d, depth_map3) # (N,3)
    points_3d_downsampled = downsample_pc(points_3d, 100)

    pcd_global = o3d.geometry.PointCloud()
    pcd_global.points = o3d.utility.Vector3dVector(points_3d_downsampled)
    global_pcd+=pcd_global  
    visualize_pc(pcd_global,headless,"4.jpg")   
    visualize_pc(global_pcd,headless,"5.jpg")

    points_2d = camera.get_image_coords_from_world_points(
                points_3d) # (N,3)

env.simulation_app.close()


