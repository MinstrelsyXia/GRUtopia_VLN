

# from vln.src.local_nav.sementic_map import BEVSemMap
# identical to semantic map/main.py
import os, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Set
import hydra
from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
import numpy as np
from omegaconf import DictConfig
import torch
import gdown
import open3d as o3d
import h5py
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "vlmaps"))
# print(sys.path)

from vlmaps.vlmaps.utils.lseg_utils import get_lseg_feat
from vlmaps.vlmaps.utils.mapping_utils import (
    save_3d_map,
    cvt_pose_vec2tf,
    load_depth_img,
    load_depth_npy,
    depth2pc,
    transform_pc,
    base_pos2grid_id_3d,
    project_point,
    get_sim_cam_mat,
    depth2pc_real_world
)
from vlmaps.vlmaps.utils.index_utils import find_similar_category_id
from vlmaps.vlmaps.lseg.modules.models.lseg_net import LSegEncNet
from vlmaps.vlmaps.map.vlmap import VLMap
from vlmaps.vlmaps.utils.clip_utils import get_lseg_score
from vlmaps.vlmaps.utils.visualize_utils import (
    pool_3d_label_to_2d,
    pool_3d_rgb_to_2d,
    visualize_rgb_map_3d,
    visualize_masked_map_2d,
    visualize_heatmap_2d,
    visualize_heatmap_3d,
    visualize_masked_map_3d,
    get_heatmap_from_mask_2d,
    get_heatmap_from_mask_3d,
)
from vlmaps.vlmaps.utils.matterport3d_categories import mp3dcat

import isaacsim
from omni.isaac.kit import SimulationApp
# simulation_app = SimulationApp({'headless': True, 'anti_aliasing': 0, 'renderer': 'RayTracing'})

from omni.isaac.core import World
from omni.isaac.sensor import Camera
import numpy as np
from omni.isaac.lab.app import AppLauncher

import clip

from grutopia.core.util.log import log

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
    

def load_depth_npy(depth_filepath: Union[Path, str]):
    with open(depth_filepath, "rb") as f:
        depth = np.load(f)
    return depth

def load_3d_map(map_path: Union[Path, str]):
    with h5py.File(map_path, "r") as f:
        mapped_iter_list = f["mapped_iter_list"][:].tolist()
        grid_feat = f["grid_feat"][:]
        grid_pos = f["grid_pos"][:]
        weight = f["weight"][:]
        occupied_ids = f["occupied_ids"][:]
        grid_rgb = f["grid_rgb"][:]
        pcd_min = f["pcd_min"][:]
        pcd_max = f["pcd_max"][:]
        cs = f["cs"][()]
        return (mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, pcd_min, pcd_max, cs)

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


from vlmaps.vlmaps.map import VLMap
class TMP(VLMap):
    def __init__(self, data_dir,map_config):
        super().__init__(map_config,data_dir)
        self.robot_z = map_config.robot_z
        self.cs = map_config.cs
        
    def _setup_paths(self, data_dir: Union[Path, str]) -> None:
        #! modified Map's function

        self.data_dir = Path(data_dir)
        self.rgb_dir = self.data_dir / "rgb"
        self.depth_dir = self.data_dir / "depth"
        self.segmentation_dir = self.data_dir / "segmentation"
        if not os.path.exists(self.segmentation_dir):
            os.makedirs(self.segmentation_dir)
        # self.rgb_dir = self.data_dir 
        # self.depth_dir = self.data_dir
        # self.semantic_dir = self.data_dir / "semantic"
        self.pose_path = self.data_dir / "poses.txt"
        try:
            # self.rgb_paths = sorted(self.rgb_dir.glob("*.png"))
            # self.depth_paths = sorted(self.depth_dir.glob("*.npy"))
            # self.semantic_paths = sorted(self.semantic_dir.glob("*.npy"))
            self.rgb_paths = sorted(self.rgb_dir.glob("*.png"),key=lambda path: int(path.stem.split('_')[-1]))
            self.depth_paths = sorted(self.depth_dir.glob("*.npy"),key=lambda path: int(path.stem.split('_')[-1]))
        except FileNotFoundError as e:
            print(e)
    def _init_clip(self, clip_version="ViT-B/32"):
        if hasattr(self, "clip_model"):
            print("clip model is already initialized")
            return
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.clip_version = clip_version
        self.clip_feat_dim = {
            "RN50": 1024,
            "RN101": 512,
            "RN50x4": 640,
            "RN50x16": 768,
            "RN50x64": 1024,
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
        }[self.clip_version]
        print("Loading CLIP model...")
        self.clip_model, self.preprocess = clip.load(self.clip_version)  # clip.available_models()
        self.clip_model.to(self.device).eval()

    def _init_lseg(self):
        crop_size = 480  # 480
        base_size = 640  # 520: 无论多大的照片，长边都会变成480
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        lseg_model = LSegEncNet("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
        model_state_dict = lseg_model.state_dict()
        # checkpoint_dir = Path(__file__).resolve().parents[1] / "lseg" / "checkpoints"
        checkpoint_dir = Path(__file__).resolve().parents[1] / "vlmaps" /"lseg" / "checkpoints"
        checkpoint_path = checkpoint_dir / "demo_e200.ckpt"
        os.makedirs(checkpoint_dir, exist_ok=True)
        if not checkpoint_path.exists():
            print("Downloading LSeg checkpoint...")
            # the checkpoint is from official LSeg github repo
            # https://github.com/isl-org/lang-seg
            checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
            gdown.download(checkpoint_url, output=str(checkpoint_path))

        pretrained_state_dict = torch.load(checkpoint_path, map_location=self.device)
        pretrained_state_dict = {k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()}
        model_state_dict.update(pretrained_state_dict)
        lseg_model.load_state_dict(pretrained_state_dict)

        lseg_model.eval()
        lseg_model = lseg_model.to(self.device)

        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        lseg_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.clip_feat_dim = lseg_model.out_c
        return lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std

    def _init_map(self, pcd_min: np.ndarray, pcd_max: np.ndarray, cs: float, map_path: Path) -> Tuple:
            """
            initialize a voxel grid of size grid_size = (pcd_max - pcd_min) / cs + 1
            """
            grid_size = np.ceil((pcd_max - pcd_min) / cs + 1).astype(int)  # original: col, height, row; isaacsim: row,column, height
            self.grid_size = grid_size
            occupied_ids = -1 * np.ones(grid_size[[0, 1, 2]], dtype=np.int32)
            grid_feat = np.zeros((grid_size[0] * grid_size[1], self.clip_feat_dim), dtype=np.float32)
            grid_pos = np.zeros((grid_size[0] * grid_size[1], 3), dtype=np.int32)
            weight = np.zeros((grid_size[0] * grid_size[1]), dtype=np.float32)
            grid_rgb = np.zeros((grid_size[0] * grid_size[1], 3), dtype=np.uint8)
            # init the map related variables
            mapped_iter_set = set()
            mapped_iter_list = list(mapped_iter_set)
            max_id = 0

            # check if there is already saved map
            # if os.path.exists(map_path):
            #     (mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, pcd_min, pcd_max, cs) = (
            #         self.load_3d_map()
            #     )
            #     mapped_iter_set = set(mapped_iter_list)
            #     max_id = grid_feat.shape[0]
            #     self.pcd_min = pcd_min

            return grid_feat, grid_pos, weight, occupied_ids, grid_rgb, mapped_iter_set, max_id

    def load_3d_map(self):
        (
            self.mapped_iter_list,
            self.grid_feat,
            self.grid_pos,
            self.weight,
            self.occupied_ids,
            self.grid_rgb,
            self.pcd_min,
            self.pcd_max,
            self.cs,
            ) = load_3d_map(self.map_save_path)
        return (
            self.mapped_iter_list,
            self.grid_feat,
            self.grid_pos,
            self.weight,
            self.occupied_ids,
            self.grid_rgb,
            self.pcd_min,
            self.pcd_max,
            self.cs,
            )

    # def update_map(self, mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, pcd_min, pcd_max, cs):
    #     '''
    #     If need to save path: self.save_3d_map() & self.load_3d_map()
    #     If not needed, use self.update_map(...)
    #     '''
    #     self.mapped_iter_list = mapped_iter_list
    #     self.grid_feat = grid_feat
    #     self.grid_pos = grid_pos
    #     self.weight = weight
    #     self.occupied_ids = occupied_ids
    #     self.grid_rgb = grid_rgb
    #     self.pcd_min = pcd_min
    #     self.pcd_max = pcd_max
    #     self.cs = cs

    def update_map(self,old_map,new_map_coord,delta_map_coord):
        '''
        old_map: shape: [x,y,z,:];[x,y,:]
        new_map: shape: [new_map_coord]
        '''
        if (len(new_map)==3):
            new_map = np.zeros(new_map_coord)
            new_map[:(old_map.shape[0]+delta_map_coord[0]),:(old_map.shape[1]+delta_map_coord[1]),:] = old_map
        else:
            new_map = np.zeros(new_map_coord)
            new_map[:(old_map.shape[0]+delta_map_coord[0]),:(old_map.shape[1]+delta_map_coord[1]),:(old_map.shape[2]+delta_map_coord[2]),:] = old_map
        return new_map

    def init_map(self):
        self._setup_paths(self.data_dir)
        self._init_clip()
        self.map_save_dir = self.data_dir / "vlmap_cam"

        # relative_path = self.data_dir.relative_to('/ssd/share/w61')

        # # 获取当前主目录
        # home_dir = Path('/ssd/xiaxinyuan/code/w61-grutopia/logs')

        # # 拼接形成新的路径
        # new_path = home_dir / relative_path
        os.makedirs(self.map_save_dir, exist_ok=True)
        self.map_save_path = self.map_save_dir / "vlmaps_cam.h5df"
        
    def convert_world_to_map(self, point_cloud):
        # Note that the pointclouds have the world corrdinates that some values are very negative
        # We need to convert it into the map coordinates
        if len(point_cloud)==0:
            log.error(f"The shape of point cloud is not correct. The shape is {point_cloud.shape}.")
            return None
        point_cloud[...,:2] = point_cloud[..., :2] - self.init_world_pos[:2]

        return point_cloud
    
    def _retrive_map(self, pcd_min: np.ndarray, pcd_max: np.ndarray, cs: float, map_path: Path) -> Tuple:
            """
            retrive the saved map route
            """

            # check if there is already saved map
            if os.path.exists(map_path):
                (mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, pcd_min, pcd_max, cs) = (
                    self.load_3d_map(self.map_save_path)
                )
                mapped_iter_set = set(mapped_iter_list)
                max_id = grid_feat.shape[0]
                self.pcd_min = pcd_min
            else:
                grid_size = np.ceil((pcd_max - pcd_min) / cs + 1).astype(int)  # col, height, row
                self.grid_size = grid_size
                occupied_ids = -1 * np.ones(grid_size[[0, 1, 2]], dtype=np.int32)
                grid_feat = np.zeros((grid_size[0] * grid_size[2], self.clip_feat_dim), dtype=np.float32)
                grid_pos = np.zeros((grid_size[0] * grid_size[2], 3), dtype=np.int32)
                weight = np.zeros((grid_size[0] * grid_size[2]), dtype=np.float32)
                grid_rgb = np.zeros((grid_size[0] * grid_size[2], 3), dtype=np.uint8)
                # init the map related variables
                mapped_iter_set = set()
                mapped_iter_list = list(mapped_iter_set)
                max_id = 0

            return grid_feat, grid_pos, weight, occupied_ids, grid_rgb, mapped_iter_set, max_id
    def cvt_global_to_pixel_map(self,pc_global):
        '''
        pc_global: [N,3]
        '''
        x_max = max(self.min_max[0,0],pc_global[:,0])
        x_min = min(self.min_max[0,1],pc_global[:,0])
        y_max = max(self.min_max[1,0],pc_global[:,1])
        y_min = min(self.min_max[1,1],pc_global[:,1])
        z_max = max(self.min_max[2,0],pc_global[:,2])
        z_min = min(self.min_max[2,1],pc_global[:,2])
        new_row = (max(self.x_max,pc_global[:,0])-min(self.x_min,pc_global[:,0]))/self.voxel_size +1
        new_height = (max(self.y_max,pc_global[:,1])-min(self.y_min,pc_global[:,1]))/self.voxel_size +1
        new_col = (max(self.z_max,pc_global[:,2])-min(self.z_min,pc_global[:,2]))/self.voxel_size +1

        delta_row = min(self.x_min,pc_global[:,0])/self.voxel_size
        delta_height = min(self.y_min,pc_global[:,1])/self.voxel_size
        delta_col = min(self.z_min,pc_global[:,2])/self.voxel_size
        # map the local map to a global map
        pc_pixel_map = np.zeros(pc_global.shape)

        for i in range(pc_global.shape[0]):
            x = int((pc_global[i,0]-min(self.x_min,pc_global[:,0]))/self.voxel_size)
            y = int((pc_global[i,1]-min(self.y_min,pc_global[:,1]))/self.voxel_size)
            z = int((pc_global[i,2]-min(self.z_min,pc_global[:,2]))/self.voxel_size)
            pc_pixel_map[i] = [x,y,z]
        return pc_pixel_map,[new_row,new_height,new_col],[delta_row,delta_height,delta_col],[[x_max,x_min],[y_max,y_min],[z_max,z_min]]
    
    def _update_semantic_map(self,camera,rgb,depth,robot_bottom_z):
        '''
        build semantic map locally, given sync camera
        
        '''
        max_depth = 10
        downsample_rate = 150
        grid_2d =  get_dummy_2d_grid(depth.shape[1],depth.shape[0])
        # depth
        depth[depth > max_depth] = 0
        # rgb
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # pose
        camera_position, camera_orientation = camera.get_world_pose()


        pc=camera.get_world_points_from_image_coords(grid_2d, depth)
        downsampled_cloud = downsample_pc(pc,downsample_rate)
        downsampled_cloud = downsampled_cloud[np.isfinite(downsampled_cloud).all(axis=1)]
        point_to_consider = downsampled_cloud = self.convert_world_to_map(downsampled_cloud)
        # adjusted_coords = (downsampled_cloud[:, :2]/self.voxel_size + [self.quadtree_width/2, self.quadtree_height/2]).astype(int) 
        # adjusted_coords_with_z = np.hstack((adjusted_coords, downsampled_cloud[:,2].reshape(-1,1)))
        # point_to_consider = adjusted_coords_with_z[(adjusted_coords_with_z[:, 0] < self.quadtree_width) & (adjusted_coords_with_z[:, 1] < self.quadtree_height)] 

        pc = point_within_robot_z = point_to_consider[(point_to_consider[:,2]>=(robot_bottom_z+self.robot_z[0])) & (point_to_consider[:,2]<=(robot_bottom_z+self.robot_z[1]))].astype(int) # points that are within the robot's height range (occupancy)

        pc_image = camera.get_image_coords_from_world_points(point_within_robot_z) # [N,2]:[[x,y],...]
        pc_image = pc_image.astype(int)

        if (self.lseg_model is None):
            self.lseg_model, self.lseg_transform, self.crop_size, self.base_size, self.norm_mean, self.norm_std = self._init_lseg()

        pix_feats ,pix_mask= get_lseg_feat(
                            self.lseg_model, rgb, ["example"], self.lseg_transform, self.device, self.crop_size, self.base_size, self.norm_mean, self.norm_std
                        ) # [B,D,H,W];[H,W]

        pc_pixel_map, new_map_coord, delta_map_coord, min_max = self.cvt_global_to_pixel_map(pc)
        grid_feat, grid_pos, weight, occupied_ids, grid_rgb, mapped_iter_set, max_id = self._retrive_map(
        self.pcd_min, self.pcd_max, self.voxel_size, self.map_save_path)
        # mapped_iter_set: literally empty
        grid_feat = self.update_map(grid_feat,new_map_coord,delta_map_coord)
        grid_pos = self.update_map(grid_pos,new_map_coord,delta_map_coord)
        weight = self.update_map(weight,new_map_coord,delta_map_coord)
        grid_rgb = self.update_map(grid_rgb,new_map_coord,delta_map_coord)
        occupied_ids = self.update_map(occupied_ids,new_map_coord,delta_map_coord)


        #  pc_image: [py,px]:pixel coord in 2d-image; 
        #  row,height column: pixel coord in 3d-map
        radial_dist_sq = np.array([np.sum(np.square(pc_dd-camera_position)) for pc_dd in pc]).T
        sigma_sq = 0.6
        alphas = np.exp(-radial_dist_sq / (2 * sigma_sq))

        pbar = tqdm(zip(pc_image,pc,alphas), total=pc.shape[0], desc="Point wise mapping")
        for (p_i, p, alpha) in (pbar): # p: [x,y,z]
            py, px = p_i[0],p_i[1] # px actually means py
            row, column, height = np.round(((p - self.pcd_min) / self.cs)).astype(int)

            if max_id >= grid_feat.shape[0]:
                grid_feat, grid_pos, weight, grid_rgb = self._reserve_map_space(
                    grid_feat, grid_pos, weight, grid_rgb
                )

            if not (py < 0 or px < 0 or py >= pix_feats.shape[2] or px >= pix_feats.shape[3]):
                rgb_v = rgb[py,px] # for given point p, its label and feature is pix_feats[px,py] and pix_mask[px,py] 
                feat = pix_feats[0, :, py, px]
                occupied_id = occupied_ids[row, column, height]
                if occupied_id == -1:
                    occupied_ids[row, column, height] = max_id
                    grid_feat[max_id] = feat.flatten() * alpha
                    grid_rgb[max_id] = rgb_v
                    weight[max_id] += alpha
                    grid_pos[max_id] = [row, column, height]
                    max_id += 1
                    # print(max_id)
                else:
                    grid_feat[occupied_id] = (
                        grid_feat[occupied_id] * weight[occupied_id] + feat.flatten() * alpha
                    ) / (weight[occupied_id] + alpha)
                    if weight[occupied_id] + alpha == 0:
                        print("weight is 0")
                    grid_rgb[occupied_id] = (grid_rgb[occupied_id] * weight[occupied_id] + rgb_v * alpha) / (
                        weight[occupied_id] + alpha
                    )
                    weight[occupied_id] += alpha


        # update min_max
        self.min_max = min_max

        self.save_3d_map(grid_feat, grid_pos, weight, grid_rgb, occupied_ids, mapped_iter_set, max_id)


        

    def build_semantic_map(self,camera, world, labels):
        camera_poses= np.loadtxt(self.pose_path)
        pbar = tqdm(
            zip(self.rgb_paths,self.depth_paths, camera_poses, ),
            total=len(self.depth_paths),
            desc="Get Global Map",
        )
        # width, height = camera._resolution[0], camera._resolution[1]
        #! change to camera's true resolution
        width, height = 480, 640
        global_pcd = o3d.geometry.PointCloud()
        grid_2d = get_dummy_2d_grid(height, width) # same as camera_resolution:[640,480]
        # depths = []
        rgbs = []
        pcs = []
        pc_images = []
        lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std = self._init_lseg()

        for frame_i , (rgb_path, depth_path, camera_pose) in enumerate(pbar):
            world.step()
            camera.set_world_pose(camera_pose[:3], camera_pose[3:])
            depth = load_depth_npy(depth_path.as_posix())
            depth[depth>5] = 0
            depth = depth.flatten()
            bgr = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgbs.append(rgb)
            pc = camera.get_world_points_from_image_coords(grid_2d, depth)
            pc_downsampled = downsample_pc(pc, 150)
            pcd_global = o3d.geometry.PointCloud()
            pcd_global.points = o3d.utility.Vector3dVector(pc_downsampled)
            global_pcd += pcd_global
            visualize_pc(global_pcd,headless=False, save_path =f"{frame_i}.jpg")
            pcs.append(pc_downsampled)

            pc_image = camera.get_image_coords_from_world_points(pc_downsampled) # or can rather sample from scratch(depth image)
            pc_image = pc_image.astype(int)
            pc_images.append(pc_image)

        self.pcd_min = np.min(np.asarray(global_pcd.points), axis=0)
        self.pcd_max = np.max(np.asarray(global_pcd.points), axis=0)

        grid_feat, grid_pos, weight, occupied_ids, grid_rgb, mapped_iter_set, max_id = self._init_map(
            self.pcd_min, self.pcd_max, self.cs, self.map_save_path
        )

        cv_map = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)
        height_map = -100 * np.ones(self.grid_size[[0, 1]], dtype=np.float32)
        pbar = tqdm(zip(rgbs,pc_images,camera_poses, pcs), total=len(self.depth_paths), desc="Building Semantic Map")
        for frame_i, (rgb, pc_image,camera_pose,pc) in enumerate(pbar):
            pix_feats, pred = get_lseg_feat(
                lseg_model, rgb, labels, lseg_transform, self.device, crop_size, base_size, norm_mean, norm_std, vis = True, save_path = str(self.segmentation_dir) + f"/seg_{frame_i}.png"
            )

            radial_dist_sq = np.array([np.sum(np.square(pc_dd-camera_pose[:3])) for pc_dd in pc]).T
            sigma_sq = 0.6
            alphas = np.exp(-radial_dist_sq / (2 * sigma_sq))

            pbar = tqdm(zip(pc_image,pc,alphas), total=pc.shape[0], desc="Point wise mapping")
            for (p_i, p, alpha) in (pbar): # p: [x,y,z]
                py, px = p_i[0],p_i[1] # px actually means py
                row, column, height = np.round(((p - self.pcd_min) / self.cs)).astype(int)

                if max_id >= grid_feat.shape[0]:
                    grid_feat, grid_pos, weight, grid_rgb = self._reserve_map_space(
                        grid_feat, grid_pos, weight, grid_rgb
                    )

                if not (py < 0 or px < 0 or py >= pix_feats.shape[2] or px >= pix_feats.shape[3]):
                    rgb_v = rgb[py,px] # for given point p, its label and feature is pix_feats[px,py] and pix_mask[px,py] 
                    feat = pix_feats[0, :, py, px]
                    occupied_id = occupied_ids[row, column, height]
                    if occupied_id == -1:
                        occupied_ids[row, column, height] = max_id
                        grid_feat[max_id] = feat.flatten() * alpha
                        grid_rgb[max_id] = rgb_v
                        weight[max_id] += alpha
                        grid_pos[max_id] = [row, column, height]
                        max_id += 1
                        # print(max_id)
                    else:
                        grid_feat[occupied_id] = (
                            grid_feat[occupied_id] * weight[occupied_id] + feat.flatten() * alpha
                        ) / (weight[occupied_id] + alpha)
                        if weight[occupied_id] + alpha == 0:
                            print("weight is 0")
                        grid_rgb[occupied_id] = (grid_rgb[occupied_id] * weight[occupied_id] + rgb_v * alpha) / (
                            weight[occupied_id] + alpha
                        )
                        weight[occupied_id] += alpha

        self.save_3d_map(grid_feat, grid_pos, weight, grid_rgb, occupied_ids, mapped_iter_set, max_id)
        self.load_3d_map()

    def _reserve_map_space(
        self, grid_feat: np.ndarray, grid_pos: np.ndarray, weight: np.ndarray, grid_rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        grid_feat = np.concatenate(
            [
                grid_feat,
                np.zeros((grid_feat.shape[0], grid_feat.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        grid_pos = np.concatenate(
            [
                grid_pos,
                np.zeros((grid_pos.shape[0], grid_pos.shape[1]), dtype=np.int32),
            ],
            axis=0,
        )
        weight = np.concatenate([weight, np.zeros((weight.shape[0]), dtype=np.int32)], axis=0)
        grid_rgb = np.concatenate(
            [
                grid_rgb,
                np.zeros((grid_rgb.shape[0], grid_rgb.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        return grid_feat, grid_pos, weight, grid_rgb
    

    def save_3d_map(
        self,
        grid_feat: np.ndarray,
        grid_pos: np.ndarray,
        weight: np.ndarray,
        grid_rgb: np.ndarray,
        occupied_ids: Set,
        mapped_iter_set: Set,
        max_id: int,
    ) -> None:
        grid_feat = grid_feat[:max_id]
        grid_pos = grid_pos[:max_id]
        weight = weight[:max_id]
        grid_rgb = grid_rgb[:max_id]
        with h5py.File(self.map_save_path, "w") as f:
            f.create_dataset("mapped_iter_list", data=np.array(list(mapped_iter_set), dtype=np.int32))
            f.create_dataset("grid_feat", data=grid_feat)
            f.create_dataset("grid_pos", data=grid_pos)
            f.create_dataset("weight", data=weight)
            f.create_dataset("occupied_ids", data=occupied_ids)
            f.create_dataset("grid_rgb", data=grid_rgb)
            f.create_dataset("pcd_min", data=self.pcd_min)
            f.create_dataset("pcd_max", data=self.pcd_max)
            f.create_dataset("cs", data=self.map_config.cell_size)

    def init_categories(self, categories: List[str]) -> np.ndarray:
            self.categories = categories
            self.scores_mat = get_lseg_score(
                self.clip_model,
                self.categories,
                self.grid_feat,
                self.clip_feat_dim,
                use_multiple_templates=True,
                add_other=True,
            )  # score for name and other
            print("score_mat", self.scores_mat.shape)
            return self.scores_mat

    def index_map(self, language_desc: str, with_init_cat: bool = True):
        if with_init_cat and self.scores_mat is not None and self.categories is not None:
            cat_id = find_similar_category_id(language_desc, self.categories)
            scores_mat = self.scores_mat
        else:
            if with_init_cat:
                raise Exception(
                    "Categories are not preloaded. Call init_categories(categories: List[str]) to initialize categories."
                )
            scores_mat = get_lseg_score(
                self.clip_model,
                [language_desc],
                self.grid_feat,
                self.clip_feat_dim,
                use_multiple_templates=True,
                add_other=True,
            )  # score for name and other
        
        max_ids = np.argmax(scores_mat, axis=1)
        mask = max_ids == cat_id
        return mask
    def get_robot_bottom_z(self):
        '''get robot bottom z'''
        return self.env._runner.current_tasks[self.task_name].robots[self.robot_name].get_ankle_base_z()-self.sim_config.config_dict['tasks'][0]['robots'][0]['ankle_height']



@hydra.main(
    version_base=None,
    config_path="../config_my",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    my_world = World(stage_units_in_meters=1.0)
    camera = my_Camera(
        prim_path="/World/camera",
        resolution=(640, 480) # (640,480)
    )
    # data_dir = '/ssd/xiaxinyuan/code/w61-grutopia/logs/sample_episodes/s8pcmisQ38h/id_2606'
    data_dir = config.data_paths.vlmaps_data_dir
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    scene_ids = config.scene_id
    if (type(scene_ids)==int):
        scene_ids = [scene_ids]
    simulation_app = SimulationApp({'headless': True, 'anti_aliasing': 0, 'renderer': 'RayTracing'})
    for scene_id in scene_ids:
        scene_dir = Path(config.data_paths.vlmaps_data_dir)
        tmp = TMP(scene_dir,config.map_config)

        i = 0
        while simulation_app.is_running():
            i += 1
            my_world.step()
            tmp.init_map()
            tmp.build_semantic_map(camera, my_world, mp3dcat[1:-1])
            
            # tmp.load_3d_map()
            tmp.init_categories(mp3dcat[1:-1])
            for cat in mp3dcat[1:]:
                mask = tmp.index_map(cat, with_init_cat=True)
                t= np.sum(mask)
                print(t)

                if (t == 0):
                    continue
                mask_2d = pool_3d_label_to_2d(mask, tmp.grid_pos, config.params.gs)
                rgb_2d = pool_3d_rgb_to_2d(tmp.grid_rgb, tmp.grid_pos, config.params.gs)
                save_path = tmp.map_save_dir / f"{cat}_masked_map_2d.jpg"
                print(save_path)
                visualize_masked_map_2d(rgb_2d, mask_2d,save_path = save_path)

                heatmap = get_heatmap_from_mask_2d(mask_2d, cell_size=config.params.cs, decay_rate=config.decay_rate)
                save_path = tmp.map_save_dir / f"{cat}_heatmap_2d.jpg"
                visualize_heatmap_2d(rgb_2d, heatmap, save_path = save_path)

                save_path = tmp.map_save_dir/ f"{cat}_masked_map_3d.pcd"
                visualize_masked_map_3d(tmp.grid_pos, mask, tmp.grid_rgb, save_path = str(save_path))
                heatmap = get_heatmap_from_mask_3d(
                    tmp.grid_pos, mask, cell_size=config.params.cs, decay_rate=config.decay_rate
                )
                save_path = tmp.map_save_dir / f"{cat}_heatmap_3d.pcd"
                visualize_heatmap_3d(tmp.grid_pos, heatmap, tmp.grid_rgb,save_path = str(save_path))


if __name__ == "__main__":
    main()