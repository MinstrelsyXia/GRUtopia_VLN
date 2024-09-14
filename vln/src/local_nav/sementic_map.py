import torch
import numpy as np
import torch.nn as nn
import cv2


import sys, os
sys.path.append((os.path.dirname(os.path.abspath(__file__))))
sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from path_planner import QuadTreeNode
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from grutopia.core.util.log import log
import torch.nn.functional as F
from matplotlib.lines import Line2D

from pointcloud import create_pointcloud_from_depth, compute_intrinsic_matrix,downsample_pc

import open3d as o3d
import os,sys

sys.path.append("/ssd/xiaxinyuan/code/w61-grutopia/vlmaps")
print(sys.path)
from vlmaps.vlmaps.utils.lseg_utils import get_lseg_feat

from vlmaps.vlmaps.utils.clip_utils import get_lseg_score

from vlmaps.vlmaps.lseg.modules.models.lseg_net import LSegEncNet

from vlmaps.vlmaps.utils.index_utils import find_similar_category_id

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Set

from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
import numpy as np
from omegaconf import DictConfig
import torch
import gdown
import open3d as o3d
import h5py


class BEVSemMap:
    def __init__(self,args,robot_init_pose):
        # super().__init__(args,robot_init_pose)
        self.args = args

        self.step_time = 0
        quadtree_config = args.maps.quadtree_config
        self.voxel_size = args.maps.voxel_size  # Resolution to present the map
        quadtree_config.width, quadtree_config.height = int(quadtree_config.width/self.voxel_size), int(quadtree_config.height/self.voxel_size)
        self.quadtree_config = quadtree_config
        self.quadtree_width = self.quadtree_config.width
        self.quadtree_height = self.quadtree_config.height
        print(f"quadtree width for semantic map: {self.quadtree_width}, quadtree height: {self.quadtree_height}")

        self.semantic_map = np.zeros((self.quadtree_height, self.quadtree_width, args.segmentation.num_categories))

        self.init_world_pos = np.array(robot_init_pose)
        self.robot_z = args.maps.robot_z  # The height(m) range of robot
        self.data_dir = "" # !!!!
        self.map_save_dir = self.data_dir / "vlmap_cam"
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
    

    def update_semantic_map(self,obs_tr,cameras,camera_dict:dict,robot_bottom_z,mode='static'):
        '''
        mode: static: read in all the data and update the map
        mode: dynamic: read in data framewise and update the map
        '''
        # single robot
        grid_feat, grid_pos, weight, occupied_ids, grid_rgb, mapped_iter_set, max_id = self.load_3d_map(self.map_save_path)
        for camera_name in camera_dict:
            # get: rgb, depth, camera_params
            cur_obs = obs_tr[camera_name]
            rgb_obs = cur_obs['rgba'][...,:3]
            depth_obs = cur_obs['depth']
            max_depth = 10
            depth_obs[depth_obs > max_depth] = 0
            # camera_params = cur_obs['camera_params']

            camera = cameras[camera_name]
            camera_pose = camera.get_world_pose()
            camera_position, camera_orientation = camera_pose[0], camera_pose[1]

            # create image coord description
            height, width = depth_obs.shape
            x = np.arange(width)
            y = np.arange(height)
            xx, yy = np.meshgrid(x, y)
            xx_flat = xx.flatten()
            yy_flat = yy.flatten()
            points_2d = np.vstack((xx_flat, yy_flat)).T  

            # create point cloud
            
            pc=camera.get_world_points_from_image_coords(points_2d, depth_obs)
            downsampled_cloud = downsample_pc(pc)

            downsampled_cloud = downsampled_cloud[np.isfinite(downsampled_cloud).all(axis=1)]
            # further obtain points within the robot's height range
            downsampled_cloud = self.convert_world_to_map(downsampled_cloud)
            adjusted_coords = (downsampled_cloud[:, :2]/self.voxel_size + [self.quadtree_width/2, self.quadtree_height/2]).astype(int) 
            adjusted_coords_with_z = np.hstack((adjusted_coords, downsampled_cloud[:,2].reshape(-1,1)))
            point_to_consider = adjusted_coords_with_z[(adjusted_coords_with_z[:, 0] < self.quadtree_width) & (adjusted_coords_with_z[:, 1] < self.quadtree_height)] 

            pc_global = point_within_robot_z = point_to_consider[(point_to_consider[:,2]>=(robot_bottom_z+self.robot_z[0])) & (point_to_consider[:,2]<=(robot_bottom_z+self.robot_z[1]))].astype(int) # points that are within the robot's height range (occupancy)

            # create semantic description
            rgb = cv2.cvtColor(rgb_obs, cv2.COLOR_BGR2RGB)
            lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std = self._init_lseg()

            pix_feats ,pix_mask= get_lseg_feat(
                            lseg_model, rgb, ["example"], lseg_transform, self.device, crop_size, base_size, norm_mean, norm_std
                        ) # [B,D,H,W];[H,W]
            pc_image = camera.get_image_coords_from_world_points(point_within_robot_z) # [N,2]:[[x,y],...]
            pc_image = pc_image.astype(int)

            # transform from global coord to semantic map:


            radial_dist_sq = np.array([np.sum(np.square(pc_dd-camera_pose)) for pc_dd in point_within_robot_z]).T
            sigma_sq = 0.6
            alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))

            
            pc_pixel_map, new_map_coord, delta_map_coord, min_max = self.cvt_global_to_pixel_map(pc_global)
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
            for (p_i,p_m) in zip(pc_image,pc_pixel_map): # p: [x,y,z]
                px,py = p_i[0],p_i[1]
                row,height,col = p_m[0],p_m[1],p_m[2]
                rgb_v = rgb[py,px] # for given point p, its label and feature is pix_feats[px,py] and pix_mask[px,py] 

                if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                    feat = pix_feats[0, :, py, px]
                    occupied_id = occupied_ids[row, height, col]
                    if occupied_id == -1:
                        occupied_ids[row, height, col] = max_id
                        grid_feat[max_id] = feat.flatten() * alpha
                        grid_rgb[max_id] = rgb_v
                        weight[max_id] += alpha
                        grid_pos[max_id] = [row, height, col]
                        max_id += 1
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
        
        
    def plot_segmentation_result(self, rgb, semantic_segmentation, save_path):
        '''
        Visualize and save the RGB image and semantic segmentation map side by side.
        Args:
            rgb: The RGB image array of shape [H, W, 3].
            semantic_segmentation: The semantic segmentation map array of shape [H, W].
            save_path: Path to save the resulting figure.
        '''
        # Create a figure and axis
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    def plot_segmentation_result(self, rgb, semantic_segmentation, save_path):
        '''
        Visualize and save the RGB image and semantic segmentation map side by side.
        Args:
            rgb: The RGB image array of shape [H, W, 3].
            semantic_segmentation: The semantic segmentation map array of shape [H, W].
            save_path: Path to save the resulting figure.
        '''
        # Create a figure and axis
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    def _reserve_map_space(
    self, grid_feat: np.ndarray, grid_pos: np.ndarray, weight: np.ndarray, grid_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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



        # Ensure semantic_segmentation is integer type
        if semantic_segmentation.dtype != np.uint8:
            semantic_segmentation = semantic_segmentation.astype(np.uint8)
        
        # Ensure semantic_segmentation is integer type
        if semantic_segmentation.dtype != np.uint8:
            semantic_segmentation = semantic_segmentation.astype(np.uint8)
        
        # Add a legend to the semantic segmentation plot
        # Only add legend entries for non-empty classes
        num_classes = np.max(semantic_segmentation) + 1
        valid_classes = np.unique(semantic_segmentation)

        # Add a legend to the semantic segmentation plot
        # Only add legend entries for non-empty classes
        num_classes = np.max(semantic_segmentation) + 1
        valid_classes = np.unique(semantic_segmentation)

    # TODO: check the hard-coded size and path
    def _init_lseg(self):
        '''
        copied from vlmap: vlmap_builder_cam.py
        '''
        crop_size = 480  # 480
        base_size = 256  # 520
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        lseg_model = LSegEncNet("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
        model_state_dict = lseg_model.state_dict()
        checkpoint_dir = Path(__file__).resolve().parents[1] / "lseg" / "checkpoints"
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

        
    @staticmethod
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
            return mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, pcd_min, pcd_max, cs
        
    # copied from vlmap.map
    def generate_obstacle_map(self, h_min: float = 0, h_max: float = 1.5) -> np.ndarray:
        """Generate topdown obstacle map from loaded 3D map

        Args:
            h_min (float, optional): The minimum height (m) of voxels considered
                as obstacles. Defaults to 0.
            h_max (float, optional): The maximum height (m) of voxels considered
                as obstacles. Defaults to 1.5.
        Return:
            obstacles_map (np.ndarray): (gs, gs) 1 is free, 0 is occupied
        """
        assert self.occupied_ids is not None, "map not loaded"
        heights = np.arange(0, self.occupied_ids.shape[-1]) * self.cs
        height_mask = np.logical_and(heights > h_min, heights < h_max)
        self.obstacles_map = np.sum(self.occupied_ids[..., height_mask] > 0, axis=2) == 0
        self.generate_cropped_obstacle_map(self.obstacles_map)
        return self.obstacles_map

    def generate_cropped_obstacle_map(self, obstacle_map: np.ndarray) -> np.ndarray:
        x_indices, y_indices = np.where(obstacle_map == 0)
        self.rmin = np.min(x_indices)
        self.rmax = np.max(x_indices)
        self.cmin = np.min(y_indices)
        self.cmax = np.max(y_indices)
        self.obstacles_cropped = obstacle_map[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]
        return self.obstacles_cropped
    

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
            cat_id = 0

        max_ids = np.argmax(scores_mat, axis=1)
        mask = max_ids == cat_id
        return mask