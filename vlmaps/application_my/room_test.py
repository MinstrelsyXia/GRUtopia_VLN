

# from vln.src.local_nav.sementic_map import BEVSemMap
# identical to semantic map/main.py
import os, sys ,re
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
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "vlmaps"))
print(sys.path)

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
from vlmaps.vlmaps.utils.index_utils import find_similar_category_id, get_segment_islands_pos, get_dynamic_obstacles_map_3d
from vlmaps.vlmaps.utils.clip_utils import get_img_feats, get_text_feats_multiple_templates
from vlmaps.application_my.utils import downsample_pc, visualize_pc, get_dummy_2d_grid, visualize_naive_occupancy_map
import clip



def load_3d_map(map_path: Union[Path, str]):
    try:
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
    except:
        print("A previous version lacking components exists. Error loading 3d map.")
        return None


from vlmaps.application_my.utils import NotFound
from vlmaps.vlmaps.map import VLMap
class TMP(VLMap):
    def __init__(self, map_config, data_dir="",test_file_save_dir="", robot_init_pose = (0,0,0)):
        
        self.robot_z = map_config.robot_z
        self.init_world_pos = np.array(robot_init_pose)
        self.cs = map_config.cell_size
        self.voxel_size = map_config.cell_size
        self.lseg_model = None
        self.min_max = np.zeros([3,2])
        self.map_config = map_config
        self.test_file_save_dir = test_file_save_dir
        
        if hasattr(map_config.model, 'threshold'):
            self.threshold = map_config.model.threshold
        else:
            self.threshold = 0.5
        if hasattr(map_config, 'pure_dynamic_map'):
            self.pure_dynamic_map = map_config.pure_dynamic_map
        else:
            self.pure_dynamic_map = False
        self.known_dict = {}
        super().__init__(map_config,data_dir)

    def _setup_paths(self, data_dir: Union[Path, str]) -> None:
        #! modified Map's function
        if self.pure_dynamic_map == False:
            self.data_dir = Path(data_dir)
            self.rgb_dir = self.data_dir / "rgb"
            self.depth_dir = self.data_dir / "depth"
            self.pose_path = self.data_dir / "poses.txt"

        self.segmentation_dir = '/ssd/xiaxinyuan/code/w61-grutopia/logs/sample_episodes/s8pcmisQ38h/id_37/segmentation'
        # self.segmentation_dir = self.test_file_save_dir + "/segmentation"
        # if not os.path.exists(self.segmentation_dir):
        #     os.makedirs(self.segmentation_dir)
        if self.pure_dynamic_map:
            print("not loading rgb paths")
            return
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
        base_size = 640  # 520: 无论多大的照片，长边都会变成640
        if torch.cuda.is_available():
            self.device = "cuda:0"
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

    def update_map(self, mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, pcd_min, pcd_max, cs):
        '''
        If need to save path: self.save_3d_map() & self.load_3d_map()
        If not needed, use self.update_map(...)
        '''
        self.mapped_iter_list = mapped_iter_list
        self.grid_feat = grid_feat
        self.grid_pos = grid_pos
        self.weight = weight
        self.occupied_ids = occupied_ids
        self.grid_rgb = grid_rgb
        self.pcd_min = pcd_min
        self.pcd_max = pcd_max
        self.cs = cs

    def init_map(self,data_dir:str,test_file_save_dir:str):
        '''
        data_dir: rgb, pose, depth
        file_save_dir: vlmaps.h5df
        '''
        # self._setup_paths(data_dir,test_file_save_dir)
        self._init_clip()
        if self.map_config.create_map == False:
            self.map_save_dir = Path(data_dir) /"vlmap"
            self.map_save_path = self.map_save_dir/ "vlmaps.h5df"
            print(self.map_save_path)
            if not self.map_save_path.exists():
                assert False, "Loading VLMap failed because the file doesn't exist."
            (
                self.mapped_iter_list,
                self.grid_feat,
                self.grid_pos,
                self.weight,
                self.occupied_ids,
                self.grid_rgb,
                self.min_max[:,0],
                self.min_max[:,1],
                self.cs,
            ) = load_3d_map(self.map_save_path)
            
        else:
            # used for building vlmap from scratch
            self.map_save_dir = Path(self.test_file_save_dir) / "vlmap_cam"
            if not os.path.exists(self.map_save_dir):
                os.makedirs(self.map_save_dir, exist_ok=True)
            self.map_save_path = self.map_save_dir / "vlmaps_cam.h5df"
            # if file of self.map_save_path exists, rewrite it
            if os.path.exists(self.map_save_path):
                os.remove(self.map_save_path)

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
            f.create_dataset("pcd_min", data=self.min_max[:,0])
            f.create_dataset("pcd_max", data=self.min_max[:,1])
            f.create_dataset("cs", data=self.map_config.cell_size)
        self.update_map(mapped_iter_list=np.array(list(mapped_iter_set), dtype=np.int32) ,grid_feat=grid_feat, grid_pos=grid_pos, weight=weight, occupied_ids=occupied_ids, grid_rgb=grid_rgb, pcd_min=self.min_max[:,0], pcd_max=self.min_max[:,1], cs=self.map_config.cell_size)
        
    def convert_world_to_map(self, point_cloud):
        # Note that the pointclouds have the world corrdinates that some values are very negative
        # We need to convert it into the map coordinates
        if len(point_cloud)==0:
            # log.error(f"The shape of point cloud is not correct. The shape is {point_cloud.shape}.")
            return None
        point_cloud[...,:2] = point_cloud[..., :2] - self.init_world_pos[:2]

        return point_cloud
    
    
    def _retrive_map(self, map_path: Path) -> Tuple:
            """
            retrive the saved map route
            """

            # check if there is already saved map
            if os.path.exists(map_path):
                (mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, pcd_min, pcd_max, cs) = (
                    self.load_3d_map()
                )
                mapped_iter_set = set(mapped_iter_list)
                max_id = grid_feat.shape[0]

            else:
                # init 
                cs = self.voxel_size
                pcd_max = self.min_max[:,1]
                pcd_min = self.min_max[:,0]
                grid_size = np.ceil((pcd_max - pcd_min) / cs + 1).astype(int)  # col, height, row
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

            return grid_feat, grid_pos, weight, occupied_ids, grid_rgb, mapped_iter_set, max_id
    
    def cvt_global_to_pixel_map(self,pc_global):
        '''
        pc_global: [N,3]
        '''
        x_max = max(self.min_max[0,1],np.max(pc_global[:,0]))
        x_min = min(self.min_max[0,0],np.min(pc_global[:,0]))
        y_max = max(self.min_max[1,1],np.max(pc_global[:,1]))
        y_min = min(self.min_max[1,0],np.min(pc_global[:,1]))
        z_max = max(self.min_max[2,1],np.max(pc_global[:,2]))
        z_min = min(self.min_max[2,0],np.min(pc_global[:,2]))

        new_row = np.ceil((x_max-x_min)/self.voxel_size+1).astype(int)
        new_height = np.ceil((y_max-y_min)/self.voxel_size+1).astype(int)
        new_col = np.ceil((z_max-z_min)/self.voxel_size+1).astype(int)

        delta_row = int((self.min_max[0,0]-x_min)/self.voxel_size)
        delta_height = int((self.min_max[1,0]-y_min)/self.voxel_size)
        delta_col = int((self.min_max[2,0]-z_min)/self.voxel_size)
        # map the local map to a global map

        self.min_max = np.array([[x_min,x_max],[y_min,y_max],[z_min,z_max]])
        print(self.min_max)
        return [new_row,new_height,new_col],[delta_row,delta_height,delta_col]
    
    def update_pos_map(self,old_map,new_map_coord,delta_map_coord):
        '''
        old_map: shape: [x,y,z]
        new_map: shape: [new_map_coord]
        '''
        new_map = -np.ones(np.array(new_map_coord),dtype=np.int32)
        new_map[delta_map_coord[0]:(old_map.shape[0]+delta_map_coord[0]),delta_map_coord[1]:(old_map.shape[1]+delta_map_coord[1]),delta_map_coord[2]:(old_map.shape[2]+delta_map_coord[2])] = old_map
        return new_map
    
    def _update_semantic_map(self,camera,rgb,depth,labels,step= 0 ):
        '''
        build semantic map locally, given sync camera
        
        '''
        max_depth = 10
        downsample_rate = 150
        grid_2d =  get_dummy_2d_grid(depth.shape[1],depth.shape[0])
        # depth
        # depth[depth > max_depth] = 0
        # rgb
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # pose
        camera_position, camera_orientation = camera.get_world_pose()

        grid_2d_ds = downsample_pc(grid_2d,downsample_rate)
        pc_image = grid_2d_ds
        depth_ds = depth[grid_2d_ds[:, 1], grid_2d_ds[:, 0]]
        downsampled_cloud = camera.get_world_points_from_image_coords(grid_2d_ds, depth_ds)
        downsampled_cloud = downsampled_cloud[np.isfinite(downsampled_cloud).all(axis=1)]
        # point_to_consider = downsampled_cloud = self.convert_world_to_map(downsampled_cloud)
        point_to_consider = downsampled_cloud
        # adjusted_coords = (downsampled_cloud[:, :2]/self.voxel_size + [self.quadtree_width/2, self.quadtree_height/2]).astype(int) 
        # adjusted_coords_with_z = np.hstack((adjusted_coords, downsampled_cloud[:,2].reshape(-1,1)))
        # point_to_consider = adjusted_coords_with_z[(adjusted_coords_with_z[:, 0] < self.quadtree_width) & (adjusted_coords_with_z[:, 1] < self.quadtree_height)] 

        # pc =  point_to_consider[(point_to_consider[:,2]>=(robot_bottom_z+self.robot_z[0])) & (point_to_consider[:,2]<=(robot_bottom_z+self.robot_z[1]))]# points that are within the robot's height range (occupancy),#! still decimal, not int!!!

        # pc = point_to_consider #! self.robot_z; robot_bottom_z should be considered

        distances = np.linalg.norm(point_to_consider - camera_position, axis=1)
        
        # 找出所有距离大于阈值的点
        threshold = 0.001
        mask = distances >= threshold
        
        # 使用布尔索引选择满足条件的点
        pc = point_to_consider[mask]

        pc_pcd = o3d.geometry.PointCloud()
        pc_pcd.points = o3d.utility.Vector3dVector(pc)
        # visualize_pc(pc_pcd,headless=False)
        if (pc.size == 0):
            print("pc is empty! Error")
            return
        if (self.lseg_model is None):
            self.lseg_model, self.lseg_transform, self.crop_size, self.base_size, self.norm_mean, self.norm_std = self._init_lseg()
            self.min_max = np.array([[np.min(pc[:,0]),np.max(pc[:,0])],[np.min(pc[:,1]),np.max(pc[:,1])],[np.min(pc[:,2]),np.max(pc[:,2])]])

        pix_feats ,pix_mask= get_lseg_feat(
                            self.lseg_model, rgb, labels, self.lseg_transform, self.device, self.crop_size, self.base_size, self.norm_mean, self.norm_std, vis = True, save_path = self.segmentation_dir + f"/{step}.jpg"
                        ) # [B,D,H,W];[H,W]
        # pix_feats = np.zeros([1,512,480,640])
        # pix_mask = np.zeros([480,640])

        new_map_coord, delta_map_coord = self.cvt_global_to_pixel_map(pc)
        grid_feat, grid_pos, weight, occupied_ids, grid_rgb, mapped_iter_set, max_id = self._retrive_map(self.map_save_path)
        
        # mapped_iter_set: literally empty
        # grid_feat, weight, grid_rgb are indexed by max_id, no need to update
        grid_pos = grid_pos + delta_map_coord
        occupied_ids = self.update_pos_map(occupied_ids,new_map_coord,delta_map_coord)




        #  pc_image: [py,px]:pixel coord in 2d-image; 
        #  row,height column: pixel coord in 3d-map
        radial_dist_sq = np.array([np.sum(np.square(pc_dd-camera_position)) for pc_dd in pc]).T
        sigma_sq = 0.6
        alphas = np.exp(-radial_dist_sq / (2 * sigma_sq))

        pbar = tqdm(zip(pc_image,pc,alphas), total=pc.shape[0], desc="Point wise mapping")
        for (p_i, p, alpha) in (pbar): # p: [x,y,z]
            py, px = p_i[0],p_i[1] # px actually means py
            row, column, height = np.round(((p - self.min_max[:,0]) / self.cs)).astype(int) # in the new coord

            if max_id >= grid_feat.shape[0]:
                grid_feat, grid_pos, weight, grid_rgb = self._reserve_map_space(
                    grid_feat, grid_pos, weight, grid_rgb
                )

            if not (py < 0 or px < 0 or py >= pix_feats.shape[2] or px >= pix_feats.shape[3]):
                rgb_v = rgb[py,px] # for given point p, its label and feature is pix_feats[px,py] and pix_mask[px,py] 
                feat = pix_feats[0, :, py, px]
                occupied_id = occupied_ids[row, column, height].astype(int)
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
        
        # max depth among the indexes in pc_image
        mask = np.zeros_like(depth)
        mask[pc_image[:, 1], pc_image[:, 0]] = 1
        max_depth = np.max(depth[mask==1])
        visualize_naive_occupancy_map(occupied_ids, save_path = "occupancy.jpg")
        return pc,max_depth


    def test_visualze(self):
        visualize_naive_occupancy_map(self.occupied_ids, save_path = "occupancy.jpg")

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
    
    def get_min_pcd(self):
        return self.min_max[:,0]
    
    def from_map_to_xyz(self, row, column):
        '''
        input: row, column
        output: x, y
        '''
        x = row * self.cs + self.pcd_min[0]
        y = column * self.cs + self.pcd_min[1]
        return x, y

    def from_xyz_to_map(self, x, y):
        row = int((x - self.pcd_min[0]) / self.cs)
        column = int((y - self.pcd_min[1]) / self.cs)
        return row, column
    
    def get_score_mat_clip(self, bgr,categories):
        # load image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_feats = get_img_feats(rgb, self.preprocess, self.clip_model) # [1,512]
        text_feats = get_text_feats_multiple_templates(categories, self.clip_model, self.clip_feat_dim)

        img_feats = img_feats / np.linalg.norm(img_feats, axis=-1, keepdims=True)
        text_feats = text_feats / np.linalg.norm(text_feats, axis=-1, keepdims=True)

        # 计算相似度（需要转换为tensor进行softmax）
        similarity = img_feats @ text_feats.T
        return similarity[0]

    def get_score_mat_lseg(self, bgr,categories,save_path):
        '''
        modified in 12/12
        Input:
            bgr: image
            categories: list of categories
            save_path: path to save the segmentation result
        Output:
            similarity: similarity score for each category
        '''
        # bgr = cv2.imread(img_path)
        device = self.device
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        outputs, _ = get_lseg_feat(
            self.lseg_model, 
            rgb,  # 需要提供RGB图像
            categories,  # 需要提供标签列表
            self.lseg_transform, 
            device, 
            crop_size=self.crop_size, 
            base_size=self.base_size, 
            norm_mean=self.norm_mean, 
            norm_std=self.norm_std, 
            vis=False, 
            save_path=save_path
        )
        B, D, H, W = outputs.shape
        text_feats = get_text_feats_multiple_templates(categories, self.clip_model, self.clip_feat_dim)
        # 对H,W维度进行平均池化，得到[B,D]
        if isinstance(outputs, np.ndarray):
            outputs = torch.from_numpy(outputs)
        if isinstance(text_feats, np.ndarray):
            text_feats = torch.from_numpy(text_feats)
        
        # 移动到指定设备
        outputs = outputs.to(device)
        text_feats = text_feats.to(device)
        
        # 对特征图进行平均池化
        img_feats = outputs.mean(dim=(-2,-1))  # [B,D]
        
        # 归一化
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        # similarity = (100.0 * img_feats @ text_feats.T).softmax(dim=-1)
        similarity = img_feats @ text_feats.T
        
        return similarity

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

    def index_map(self, language_desc: str, with_init_cat: bool = True, verbose: bool = True) -> np.ndarray:
        # init map
        # if with_init_cat and self.scores_mat is not None and self.categories is not None:
        #     if language_desc in self.known_dict.keys():
        #         cat_id = self.known_dict[language_desc]
        #         scores_mat = self.scores_mat
        #     else:
        #         cat_id = find_similar_category_id(language_desc, self.categories)
        #         self.known_dict[language_desc] = cat_id
        #         scores_mat = self.scores_mat
        # else:
        #     if with_init_cat:
        #         raise Exception(
        #             "Categories are not preloaded. Call init_categories(categories: List[str]) to initialize categories."
        #         )
            
        #     scores_mat = get_lseg_score(
        #         self.clip_model,
        #         [language_desc],
        #         self.grid_feat,
        #         self.clip_feat_dim,
        #         use_multiple_templates=True,
        #         add_other=True,
        #     )  # score for name and other
        scores_mat = get_lseg_score(
                self.clip_model,
                [language_desc],
                self.grid_feat,
                self.clip_feat_dim,
                use_multiple_templates=True,
                add_other=True,
            )  # score for name and other
        # if language_desc in self.known_dict.keys():
        #     cat_id = self.known_dict[language_desc]
        # else:
        #     cat_id = find_similar_category_id(language_desc, self.categories)
        #     self.known_dict[language_desc] = cat_id
        # # score_mat: [h*w,c]
        max_ids = np.argmax(scores_mat, axis=1)
        
        # mask = max_ids == cat_id
        mask = max_ids == 0 #['landmark', 'others']
        print(np.sum(mask))
        for i in range(len(max_ids)):
            if scores_mat[i, max_ids[i]] <= self.threshold:
                mask[i]  = 0  # 将不符合条件的值设为 -1
        if np.sum(mask)>50 and verbose:
            mask_2d = pool_3d_label_to_2d(mask, self.grid_pos, self.gs)
            rgb_2d = pool_3d_rgb_to_2d(self.grid_rgb, self.grid_pos, self.gs)
            save_path = os.path.join(os.path.dirname(self.map_save_path), f"{language_desc.replace(' ', '_')}.jpg")
            visualize_masked_map_2d(rgb_2d, mask_2d,save_path=save_path)
        return mask
    
    def judge_room(self, obs):
        room_names = ['bedroom', 'dining room', 'kitchen', 'living room', 'office', 'hallway', 'unknown']
        similarity = []
        for room_name in room_names:
            similarity.append(self.get_score_mat_clip(obs, [room_name]))
        print(similarity)
        return room_names[np.argmax(similarity)]


    
    ############################## from VLMap(Map) ########################################
    
    def get_pos(self, name: str) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], Any]:
        """
        Get the contours, centers, and bbox list of a certain category
        on a full map
        """
        assert self.categories
        # cat_id = find_similar_category_id(name, self.categories)
        # labeled_map_cropped = self.scores_mat.copy()  # (N, C) N: number of voxels, C: number of categories
        # labeled_map_cropped = np.argmax(labeled_map_cropped, axis=1)  # (N,)
        # pc_mask = labeled_map_cropped == cat_id # (N,)
        # self.grid_pos[pc_mask]
        pc_mask = self.index_map(name, with_init_cat=True)
        if pc_mask is None or np.sum(pc_mask) < 50:
            raise NotFound(f"pc_mask for object '{name}' is either empty or too small.")

        mask_2d = pool_3d_label_to_2d(pc_mask, self.grid_pos, self.gs)
        # mask_2d = mask_2d[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]
        
        # print(f"showing mask for object cat {name}")
        # cv2.imshow(f"mask_{name}", (mask_2d.astype(np.float32) * 255).astype(np.uint8))
        # cv2.waitKey(1)

        foreground = binary_closing(mask_2d, iterations=3)
        foreground = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3)
        foreground = foreground > 0.5
        # cv2.imshow(f"mask_{name}_gaussian", (foreground * 255).astype(np.uint8))
        foreground = binary_dilation(foreground)
        # cv2.imshow(f"mask_{name}_processed", (foreground.astype(np.float32) * 255).astype(np.uint8))
        # cv2.waitKey(1)

        contours, centers, bbox_list, _ = get_segment_islands_pos(foreground, 1)
        # print("centers", centers)

        # whole map position
        # for i in range(len(contours)):
        #     centers[i][0] += self.rmin
        #     centers[i][1] += self.cmin
        #     bbox_list[i][0] += self.rmin
        #     bbox_list[i][1] += self.rmin
        #     bbox_list[i][2] += self.cmin
        #     bbox_list[i][3] += self.cmin
        #     for j in range(len(contours[i])):
        #         contours[i][j, 0] += self.rmin
        #         contours[i][j, 1] += self.cmin
        if len(contours) == 0 or len(centers) == 0 or len(bbox_list) == 0:
            raise NotFound(f"contours, centers, bbox_list for object '{name}' is empty.")
        return contours, centers, bbox_list

    def check_object(self, name: str) -> bool:
        """
        Check if an object exists in the map
        """
        pc_mask = self.index_map(name[0], with_init_cat=True)
        if pc_mask is None or np.sum(pc_mask) < 10:
            return False
        return True

    def get_room_pos(self, room_name: str) -> np.ndarray:
        pass