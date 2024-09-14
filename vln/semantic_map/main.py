from vln.src.local_nav.sementic_map import BEVSemMap


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
print(sys.path)

from vlmaps.vlmaps.utils.lseg_utils import get_lseg_feat
from vlmaps.vlmaps.utils.mapping_utils import (
    load_3d_map,
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
simulation_app = SimulationApp({'headless': True, 'anti_aliasing': 0, 'renderer': 'RayTracing'})

from omni.isaac.core import World
from omni.isaac.sensor import Camera
import numpy as np
from omni.isaac.lab.app import AppLauncher



class my_Camera(Camera):
    def __init__(prim_path, resolution):
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
            [[221.70248,   0.     , 128.     ],
            [  0.     , 221.70248, 128.     ],
            [  0.     ,   0.     ,   1.     ]],dtype="float32", device=self._device)
        

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
            result = (mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, pcd_min, pcd_max, cs)
    except (OSError, KeyError):
        print("fail to load map, files does not exist")
        result = None
    # 返回结果
    return result

def load_depth_npy(depth_filepath: Union[Path, str]):
    with open(depth_filepath, "rb") as f:
        depth = np.load(f)
    return depth

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


class TMP:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.cs = 0.1 # TODO: to further code it
    
    def _setup_paths(self, data_dir: Union[Path, str]) -> None:
        self.data_dir = Path(data_dir)
        self.rgb_dir = self.data_dir / "rgb"
        self.depth_dir = self.data_dir / "depth"
        # self.semantic_dir = self.data_dir / "semantic"
        self.pose_path = self.data_dir / "poses.txt"
        try:
            self.rgb_paths = sorted(self.rgb_dir.glob("*.png"))
            self.depth_paths = sorted(self.depth_dir.glob("*.npy"))
            # self.semantic_paths = sorted(self.semantic_dir.glob("*.npy"))
        except FileNotFoundError as e:
            print(e)
    
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


    def build_semantic_map(self,camera, world):
        self._setup_paths(self.data_dir)
        self.map_save_dir = self.data_dir / "vlmap_cam"
        os.makedirs(self.map_save_dir, exist_ok=True)
        self.map_save_path = self.map_save_dir / "vlmaps_cam.h5df"

        camera_pose= np.loadtxt(self.pose_path)

        pbar = tqdm(
            zip(self.rgb_paths,self.depth_paths, camera_pose, ),
            total=len(self.depth_paths),
            desc="Get Global Map",
        )
        width,height = camera.get_resolution()
        global_pcd = o3d.geometry.PointCloud()
        grid_2d = get_dummy_2d_grid(width,height)
        depths = []
        rgbs = []
        pcs = []
        lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std = self._init_lseg()

        for frame_i , (rgb_path, depth_path, camera_pose) in enumerate(pbar):
            world.step()
            camera.set_pose(camera_pose)
            depth = load_depth_npy(depth_path.as_posix())
            depths.apped(depth)
            bgr = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgbs.append(rgb)
            pc = camera.get_world_points_from_image_coords(grid_2d, depth)
            pc_downsampled = downsample_pc(pc, 50)
            pcd_global = o3d.geometry.PointCloud()
            pcd_global.points = o3d.utility.Vector3dVector(pc_downsampled)
            global_pcd += pcd_global
            visualize_pc(pcd_global,headless=True, save_path ="1.jpg")

            pcs.append(pc_downsampled)

        self.pcd_min = np.min(np.asarray(global_pcd.points), axis=0)
        self.pcd_max = np.max(np.asarray(global_pcd.points), axis=0)

        grid_feat, grid_pos, weight, occupied_ids, grid_rgb, mapped_iter_set, max_id = self._init_map(
            self.pcd_min, self.pcd_max, self.cs, self.map_save_path
        )

        cv_map = np.zeros((self.grid_size[0], self.grid_size[2], 3), dtype=np.uint8)
        height_map = -100 * np.ones(self.grid_size[[0, 2]], dtype=np.float32)
        pbar = tqdm(zip(rgbs, depths, camera_pose,pcs), total=camera_pose.shape[0], desc="Building Semantic Map")
        for frame_i, (rgb, depth, camera_pose, pc) in enumerate(pbar):
            pix_feats,_ = get_lseg_feat(
                lseg_model, rgb, ["example"], lseg_transform, self.device, crop_size, base_size, norm_mean, norm_std
            )
            pc_image = camera.get_image_coords_from_world_points(pc)
            pc_image = pc_image.astype(int)

            radial_dist_sq = np.array([np.sum(np.square(pc_dd-camera_pose)) for pc_dd in pc]).T
            sigma_sq = 0.6
            alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))
            for p_i, p in zip(pc_image,pc): # p: [x,y,z]
                px,py = p_i[0],p_i[1]
                row, height, col = np.round(((p - self.pcd_min) / self.cs)).astype(int)
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
        self.save_3d_map(grid_feat, grid_pos, weight, grid_rgb, occupied_ids, mapped_iter_set, max_id)

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




@hydra.main(
    version_base=None,
    config_path="../vlnmaps/config_my",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    my_world = World(stage_units_in_meters=1.0)
    camera = my_Camera(
        prim_path="/World/camera",
        resolution=(640, 480)
    )
    data_dir = '/ssd/xiaxinyuan/code/w61-grutopia/logs/sample_episodes/s8pcmisQ38h/id_2606'
    tmp = TMP(data_dir)

    i = 0
    while simulation_app.is_running():
        i += 1
        my_world.step()
        if config.init_categories:
            tmp.init_categories(mp3dcat[1:-1])
            cat = mp3dcat[1:10]
            mask = tmp.index_map(cat, with_init_cat=True)
        else:
            mask = tmp.index_map(cat, with_init_cat=False)

        if config.index_2d:
            mask_2d = pool_3d_label_to_2d(mask, tmp.grid_pos, config.params.gs)
            rgb_2d = pool_3d_rgb_to_2d(tmp.grid_rgb, tmp.grid_pos, config.params.gs)
            visualize_masked_map_2d(rgb_2d, mask_2d)
            heatmap = get_heatmap_from_mask_2d(mask_2d, cell_size=config.params.cs, decay_rate=config.decay_rate)
            visualize_heatmap_2d(rgb_2d, heatmap)
        else:
            visualize_masked_map_3d(tmp.grid_pos, mask, tmp.grid_rgb)
            heatmap = get_heatmap_from_mask_3d(
                tmp.grid_pos, mask, cell_size=config.params.cs, decay_rate=config.decay_rate
            )
            visualize_heatmap_3d(tmp.grid_pos, heatmap, tmp.grid_rgb)

if __name__ == "__main__":
    main()