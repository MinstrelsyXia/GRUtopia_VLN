import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from copy import copy
from scipy.ndimage import binary_fill_holes, label, generate_binary_structure

import omni
from pxr import UsdGeom, Usd, Sdf, Gf, UsdPhysics, UsdUtils
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import omni.replicator.core as rep

from .BEVmap import BEVMap

class CamOccupancyMap:
    def __init__(self, args, robot_prim, start_point):
        self.args = args
        self.width = 512
        self.height = 512

        self.center_x, self.center_y = self.width // 2, self.height// 2

        # Create a camera object
        # self.topdown_camera_prim_path = "/World/env_0/CamOccCamera"
        # self.create_new_orthogonal_camera()
        
        # load from a pre-defined top-down camera
        self.topdown_camera_prim_path = robot_prim + "/topdown_camera"
        self.topdown_camera = Camera(prim_path=self.topdown_camera_prim_path,resolution=(self.width, self.height))
        self.topdown_camera.initialize()
        self.topdown_camera.add_distance_to_image_plane_to_frame()
        self.topdown_camera.add_normals_to_frame()
        self.aperture = 50

        self.rp = rep.create.render_product(self.topdown_camera_prim_path, (self.width, self.height))
        # rgba
        self.rgba_receiver = rep.AnnotatorRegistry.get_annotator("LdrColor")
        self.rgba_receiver.attach(self.rp)
        
        # depth
        self.depth_reveiver = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.depth_reveiver.attach(self.rp)

        # normals
        self.normals_receiver = rep.AnnotatorRegistry.get_annotator("normals")
        self.normals_receiver.attach(self.rp)

        # camera_params
        self.camera_params_receiver = rep.AnnotatorRegistry.get_annotator("CameraParams")
        self.camera_params_receiver.attach(self.rp)

        self.camera_occupancy_file = os.path.join(self.args.log_image_dir, "camera_occupancy_map.npy")

        # get current scene bounding box
        self.scene_min_bounds, self.scene_max_bounds = self.get_scene_bbox()
        self.scene_center = (self.scene_min_bounds + self.scene_max_bounds) / 2

        # set the topdown_camera's coordinate system to the scene's coordinate system
        # self.topdown_camera.set_world_pose(start_point+[0,0,0.65])
        # self.topdown_camera_init_robot_pose = False
        print("Topdown camera pose init:", self.topdown_camera.get_world_pose())

    def get_camera_data(self, data_type: list): 
        output_data = {}
        if "rgba" in data_type:
            output_data["rgba"] = self.rgba_receiver.get_data()
        if "depth" in data_type:
            output_data["depth"] = self.depth_reveiver.get_data()
        if "normals" in data_type:
            output_data["normals"] = self.normals_receiver.get_data()
        if "camera_params" in data_type:
            output_data["camera_params"] = self.camera_params_receiver.get_data()
        return output_data
    
    def set_topdown_camera_pose(self, pose):
        self.topdown_camera.set_world_pose(pose)
        self.topdown_camera_init_robot_pose = True
        print("Update the topdown camera pose:", pose)
    
    def create_new_orthogonal_camera(self):
        ''' Create a new orthogonal camera for top-down map
        '''
        self.topdown_camera = Camera(
            prim_path=self.topdown_camera_prim_path,
            frequency=20,
            resolution=(self.width, self.height),
            orientation=(euler_angles_to_quat([0,0,180]))
        )
        
        self.topdown_camera.initialize()
        self.topdown_camera.add_distance_to_image_plane_to_frame()
        self.topdown_camera.add_pointcloud_to_frame()

        self.topdown_camera.set_projection_mode("orthographic")
        self.topdown_camera.set_horizontal_aperture(50.0)
        self.topdown_camera.set_vertical_aperture(50.0)
        # self.topdown_camera.set_clipping_range(0.5, 100000000)
        self.topdown_camera.set_clipping_range(0, 1000)
        print("Topdown camera created")

    
    def get_scene_bbox(self, scene_prim_path="/World/env_0/scene"):
        # Get the bounding box information of the model
        stage = omni.usd.get_context().get_stage()
        model_prim = stage.GetPrimAtPath(scene_prim_path) # NOTE: how to achieve the current scene when there are different envs
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(model_prim)

        # Get the minimum and maximum values of the bounding box
        min_point = bbox.GetRange().GetMin()
        max_point = bbox.GetRange().GetMax()
        return min_point, max_point
    
    def _get_topdown_map(self, camera, height=1.7):
        rgb = camera.get_rgb()
        rgb = np.array(rgb)
        
        depth = camera.get_depth()
        depth = np.array(depth)

        pointcloud = camera.get_pointcloud()
        # pointcloud = pointcloud[~np.isnan(pointcloud[:,0])]  # Filter out NaN values
        
        mask = depth < height
        row = mask.astype(np.int32).sum(1)
        col = mask.astype(np.int32).sum(0)

        row_min, row_max = self.find_non_zero_range(row)
        col_min, col_max = self.find_non_zero_range(col)

        return rgb, depth, mask, (row_min, row_max), (col_min, col_max), pointcloud

        # pass
    
    def find_non_zero_range(self, array):
        length = len(array)
        l = 0
        r = length - 1
        while l < length:
            if array[l] != 0:
                break
            l += 1
        while r >= 0:
            if array[r] != 0:
                break
            r -= 1
        
        return l, r

    def vis_depth(self, depth, farest=20):
        depth = np.array(depth)
        depth = (np.clip(depth, 0, farest) / farest * 255.).astype(np.uint8)
        img = Image.fromarray(depth)
        return img
    
    def set_visibility(self, prim, visible=False):
        # Note that this is useless unless use the env.step() to update the visibility
        if not Usd.Object.IsValid(prim):
            print(prim, "is not a valid prim")
            return 
        vis = prim.GetAttribute("visibility")
        if visible:
            vis.Set("inherited")
        else:
            vis.Set("invisible")

    def pixel_to_world(self, pixel, camera_pose):
        # Convert aperture to radians
        pixel_x, pixel_y = pixel
        fov_rad = np.deg2rad(self.aperture)
        
        # Calculate the scale factor
        scale_x = 2 * np.tan(fov_rad / 2) / self.width
        scale_y = 2 * np.tan(fov_rad / 2) / self.height
        
        # Calculate the offsets from the center of the image
        offset_x = (pixel_x - self.width / 2) * scale_x
        offset_y = (pixel_y - self.height / 2) * scale_y
        
        # Compute the world coordinates
        world_x = camera_pose[0] + offset_x
        world_y = camera_pose[1] + offset_y
        
        return world_x, world_y
    
    def create_robot_mask(self, mask_size=20):
        # Calculate the top-left and bottom-right coordinates
        half_size = mask_size // 2
        top_left_x = self.center_x - half_size
        top_left_y = self.center_y - half_size
        bottom_right_x = self.center_x + half_size
        bottom_right_y = self.center_y + half_size

        # Create the mask
        robot_mask = np.zeros((self.width, self.height), dtype=np.uint8)
        robot_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1
        return robot_mask

    def get_surrounding_free_map(self, robot_prim, robot_height=1.05+0.8, verbose=False):
        # Define height range for free map
        # free_map: 1 for free space, 0 for occupied space

        min_height = robot_height
        max_height = robot_height + 0.8
        normal_threshold = 0.005

        # rgb_init, depth_init, mask, (row_min, row_max), (col_min, col_max), pointcloud = self._get_topdown_map(self.topdown_camera)

        # get info
        data_info = self.get_camera_data(["rgba", "depth", "normals"])
        rgb = np.array(data_info["rgba"])
        depth = np.array(data_info["depth"])
        normals = np.array(data_info["normals"])

        # Generate free map using normal vectors and depth information

        # Normalize the normal vectors (ignoring the last component if it exists)
        norm_magnitudes = np.linalg.norm(normals[..., :3], axis=2)
        normalized_normals = normals[..., :3] / norm_magnitudes[..., np.newaxis]

        # Generate mask for flat surfaces based on normal vectors
        flat_surface_mask = np.abs(normalized_normals[..., 2] - 1) < normal_threshold

        # Generate mask for depth within the acceptable range
        depth_mask = (depth >= min_height) & (depth < max_height)

        # robot_mask
        robot_mask = self.create_robot_mask()

        # Combine masks to determine free space
        free_map = np.zeros_like(depth, dtype=int)
        free_map[flat_surface_mask & depth_mask] = 1  # Free space is where conditions are met
        free_map[robot_mask == 1] = 1  # Robot's location is free space

        if verbose:
            img = Image.fromarray(rgb)
            if os.path.exists(self.args.log_image_dir+"/cam_free") == False:
                os.makedirs(self.args.log_image_dir+"/cam_free")
            img_path = os.path.join(self.args.log_image_dir, "cam_free","rgb_scene_center.png")
            img.save(img_path)
            print("Image saved at", img_path)

            depth_img = self.vis_depth(depth, robot_height)
            depth_img_path = os.path.join(self.args.log_image_dir, "cam_free", "depth_scene_center.png")
            depth_img.save(depth_img_path)
            print("Depth saved at", depth_img_path)
        
            # Visualize the free map
            plt.clf()
            plt.imshow(free_map, cmap='gray')
            plt.title('Free Map')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            # plt.show()
            free_map_path = os.path.join(self.args.log_image_dir, "cam_free", "cam_global_free_map.png")
            plt.savefig(free_map_path)
            print("Free map saved at", free_map_path)
        
        # extract connectd free area
        connected_free_area = self.extract_connected_free_area(free_map, verbose=verbose)
        # print("Update the topdown camera pose:", self.topdown_camera.get_world_pose())
        return free_map, connected_free_area
        
    def extract_connected_free_area(self, free_map, verbose=False):
        """
        Extract the free area directly connected to the robot's location (center of the map).

        Args:
            free_map (np.array): Binary free map indicating walkable areas (1) and non-walkable areas (0).

        Returns:
            np.array: Binary map of connected free areas.
        """
        # Determine the center of the map
        center_x, center_y = free_map.shape[1] // 2, free_map.shape[0] // 2

        # Initialize a mask for the connected component
        connected_free_area = np.zeros_like(free_map)

        # Initialize a mask for the connected component
        labeled_map, num_features = label(free_map, structure=generate_binary_structure(2, 2))
        
        # Get the label of the region connected to the center
        center_label = labeled_map[center_y, center_x]
        
        # Create a mask for the connected area
        connected_free_area = (labeled_map == center_label)
        if verbose:
            # Visualize the connected free area
            plt.clf()
            plt.imshow(connected_free_area, cmap='gray')
            plt.title('Connected Free Area')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            # plt.show()
            connected_free_area_path = os.path.join(self.args.log_image_dir, "cam_free", "connected_free_area.png")
            plt.savefig(connected_free_area_path)
            print("Connected free area saved at", connected_free_area_path)

        return connected_free_area