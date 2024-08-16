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
from grutopia.core.util.log import log

class CamOccupancyMap:
    def __init__(self, args, robot_prim, start_point, local=True):
        self.args = args
        self.width = 512
        self.height = 512

        self.center_x, self.center_y = self.width // 2, self.height// 2

        # Create a camera object
        # self.topdown_camera_prim_path = "/World/env_0/CamOccCamera"
        # self.create_new_orthogonal_camera()
        
        # load from a pre-defined top-down camera
        if local:
            self.topdown_camera_prim_path = robot_prim + "/topdown_camera_50"
            self.aperture = 50
        else:
            self.topdown_camera_prim_path = robot_prim + "/topdown_camera_500"
            self.aperture = self.args.maps.global_topdown_config.aperture
            self.width = self.args.maps.global_topdown_config.width
            self.height = self.args.maps.global_topdown_config.height
            self.center_x, self.center_y = self.width // 2, self.height// 2
        self.topdown_camera = Camera(prim_path=self.topdown_camera_prim_path,resolution=(self.width, self.height))
        self.topdown_camera.initialize()
        self.topdown_camera.add_distance_to_image_plane_to_frame()
        self.topdown_camera.add_normals_to_frame()
        
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

    def world_to_pixel(self, world_coords, camera_pose):
        world_x, world_y = world_coords[0], world_coords[1]
        
        # Convert aperture to radians
        fov_rad = np.deg2rad(self.aperture)
        
        # Calculate the scale factor
        scale_x = 2 * np.tan(fov_rad / 2) / self.width
        scale_y = 2 * np.tan(fov_rad / 2) / self.height
        
        # Calculate offsets from camera pose
        offset_x = world_x - camera_pose[0]
        offset_y = world_y - camera_pose[1]
        
        # Convert world offsets to pixel offsets
        pixel_offset_x = offset_x / scale_x
        pixel_offset_y = offset_y / scale_y
        
        # Calculate pixel coordinates
        pixel_x = int(pixel_offset_x + self.width / 2)
        pixel_y = int(pixel_offset_y + self.height / 2)
        
        return pixel_x, pixel_y
    
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
    
    def open_windows_head(self, text_info=None):
        # plt.ion()  # Turn on interactive mode
        # self.window_fig = plt.figure(1)  # Create and store a specific figure
        self.window_fig, self.ax = plt.subplots(figsize=(5, 5))  # Create a figure and a subplot
        self.ax = self.window_fig.add_subplot(111)  # Add a subplot to the figure
        self.image_display = self.ax.imshow(np.zeros((10, 10, 3)), aspect='auto')  # Placeholder for the image
        if text_info is not None:
            self.ax.text(4, 11, text_info, fontsize=10, ha='center', va='bottom', wrap=True)
        self.ax.set_title('Top-down RGB Image')

    def update_windows_head(self, robot_pos, text_info=None, mode="show"):
        rgb_data = self.get_camera_data(["rgba"])["rgba"]
        self.topdown_camera.set_world_pose([robot_pos[0], robot_pos[1], robot_pos[2] + 0.8])
        self.image_display.set_data(rgb_data)  # Update the image data
        if text_info is not None:
            self.ax.text(0.5, 0.01, text_info, fontsize=10, ha='left', va='bottom', wrap=True)
        # self.ax.draw_artist(self.ax.patch)  # Efficiently redraw the background
        self.ax.draw_artist(self.image_display)  # Efficiently redraw the image
        if mode == 'show':
            plt.show(block=False)
            plt.pause(0.001)  # This is necessary to update the window
        elif mode == 'save':
            img_save_path = self.args.log_image_dir + "/window_topdown_image.png"
            self.window_fig.savefig(img_save_path, bbox_inches='tight')

    def close_windows_head(self):
        plt.close('all')  # Close all figures

    def get_surrounding_free_map(self, robot_pos, robot_height=1.05+0.8, verbose=False):
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
        try:
            normalized_normals = normals[..., :3] / (norm_magnitudes[..., np.newaxis]+1e-8)
            # Generate mask for flat surfaces based on normal vectors
            flat_surface_mask = np.abs(normalized_normals[..., 2] - 1) < normal_threshold
        except Exception:
            # NOTE: Sometimes the normal vectors are not available? 
            # the issue is: RuntimeWarning: invalid value encountered in divide
            flat_surface_mask = np.zeros_like(depth, dtype=bool)

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
            img_path = os.path.join(self.args.log_image_dir, "cam_free","rgb_scene_local.png")
            img.save(img_path)
            # print("Image saved at", img_path)

            depth_img = self.vis_depth(depth, robot_height)
            depth_img_path = os.path.join(self.args.log_image_dir, "cam_free", "depth_scene_local.png")
            depth_img.save(depth_img_path)
            # print("Depth saved at", depth_img_path)
        
            # Visualize the free map
            free_map_normalized = free_map.astype(bool)
            # free_map_normalized = ((free_map - free_map.min()) * (1/(free_map.max() - free_map.min()) * 255)).astype('uint8')
            free_map_image = Image.fromarray(free_map_normalized)
            # Save the image
            free_map_path = os.path.join(self.args.log_image_dir, "cam_free", "topdown_local_freemap.png")
            free_map_image.save(free_map_path)
            # print("Free map saved at", free_map_path)
        
        # extract connectd free area
        connected_free_area = self.extract_connected_free_area(free_map, verbose=verbose)

        # update the pose of the camera based on robot's pose
        self.topdown_camera.set_world_pose([robot_pos[0], robot_pos[1], robot_pos[2]+0.8])

        return free_map, connected_free_area
    
    def analysize_depth(self, depth_array):
        mask = ~np.isinf(depth_array)

        valid_depths = depth_array[mask]

        min_depth = np.min(valid_depths)
        max_depth = np.max(valid_depths)

        mean_depth = np.mean(valid_depths)
        std_depth = np.std(valid_depths)

        print(f"depth range: {min_depth} - {max_depth}")
        print(f"depth mean: {mean_depth}")
        print(f"depth std: {std_depth}")
    
    def set_world_pose(self, robot_pos):
        # update the pose of the camera based on robot's pose
        self.topdown_camera.set_world_pose([robot_pos[0], robot_pos[1], robot_pos[2]+0.8])

    def get_global_free_map(self, robot_pos, robot_height=1.05+0.8, norm_filter=False, connect_filter=False, verbose=False):
        # Define height range for free map
        # free_map: 1 for free space, 0 for occupied space

        min_height = robot_height
        # max_height = robot_height + 0.8
        max_height = 4
        normal_threshold = 0.005

        # rgb_init, depth_init, mask, (row_min, row_max), (col_min, col_max), pointcloud = self._get_topdown_map(self.topdown_camera)

        # get info
        data_info = self.get_camera_data(["rgba", "depth", "normals"])
        rgb = np.array(data_info["rgba"])
        depth = np.array(data_info["depth"])
        normals = np.array(data_info["normals"])

        # Generate free map using normal vectors and depth information

        # Normalize the normal vectors (ignoring the last component if it exists)
        if norm_filter:
            norm_magnitudes = np.linalg.norm(normals[..., :3], axis=2)
            try:
                normalized_normals = normals[..., :3] / (norm_magnitudes[..., np.newaxis]+1e-8)
                # Generate mask for flat surfaces based on normal vectors
                flat_surface_mask = np.abs(normalized_normals[..., 2] - 1) < normal_threshold
            except Exception:
                # NOTE: Sometimes the normal vectors are not available? 
                # the issue is: RuntimeWarning: invalid value encountered in divide
                flat_surface_mask = np.ones_like(depth, dtype=bool)
        else:
            flat_surface_mask = np.ones_like(depth, dtype=bool)

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
            img_path = os.path.join(self.args.log_image_dir, "cam_free","rgb_scene_global.png")
            img.save(img_path)
            # print("Image saved at", img_path)

            depth_img = self.vis_depth(depth, robot_height)
            depth_img_path = os.path.join(self.args.log_image_dir, "cam_free", "depth_scene_global.png")
            depth_img.save(depth_img_path)
            # print("Depth saved at", depth_img_path)
        
            # Visualize the free map
            free_map_normalized = free_map.astype(bool)
            # free_map_normalized = ((free_map - free_map.min()) * (1/(free_map.max() - free_map.min()) * 255)).astype('uint8')
            free_map_image = Image.fromarray(free_map_normalized)
            # Save the image
            free_map_path = os.path.join(self.args.log_image_dir, "cam_free", "topdown_global_freemap.png")
            free_map_image.save(free_map_path)
            log.info(f"Global free map saved at {free_map_path}.")
        
        # extract connectd free area
        if connect_filter:
            connected_free_area = self.extract_connected_free_area(free_map, verbose=verbose)
        else:
            connected_free_area = None

        # update the pose of the camera based on robot's pose
        self.topdown_camera.set_world_pose([robot_pos[0], robot_pos[1], robot_pos[2]+0.8])

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
            connected_free_area_image = Image.fromarray(connected_free_area)
            connected_free_area_path = os.path.join(self.args.log_image_dir, "cam_free", "connected_free_area.png")
            connected_free_area_image.save(connected_free_area_path)
            print("Connected free area saved at", connected_free_area_path)

        return connected_free_area