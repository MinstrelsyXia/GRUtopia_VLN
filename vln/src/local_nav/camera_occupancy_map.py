import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

import omni
from pxr import UsdGeom, Usd, Sdf, Gf, UsdPhysics, UsdUtils
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.rotations import euler_angles_to_quat

class CamOccupancyMap:
    def __init__(self, args):
        self.args = args
        self.width=1440
        self.height=1440

        # Create a camera object
        self.topdown_camera_prim_path = "/World/env_0/CamOccCamera"
        self.create_new_orthogonal_camera()

        self.camera_occupancy_file = os.path.join(self.args.log_image_dir, "camera_occupancy_map.npy")

        # get current scene bounding box
        self.scene_min_bounds, self.scene_max_bounds = self.get_scene_bbox()
        self.scene_center = (self.scene_min_bounds + self.scene_max_bounds) / 2

        # set the topdown_camera's coordinate system to the scene's coordinate system
        self.topdown_camera.set_world_pose(self.scene_center)

        topdown_camera_pose = self.topdown_camera.get_world_pose()
        print("Topdown camera pose:", topdown_camera_pose)
    
    def create_new_orthogonal_camera(self):
        ''' Create a new orthogonal camera for top-down map
        '''
        self.topdown_camera = Camera(
            prim_path=self.topdown_camera_prim_path,
            frequency=20,
            resolution=(self.width, self.height),
            orientation=(euler_angles_to_quat([0,180,0]))
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
    
    def _get_topdown_map(self, camera):
        rgb = camera.get_rgb()
        rgb = np.array(rgb)
        
        depth = camera.get_depth()
        depth = np.array(depth)

        pointcloud = camera.get_pointcloud()
        
        mask = depth < 15
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

    def generate_occupancy_map(self, agent_position, agent_bottom_z, verbose=False):
        camera_pose = (self.scene_center[0], self.scene_center[1], agent_bottom_z+1.7)
        # camera_pose = (current_position[0], current_position[1], current_position[2]+1)
        self.topdown_camera.set_world_pose(camera_pose)
        print("Update the topdown camera pose:", camera_pose)
        rgb_init, depth_init, mask, (row_min, row_max), (col_min, col_max), pointcloud = self._get_topdown_map(self.topdown_camera)
        
        if verbose:
            img = Image.fromarray(rgb_init)
            img_path = os.path.join(self.args.log_image_dir, "cam_occ","rgb_before.png")
            img.save(img_path)
            print("Image saved at", img_path)

            depth_img = self.vis_depth(depth_init,3)
            depth_img_path = os.path.join(self.args.log_image_dir, "cam_occ", "depth_before.png")
            depth_img.save(depth_img_path)
            print("Depth saved at", depth_img_path)
        
        # Define height range for occupancy map
        max_height = 1.7
        min_height = 0

        # Generate occupancy map
        occupancy_map = np.zeros_like(depth_init, dtype=int)
        occupancy_map[np.logical_and(depth_init > min_height, depth_init < max_height)] = 1
        occupancy_map[depth_init >= max_height] = 0
        occupancy_map[depth_init <= min_height] = 0

        if verbose:
            # Visualize the occupancy map
            plt.clf()
            plt.imshow(occupancy_map, cmap='gray')
            plt.title('Occupancy Map')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            # plt.show()
            occupancy_map_path = os.path.join(self.args.log_image_dir, "cam_occ", "cam_global_occupancy_map.png")
            plt.savefig(occupancy_map_path)
            print("Occupancy map saved at", occupancy_map_path)
        
        return occupancy_map

    def get_free_map(self, topdown_camera, min_point, max_point):
        ''' Get the free map of the scene
        '''
        rgb, depth, mask, (row_min, row_max), (col_min, col_max) = self._get_topdown_map(topdown_camera)
        free_map = mask[row_min:row_max, col_min:col_max]
        return free_map
        