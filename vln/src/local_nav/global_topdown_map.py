import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import defaultdict

from PIL import Image
from copy import copy
from scipy.ndimage import binary_fill_holes, label, generate_binary_structure, binary_dilation
import matplotlib.colors as mcolors

from .BEVmap import BEVMap
from grutopia.core.util.log import log
from vln.src.local_nav.camera_occupancy_map import CamOccupancyMap
from vln.src.local_nav.path_planner import QuadTreeNode, Node, RRTstarPathPlanning, AStarPlanner

class GlobalTopdownMap:
    def __init__(self, args, scan_name):
        self.args = args
        self.scan_name = scan_name

        self.camera_aperture = self.args.maps.global_topdown_config.aperture
        self.width = self.args.maps.global_topdown_config.width
        self.height = self.args.maps.global_topdown_config.height
        
        self.floor_maps = defaultdict(lambda: {})
        self.floor_heights = []

        self.agent_radius = args.maps.agent_radius  # The radius(m) of robot
        if self.args.maps.add_dilation: # TODO
            self.dilation_structure = self.create_dilation_structure(self.agent_radius)

        # Attributes for path_planner
        self.planner_config = args.planners
        self.path_planner = AStarPlanner(args=self.args, 
                                         map_width=self.width,map_height=self.height,max_step=self.planner_config.a_star_max_iter,
                            windows_head=self.args.windows_head,
                            for_llm=self.args.settings.use_llm,
                            verbose=True)
    
    def save_map(self, robot_pos=None):
        height = int(robot_pos[2])
        occupancy_map = self.get_map(robot_pos)
        if occupancy_map is None:
            return None

        # Define the color map
        cmap = mcolors.ListedColormap(['white', 'green', 'gray', 'black'])  # Colors for 0, between 1-254, 2, 255
        bounds = [0, 1, 3, 254, 256]  # Boundaries for the colors
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(10, 10))
        plt.xlim(0, occupancy_map.shape[1])
        plt.ylim(0, occupancy_map.shape[0])
        plt.imshow(occupancy_map, cmap=cmap, norm=norm)
        if robot_pos is not None:
            robot_pixel = self.world_to_pixel(robot_pos)
            plt.plot(robot_pixel[0], robot_pixel[1], 'blue', markersize=10, label="current position (%.2f, %.2f)" % (robot_pos[0], robot_pos[1]))
        plt.legend()
        
        img_save_path = os.path.join(self.args.log_image_dir, f"global_topdown_map_{self.scan_name}_{height}.jpg")
        plt.savefig(img_save_path, pad_inches=0, bbox_inches='tight', dpi=100)

        log.info(f"Saved global topdown map at height {height} to {img_save_path}")
    
    def update_map(self, freemap, camera_pose, update_map=False, verbose=False):
        height = int(camera_pose[2])
        if height not in self.floor_map:
            self.floor_heights.append(height)
            self.floor_maps[height] = {
                'camera_pose': camera_pose,
                'freemap': freemap,
                'occupancy_map': self.freemap_to_accupancy_map(freemap, add_dilation=self.args.maps.add_dilation)
            } 
            log.info("update global topdown map at height: {}".format(height))
            if verbose:
                self.save_map(robot_pos=camera_pose)
        else:
            if update_map:
                self.floor_maps[height] = {
                    'camera_pose': camera_pose,
                    'freemap': freemap,
                    'occupancy_map': self.freemap_to_accupancy_map(freemap, add_dilation=self.args.maps.add_dilation)
                }

    def get_map(self, world_pose):
        height = int(world_pose[2])
        if height in self.floor_maps.keys():
            return self.floor_maps[height]['occupancy_map']
        else:
            log.error("Floor height not found in global topdown map")
            return None

    def pixel_to_world(self, pixel, camera_pose):
        # Convert aperture to radians
        pixel_x, pixel_y = pixel
        fov_rad = np.deg2rad(self.camera_aperture)
        
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

    def world_to_pixel(self, world_coords):
        height = int(world_coords[2])

        if height not in self.floor_maps.keys():
            log.error("Floor height not found in global topdown map")
            return None
        else:
            camera_pose = self.floor_maps[height]['camera_pose']

            world_x, world_y = world_coords[0], world_coords[1]
            
            # Convert aperture to radians
            fov_rad = np.deg2rad(self.camera_aperture)
            
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
            
            return pixel_x, pixel_y, world_coords[2]
    
    def clear_map(self):
        self.floor_maps = defaultdict(lambda: {})
        self.floor_heights = []
    
    def freemap_to_accupancy_map(self, freemap, add_dilation=False):
        occupancy_map = np.zeros((self.width, self.height))
        for i in range(self.width):
            for j in range(self.height):
                if freemap[i, j] == 1:
                    occupancy_map[i, j] = 2
                elif freemap[i, j] == 0:
                    occupancy_map[i, j] = 255
        if add_dilation:
            for i in range(1, self.args.maps.dilation_iterations):
                ob_mask = np.logical_and(occupancy_map!=0, occupancy_map!=2)
                expanded_ob_mask = binary_dilation(ob_mask, structure=self.dilation_structure, iterations=1)
                occupancy_map[expanded_ob_mask&(np.logical_or(occupancy_map==0,occupancy_map==2))] = 255 - i*10
        return occupancy_map

    def create_dilation_structure(self, radius):
        """
        Creates a dilation structure based on the robot's radius.
        """
        radius_cells = int(np.ceil(radius / self.voxel_size))
        # Create a structuring element for dilation (a disk of the robot's radius)
        dilation_structure = np.zeros((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
        cy, cx = radius_cells, radius_cells
        for y in range(2 * radius_cells + 1):
            for x in range(2 * radius_cells + 1):
                if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius_cells:
                    dilation_structure[y, x] = True
        return dilation_structure

    def navigate_p2p(self, start, goal, step_time=0):
        start_height = int(start[2])
        goal_height = int(goal[2])

        if start_height != goal_height:
            # different floor
            return None

        occupancy_map = self.get_map(start)
        
        start_pixel = self.world_to_pixel(start)
        goal_pixel = self.world_to_pixel(goal)

        path = self.path_planner.planning(start_pixel[1], start_pixel[0],
                                          goal_pixel[1], goal_pixel[0],
                                          occupancy_map,
                                          min_final_meter=self.planner_config.last_scope,
                                          img_save_path=os.path.join(self.args.log_image_dir, "global_path_"+str(step_time)+".jpg"))

        return path