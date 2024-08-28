import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import defaultdict
import math

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
        self.camera_height = self.args.maps.global_topdown_config.camera_transform_height
        self.width = self.args.maps.global_topdown_config.width
        self.height = self.args.maps.global_topdown_config.height
        
        self.floor_maps = defaultdict(lambda: {})
        self.floor_heights = [] # this height is built based on robot's base (not camera base)

        self.agent_radius = args.maps.agent_radius  # The radius(m) of robot
        self.voxel_size = args.maps.global_topdown_config.voxel_size # TODO
        if self.args.maps.add_dilation: # TODO
            self.dilation_structure = self.create_dilation_structure(self.agent_radius)

        # Attributes for path_planner
        self.planner_config = args.planners
        self.path_planner = AStarPlanner(args=self.args, 
                                         map_width=self.width,map_height=self.height,max_step=self.planner_config.a_star_max_iter,
                            windows_head=self.args.windows_head,
                            for_llm=self.args.settings.use_llm,
                            verbose=True)

        # init vis settings
        self.cmap = mcolors.ListedColormap(['white', 'green', 'gray', 'black'])  # Colors for 0, between 1-254, 2, 255
        self.bounds = [0, 1, 3, 254, 256]  # Boundaries for the colors
        self.norm = mcolors.BoundaryNorm(self.bounds, self.cmap.N)
    
    def save_map(self, robot_pos=None, is_camera_base=False):
        height = self.get_height(robot_pos, is_camera_base=is_camera_base)
        occupancy_map = self.get_map(robot_pos, is_camera_base=is_camera_base)
        if occupancy_map is None:
            return None

        # Define the color map
        cmap = mcolors.ListedColormap(['white', '#C1FFC1', 'gray', 'black'])  # Colors for 0, between 1-254, 2, 255
        bounds = [0, 1, 3, 254, 256]  # Boundaries for the colors
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(10, 10))
        # plt.xlim(0, occupancy_map.shape[1])
        # plt.ylim(0, occupancy_map.shape[0])
        plt.imshow(occupancy_map, cmap=cmap, norm=norm)
        if robot_pos is not None:
            robot_pixel = self.world_to_pixel(robot_pos, is_camera_base=is_camera_base)
            plt.scatter(robot_pixel[0], robot_pixel[1], color='blue', marker='o', label="current position (%.2f, %.2f, %.2f)" % (robot_pos[0], robot_pos[1], robot_pos[2]))
        plt.legend()
        
        img_save_path = os.path.join(self.args.log_image_dir, f"global_topdown_map_{self.scan_name}_{height}.jpg")
        plt.savefig(img_save_path, pad_inches=0, bbox_inches='tight', dpi=100)

        log.info(f"Saved global topdown map at height {height} to {img_save_path}")
    
    def get_height(self, pos, is_camera_base=False):
        # if is_camera_base:
        #     return math.floor(pos[2] - self.camera_height)
        # return math.floor(pos[2])
        if is_camera_base:
            return round(pos[2] - self.camera_height)
        return round(pos[2])
    
    def update_map(self, freemap, camera_pose, update_map=False, verbose=False):
        height = self.get_height(camera_pose, is_camera_base=True)
        if height not in self.floor_maps:
            self.floor_heights.append(height)
            self.floor_maps[height] = {
                'camera_pose': camera_pose,
                'freemap': freemap,
                'occupancy_map': self.freemap_to_accupancy_map(freemap, add_dilation=self.args.maps.add_dilation)
            } 
            log.info("update global topdown map at height: {}".format(height))
            if verbose:
                self.save_map(robot_pos=camera_pose, is_camera_base=True)
        else:
            if update_map:
                self.floor_maps[height] = {
                    'camera_pose': camera_pose,
                    'freemap': freemap,
                    'occupancy_map': self.freemap_to_accupancy_map(freemap, add_dilation=self.args.maps.add_dilation)
                }

    def get_map(self, world_pose, is_camera_base=False, return_camera_pose=False):
        height = self.get_height(world_pose, is_camera_base=is_camera_base)
        if height in self.floor_maps.keys():
            if return_camera_pose:
                return self.floor_maps[height]['occupancy_map'], self.floor_maps[height]['camera_pose']
            return self.floor_maps[height]['occupancy_map']
        else:
            log.error("Floor height not found in global topdown map")
            return None

    def world_to_pixel_old(self, world_pose, specific_height=None, is_camera_base=False):
        if specific_height is None:
            height = self.get_height(world_pose, is_camera_base=is_camera_base)

            if height not in self.floor_maps.keys() and specific_height is None:
                log.error("Floor height not found in global topdown map")
                return None
        else:
            height = specific_height
        
        camera_pose = self.floor_maps[height]['camera_pose']

        cx, cy = camera_pose[1]*10/self.camera_aperture*self.width, -camera_pose[0]*10/self.camera_aperture*self.height
        X, Y = world_pose[1]*10/self.camera_aperture*self.width, -world_pose[0]*10/self.camera_aperture*self.height

        pixel_x = X - cx + self.width/2
        pixel_y = self.height - (Y - cy + self.height/2)

        return [pixel_x, pixel_y]
    
    def world_to_pixel(self, world_pose, specific_height=None, is_camera_base=False):
        if specific_height is None:
            height = self.get_height(world_pose, is_camera_base=is_camera_base)

            if height not in self.floor_maps.keys() and specific_height is None:
                log.error("Floor height not found in global topdown map")
                return None
        else:
            height = specific_height
        
        camera_pose = self.floor_maps[height]['camera_pose']

        cx, cy = camera_pose[0]*10/self.camera_aperture*self.width, -camera_pose[1]*10/self.camera_aperture*self.height
        X, Y = world_pose[0]*10/self.camera_aperture*self.width, -world_pose[1]*10/self.camera_aperture*self.height

        pixel_x = self.width - (X - cx + self.width/2)
        pixel_y = Y - cy + self.height/2

        return [pixel_x, pixel_y]

    def pixel_to_world_old(self, pixel, camera_pose):
        cx, cy = camera_pose[1]*10/self.camera_aperture*self.width, -camera_pose[0]*10/self.camera_aperture*self.height

        px = pixel[0] + cx - self.height/2
        py = pixel[1] + cy - self.width/2

        world_x = -py/10/self.height*self.camera_aperture
        world_y = px/10/self.width*self.camera_aperture
        
        return [world_x, world_y]

    def pixel_to_world(self, pixel, camera_pose):
        cx, cy = camera_pose[0]*10/self.camera_aperture*self.width, -camera_pose[1]*10/self.camera_aperture*self.height

        # px = self.height - (pixel[0] + cx - self.height/2)
        px = self.height - pixel[0] + cx - self.height/2
        py = pixel[1] + cy - self.width/2

        world_x = px/10/self.height*self.camera_aperture
        world_y = -py/10/self.width*self.camera_aperture
        
        return [world_x, world_y]
    
    
    def clear_map(self):
        self.floor_maps = defaultdict(lambda: {})
        self.floor_heights = []
    
    def freemap_to_accupancy_map(self, freemap, add_dilation=False):
        occupancy_map = np.zeros((self.width, self.height))
        occupancy_map[freemap == 1] = 2
        occupancy_map[freemap == 0] = 255
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

    def sample_points_between_two_points(self, start, end, step=1):
        start = np.array(start)
        end = np.array(end)

        delta = (end-start)/step
        sampled_points = [start + i*delta for i in range(step+1)]

        return sampled_points

    def navigate_p2p(self, start, goal, step_time=0, verbose=False, all_paths=[], save_dir=None):
        # start_height = int(start[2])
        # goal_height = int(goal[2])
        if save_dir is None:
            save_dir = self.args.log_image_dir

        if abs(start[2] - goal[2]) >= 0.3:
            # different floor! we directly sample the nodes
            # return None
            transfer_paths = self.sample_points_between_two_points(start, goal, step=self.args.planners.stair_sample_step) # !!!
            if verbose:
                occupancy_map = self.get_map(start)
                map_nodes = [self.world_to_pixel(x, specific_height=self.get_height(start)) for x in transfer_paths]
                self.path_planner.vis_path(occupancy_map, map_nodes[0][0], map_nodes[0][1], map_nodes[-1][0], map_nodes[-1][1], map_nodes, os.path.join(save_dir, "global_path_"+str(step_time)+".jpg"), legend=True)
        
        else:
            occupancy_map, camera_pose = self.get_map(start, return_camera_pose=True)
            
            start_pixel = self.world_to_pixel(start)
            goal_pixel = self.world_to_pixel(goal)

            # test
            if verbose and len(all_paths) > 0:
                plt.clf()
                plt.imshow(occupancy_map, cmap=self.cmap, norm=self.norm, origin='upper')
                path_pixel_list = []
                for path in all_paths:
                    path_pixel = self.world_to_pixel(path)
                    path_pixel_list.append(path_pixel)
                    plt.scatter(path_pixel[1], path_pixel[0])
                all_paths_save_path = os.path.join(save_dir, "all_paths.jpg")
                plt.savefig(all_paths_save_path)
                log.info(f"Saved all paths visualization to {all_paths_save_path}")
                plt.clf()

            # self.path_planner.update_obs_map(occupancy_map)

            paths, find_flag = self.path_planner.planning(start_pixel[0], start_pixel[1],
                                            goal_pixel[0], goal_pixel[1],
                                            obs_map=occupancy_map,
                                            min_final_meter=self.planner_config.last_scope,
                                            vis_path=False)
            if verbose:
                if len(paths) > 0:
                    self.vis_nav_path(start_pixel, goal_pixel, paths, occupancy_map, img_save_path=os.path.join(save_dir, "global_path_"+str(step_time)+".jpg"))

            if find_flag:
                transfer_paths = []
                for node in paths:
                    world_coords = self.pixel_to_world([node[0],node[1]], camera_pose)
                    transfer_paths.append([world_coords[0], world_coords[1], start[2]])
            else:
                transfer_paths = None

        # remove the first point
        if transfer_paths is not None and len(transfer_paths)>1:
            transfer_paths.pop(0)

        return transfer_paths

    def vis_nav_path(self, start_pixel, goal_pixel, points, occupancy_map, img_save_path='path_planning.jpg'):
        plt.figure(figsize=(10, 10))
        # plt.imshow(occupancy_map, cmap='binary', origin='lower')
        plt.imshow(occupancy_map, cmap=self.cmap, norm=self.norm, origin='upper')

        # Plot start and goal points
        plt.plot(start_pixel[1], start_pixel[0], 'ro', markersize=6, label='Start')
        plt.plot(goal_pixel[1], goal_pixel[0], 'go', markersize=6, label='Goal')

        # Plot the path
        path = np.array(points)
        plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=2, label='Path')

        # Customize the plot
        plt.title('Path planning')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid()
        plt.colorbar(label='Occupancy (0: Free, 1: Occupied)')

        # Save the plot
        plt.savefig(img_save_path, pad_inches=0, bbox_inches='tight', dpi=100)
        log.info(f"Saved path planning visualization to {img_save_path}")