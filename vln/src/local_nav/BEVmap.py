import os
import numpy as np
import pandas as pd
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.ndimage import binary_dilation

from grutopia.core.util.log import log

from .path_planner import QuadTreeNode, Node, PathPlanning

class BEVMap:
    def __init__(self, args, robot_init_pose=(0, 0, 0)):
        self.args = args
        
        self.step_time = 0
        # Attributes for occupancy_map
        quadtree_config = args.maps.quadtree_config
        self.voxel_size = args.maps.voxel_size  # Resolution to present the map
        quadtree_config.width, quadtree_config.height = int(quadtree_config.width/self.voxel_size), int(quadtree_config.height/self.voxel_size)
        self.quadtree_config = quadtree_config
        self.quadtree_width = self.quadtree_config.width
        self.quadtree_height = self.quadtree_config.height
        self.occupancy_map = np.ones((self.quadtree_height, self.quadtree_width))  # 2D occupancy map
        self.quad_tree_root = QuadTreeNode(x=0, y=0, width=self.quadtree_width, height=self.quadtree_height,
                                           map_data = self.occupancy_map, 
                                           max_depth=self.quadtree_config.max_depth, threshold=self.quadtree_config.threshold)  # quadtrees
        
        # let the agent's initial point to be the origin of the map
        self.init_world_pos = np.array(robot_init_pose)
        self.robot_z = args.maps.robot_z  # The height(m) range of robot
        self.robot_radius = args.maps.robot_radius  # The radius(m) of robot
        self.dilation_structure = self.create_dilation_structure(self.robot_radius)

        # Attributes for path_planner
        self.planner_config = args.planners
        
        log.info("BEVMap initialized")
    
    def reset(self):
        self.occupancy_map = np.ones((self.quadtree_height, self.quadtree_width))
        self.quad_tree_root = QuadTreeNode(0, 0, map_data = self.occupancy_map, **self.quadtree_config)
        self.step_time = 0
        self.world_min_x, self.world_min_y = 0, 0
    
    def convert_world_to_map(self, point_cloud):
        # Note that the pointclouds have the world corrdinates that some values are very negative
        # We need to convert it into the map coordinates
        if len(point_cloud)==0:
            log.error(f"The shape of point cloud is not correct. The shape is {point_cloud.shape}.")
            return None
        point_cloud = point_cloud - self.init_world_pos

        return point_cloud
    
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
        
    ######################## update_occupancy_map ########################
    def update_occupancy_map(self, point_cloud, robot_bottom_z, add_dilation=False, verbose = False):
        """
        Updates the occupancy map based on the new point cloud data.
        Args:
            point_cloud: The new point cloud data.
            robot_bottom_z: The z value of the robot's bottom
            add_dilation: Whether to add dilation based on robot's radius to avoid the collision
        """
        # Store the new point cloud after downsampling
        if point_cloud is not None:
            if isinstance(point_cloud, list):
                pos_point_cloud = [self.convert_world_to_map(p) for p in point_cloud]
                pos_point_cloud = [p for p in pos_point_cloud if p is not None]
                pos_point_cloud = np.vstack(pos_point_cloud)
            pos_point_cloud = pd.DataFrame(pos_point_cloud)
            if not pos_point_cloud.isna().all().all():
                pos_point_cloud = pos_point_cloud.dropna().to_numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pos_point_cloud)
                downsampled_cloud = np.asarray(pcd.voxel_down_sample(voxel_size=self.voxel_size).points)

                adjusted_coords = (downsampled_cloud[:, :2]/self.voxel_size + [self.quadtree_width/2, self.quadtree_height/2]).astype(int) 
                adjusted_coords_with_z = np.hstack((adjusted_coords, downsampled_cloud[:,2].reshape(-1,1)))
                point_to_consider = adjusted_coords_with_z[(adjusted_coords_with_z[:, 0] < self.quadtree_width) & (adjusted_coords_with_z[:, 1] < self.quadtree_height)] # !!! there seems that 0 and 1 are reversed

                point_within_robot_z = point_to_consider[(point_to_consider[:,2]>(robot_bottom_z+self.robot_z[0])) & (point_to_consider[:,2]<(robot_bottom_z+self.robot_z[1]))].astype(int)

                unique_data_0 = np.unique(point_within_robot_z[:, :2], axis=0)
                unique_data_all = np.unique(point_to_consider[:, :2], axis=0).astype(int)
                unique_data_1 = np.array(list(set(map(tuple, unique_data_all)) - set(map(tuple, unique_data_0)))).astype(int)

                last_map = 1 - (self.occupancy_map == 0)
                if unique_data_1.size > 0:
                    self.occupancy_map[unique_data_1[:,1], unique_data_1[:,0]]=2 # !!!
                if unique_data_0.size > 0: 
                    self.occupancy_map[unique_data_0[:,1],unique_data_0[:,0]]=0
                    if add_dilation:
                        # Create a mask for the free positions and dilate it
                        free_mask = np.zeros_like(self.occupancy_map, dtype=bool)
                        free_mask[unique_data_0[:, 1], unique_data_0[:, 0]] = True
                        expanded_free_mask = binary_dilation(free_mask, structure=self.dilation_structure)
                        
                        # Set the expanded free positions to 0
                        self.occupancy_map[expanded_free_mask] = 0

                self.occupancy_map = self.occupancy_map*last_map
                x, y = np.min(adjusted_coords, axis = 0)
                width, height = 1 + np.max(adjusted_coords, axis = 0) - (x, y)
                quadtree_map = 1 - (self.occupancy_map == 0) # This makes all 0 to 0, and other values to 1
                self.quad_tree_root.update(quadtree_map, x, y, width, height) # !!!
                if verbose:
                    img_save_path = os.path.join(self.args.log_image_dir, "occupancy_"+str(self.step_time)+".jpg")
                    plt.imsave(img_save_path, self.occupancy_map, cmap = "gray")
                    log.info(f"Occupancy map saved at {img_save_path}")
    
    @property
    def free_points(self, occupancy_map):
        free_points = np.column_stack(np.where((1 - (occupancy_map == 0))==1))
        return free_points

    def update_occupancy_and_candidates(self, pointclouds, update_candidates = True, verbose = False):
        """
        Updates the BEV map content
        """
        self.update_occupancy_map(pointclouds, verbose)

        # if update_candidates:
        #     self.update_candidates(rgb_image, depth_image, verbose)

    def node_to_sim(self, node):
        if isinstance(node, Node):
            return [(node.x-self.quadtree_width/2)*self.voxel_size + self.init_world_pos[0], 
                    (node.y-self.quadtree_height/2)*self.voxel_size + self.init_world_pos[1], 
                    node.z] 
        if len(list(node))==2:
            return [(node[1]-self.quadtree_width/2)*self.voxel_size, (node[0]-self.quadtree_height/2)*self.voxel_size, node.z]
        else:
            raise TypeError(f"Point must be a Node or has length of 2 or 3, but got {type(node).__name__}")

    def transfer_to_node(self, point):
        if isinstance(point, Node):
            return point
        elif len(list(point))==3:
            return Node((point[0]-self.init_world_pos[0])/self.voxel_size+self.quadtree_width/2, 
                        (point[1]-self.init_world_pos[1])/self.voxel_size+self.quadtree_height/2, 
                        point[2])
        else:
            raise TypeError(f"Point must be a Node or has length of 2 or 3, but got {type(point).__name__}")

    def sample_points_between_two_points(self, start, end, step=1):
        x1, y1, z1 = start
        x2, y2, z2 = end
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        unit_vector = ((x2 - x1) / distance, (y2 - y1) / distance)

        num_samples = int(distance / step)
        sampled_points = [[
            x1 + n * step * unit_vector[0], 
            y1 + n * step * unit_vector[1],
            z1
         ] for n in range(num_samples + 1)]

        return sampled_points
    
    def navigate_p2p(self, current, target, verbose = False) -> list:
        """
        Sends the target to P2P Navigation for pathfinding.
        """
        # Code to navigate to the next target point
        current = self.transfer_to_node(current)
        target = self.transfer_to_node(target)
        # refresh the map before navigation
        quad_tree = deepcopy(self.quad_tree_root)
        radius = int(np.ceil(self.planner_config.agent_radius))
        area_bottom_left_x, area_bottom_left_y = int(current.x - radius), int(current.y - radius)
        area_width, area_height = int(2*radius), int(2*radius)
        quadtree_map = 1 - (self.occupancy_map == 0)

        quadtree_map[area_bottom_left_y: area_bottom_left_y + area_height, area_bottom_left_x: area_bottom_left_x + area_width] = np.ones((area_height, area_width)) # !!!
        # quadtree_map[area_bottom_left_x: area_bottom_left_x + area_width, area_bottom_left_y: area_bottom_left_y + area_height] = np.ones((area_width, area_height))

        # x, y = int(max(current.x - radius, 0)), int(max(current.y - radius, 0))
        # width, height = int(current.x + radius - x), int(current.y + radius - y)
        # quadtree_map = 1 - (self.occupancy_map == 0)
        # quadtree_map[y: y + height, x: x + width] = np.ones((height, width))
        quad_tree.update(quadtree_map, area_bottom_left_x, area_bottom_left_y, area_width, area_height)

        path_planner = PathPlanning(quad_tree, quadtree_map, 
                            agent_radius=self.planner_config.agent_radius,
                            last_scope=self.planner_config.last_scope,
                            goal_sampling_rate=self.planner_config.goal_sampling_rate,
                            max_iter=self.planner_config.max_iter,
                            extend_length=self.planner_config.extend_length,
                            consider_range=self.planner_config.consider_range) # Navigation method
        # path_planner = PathPlanning(self.bev_map.quad_tree_root, 1-(self.bev_map.occupancy_map==0), **self.planner_config) # Navigation method
        node, node_type= path_planner.rrt_star(current, target)
        if verbose:
            path_save_path = os.path.join(self.args.log_image_dir, "path_"+str(self.step_time)+".jpg")
            path_planner.plot_path(node, current, target, path_save_path)
            log.info(f"Path saved at {path_save_path}")

        path = []
        while node.parent is not None:
            path.append(self.node_to_sim(node))
            node = node.parent
        
        final_path = []
        path.reverse()
        start_point = self.node_to_sim(current)

        if len(path) > 1:
            interval_z = (path[-1][2] - start_point[2])/len(path)
            for i, point in enumerate(path):
                path[i][2] = start_point[2] + (i+1)*interval_z

        for end_point in path:
            sampled_points = self.sample_points_between_two_points(start_point, end_point)
            final_path.extend(sampled_points)
            start_point = end_point
        # if node_type != 0:
        #     final_path.append(self.node_to_sim(target))
        final_path = [tuple(i) for i in final_path]
        if len(final_path)>0:
            final_path.pop(0)
        return final_path, node_type

    def is_collision(self, current, target) -> bool:
        # format input
        current = self.transfer_to_node(current)
        target = self.transfer_to_node(target)
        
        path_planner = PathPlanning(self.quad_tree_root, self.occupancy_map==2, # ??? why 2 
                            agent_radius=self.planner_config.agent_radius,
                            last_scope=self.planner_config.last_scope,
                            goal_sampling_rate=self.planner_config.goal_sampling_rate,
                            max_iter=self.planner_config.max_iter,
                            extend_length=self.planner_config.extend_length,
                            consider_range=self.planner_config.consider_range) # Navigation method
        # path_planner = PathPlanning(self.bev_map.quad_tree_root, 1-(self.bev_map.occupancy_map==0), **self.planner_config) # Navigation method
        return not path_planner.collision_free(current, target)
    