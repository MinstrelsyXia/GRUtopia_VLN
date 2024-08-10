import os
import numpy as np
import pandas as pd
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from copy import deepcopy
from scipy.ndimage import binary_dilation
import json

from grutopia.core.util.log import log

from .path_planner import QuadTreeNode, Node, RRTstarPathPlanning, AStarPlanner

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
        print(f"quadtree width: {self.quadtree_width}, quadtree height: {self.quadtree_height}")
        # self.occupancy_map = np.ones((self.quadtree_height, self.quadtree_width)) * 255  # 2D occupancy map
        self.occupancy_map = np.zeros((self.quadtree_height, self.quadtree_width)) # 0 for Unknown, 2 for free, 255 for occupied, 255-2 for dilation
        self.quad_tree_root = QuadTreeNode(x=0, y=0, width=self.quadtree_width, height=self.quadtree_height,
                                           map_data = self.occupancy_map, 
                                           max_depth=self.quadtree_config.max_depth, threshold=self.quadtree_config.threshold)  # quadtrees
        
        # let the agent's initial point to be the origin of the map
        self.init_world_pos = np.array(robot_init_pose)
        self.robot_z = args.maps.robot_z  # The height(m) range of robot
        self.agent_radius = args.maps.agent_radius  # The radius(m) of robot
        self.dilation_structure = self.create_dilation_structure(self.agent_radius)
        self.last_obmap_pure = None

        # Attributes for path_planner
        self.planner_config = args.planners
        self.path_planner = AStarPlanner(map_width=self.quadtree_width,map_height=self.quadtree_height,max_step=self.planner_config.a_star_max_iter,
                            windows_head=self.args.windows_head,
                            for_llm=self.args.settings.use_llm,
                            verbose=True)
        
        log.info("BEVMap initialized")
    
    def reset(self):
        self.occupancy_map = np.zeros((self.quadtree_height, self.quadtree_width))
        self.quad_tree_root = QuadTreeNode(0, 0, map_data = self.occupancy_map, **self.quadtree_config)
        self.step_time = 0
        self.world_min_x, self.world_min_y = 0, 0
    
    def convert_world_to_map(self, point_cloud):
        # Note that the pointclouds have the world corrdinates that some values are very negative
        # We need to convert it into the map coordinates
        if len(point_cloud)==0:
            log.error(f"The shape of point cloud is not correct. The shape is {point_cloud.shape}.")
            return None
        point_cloud[...,:2] = point_cloud[..., :2] - self.init_world_pos[:2]

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
    def update_occupancy_map(self, point_cloud, robot_bottom_z, add_dilation=False, verbose = False, global_bev=False, robot_coords=None):
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
            else:
                pos_point_cloud = self.convert_world_to_map(point_cloud)
            pos_point_cloud = pd.DataFrame(pos_point_cloud)
            if not pos_point_cloud.isna().all().all():
                pos_point_cloud = pos_point_cloud.dropna().to_numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pos_point_cloud)
                downsampled_cloud = np.asarray(pcd.voxel_down_sample(voxel_size=self.voxel_size).points)

                adjusted_coords = (downsampled_cloud[:, :2]/self.voxel_size + [self.quadtree_width/2, self.quadtree_height/2]).astype(int) 
                adjusted_coords_with_z = np.hstack((adjusted_coords, downsampled_cloud[:,2].reshape(-1,1)))
                point_to_consider = adjusted_coords_with_z[(adjusted_coords_with_z[:, 0] < self.quadtree_width) & (adjusted_coords_with_z[:, 1] < self.quadtree_height)] 

                point_within_robot_z = point_to_consider[(point_to_consider[:,2]>=(robot_bottom_z+self.robot_z[0])) & (point_to_consider[:,2]<=(robot_bottom_z+self.robot_z[1]))].astype(int) # points that are within the robot's height range (occupancy)

                unique_data_0 = np.unique(point_within_robot_z[:, :2], axis=0)
                unique_data_all = np.unique(point_to_consider[:, :2], axis=0).astype(int)
                unique_data_1 = np.array(list(set(map(tuple, unique_data_all)) - set(map(tuple, unique_data_0)))).astype(int) # points that are not within the robot's height range (free)

                occupancy_map = np.zeros((self.quadtree_height, self.quadtree_width))
                if unique_data_1.size > 0:
                    occupancy_map[unique_data_1[:,1], unique_data_1[:,0]] = 2 # free
                if unique_data_0.size > 0: 
                    occupancy_map[unique_data_0[:,1],unique_data_0[:,0]] = 255 # occupied
                    if add_dilation:
                        # Create a mask for the free positions and dilate it
                        if self.last_obmap_pure is None:
                            self.last_obmap_pure = occupancy_map.copy() # pure obs_map without any dilation
                            self.occupancy_map = occupancy_map
                        else:
                            # combine the current occupancy with the last one
                            self.occupancy_map = np.maximum(self.last_obmap_pure,occupancy_map)
                            self.last_obmap_pure = self.occupancy_map.copy()
                        for i in range(1, self.args.maps.dilation_iterations):
                            ob_mask = np.logical_and(self.occupancy_map!=0, self.occupancy_map!=2)
                            expanded_ob_mask = binary_dilation(ob_mask, structure=self.dilation_structure, iterations=1)
                            self.occupancy_map[expanded_ob_mask&(np.logical_or(self.occupancy_map==0,self.occupancy_map==2))] = 255 - i*10
                    else:
                        last_obmap_pure = self.occupancy_map
                        self.occupancy_map = np.maximum(occupancy_map, last_obmap_pure)
                x, y = np.min(adjusted_coords, axis = 0)
                width, height = 1 + np.max(adjusted_coords, axis = 0) - (x, y)
                quadtree_map = 1 - (self.occupancy_map == 255) # This makes all 0 to 0, and other values to 1
                self.quad_tree_root.update(quadtree_map, x, y, width, height) # !!!
                if verbose:
                    if global_bev:
                        img_save_path = os.path.join(self.args.log_image_dir, "global_occupancy_"+str(self.step_time)+".jpg")
                    else:
                        img_save_path = os.path.join(self.args.log_image_dir, "occupancy_"+str(self.step_time)+".jpg")

                    # draw the robot's position using the red 'x' mark
                    self.plot_occupancy_map(self.occupancy_map, robot_coords, img_save_path, for_llm=self.args.settings.use_llm)
                    log.info(f"Occupancy map saved at {img_save_path}")
                return True
    
    def plot_occupancy_map(self, occupancy_map, robot_coords, img_save_path, for_llm=False):
        # Define the color map
        cmap = mcolors.ListedColormap(['white', 'green', 'gray', 'black'])  # Colors for 0, between 1-254, 2, 255
        bounds = [0, 1, 3, 254, 256]  # Boundaries for the colors
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        convert_robot_coords = ((robot_coords[:2]- self.init_world_pos[:2])/self.voxel_size + [self.quadtree_width/2, self.quadtree_height/2])

        plt.figure(figsize=(10, 10))
        plt.xlim(0, occupancy_map.shape[1])  # Fix: should be 0 to width
        plt.ylim(0, occupancy_map.shape[0])  # Fix: should be 0 to height
        plt.imshow(occupancy_map, cmap=cmap, norm=norm)
        plt.scatter(convert_robot_coords[0], convert_robot_coords[1], color='blue', marker='o', label="current position: (%.2f, %.2f)"%(convert_robot_coords[0], convert_robot_coords[1]))
        if for_llm:
            # Add text annotation for the x-coordinate
            # x_coord = convert_robot_coords[0]
            # y_coord = convert_robot_coords[1]
            # plt.text(x_coord, y_coord, f'({x_coord}, {y_coord})', fontsize=12, color='black', ha='right')
            # Customize the appearance of the grid
            major_ticks_x = np.linspace(0, occupancy_map.shape[1], 50)
            # minor_ticks_x = np.linspace(0, occupancy_map.shape[1], 50)
            major_ticks_y = np.linspace(0, occupancy_map.shape[0], 50)
            # minor_ticks_y = np.linspace(0, occupancy_map.shape[0], 50)

            # Set major and minor ticks
            plt.xticks(major_ticks_x, rotation=45)
            plt.yticks(major_ticks_y)
            # plt.xticks(minor_ticks_x, minor=True)
            # plt.yticks(minor_ticks_y, minor=True)

            # plt.grid(which='minor', linewidth='0.75', color='gray', alpha=0.6)
            plt.grid(which='major', linewidth='0.75', color='gray', alpha=0.75)

            plt.legend()

            plt.xlabel('x')
            plt.ylabel('y')
        plt.savefig(img_save_path, pad_inches=0, bbox_inches='tight', dpi=100)
        plt.close()
    
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

    def node_to_sim(self, node, z=0):
        if isinstance(node, Node):
            return [(node.x-self.quadtree_width/2)*self.voxel_size + self.init_world_pos[0], 
                    (node.y-self.quadtree_height/2)*self.voxel_size + self.init_world_pos[1], 
                    node.z] 
        if len(list(node))==2:
            return [(node[0]-self.quadtree_width/2)*self.voxel_size + self.init_world_pos[0], (node[1]-self.quadtree_height/2)*self.voxel_size + self.init_world_pos[1], z]
        else:
            raise TypeError(f"Point must be a Node or has length of 2 or 3, but got {type(node).__name__}")

    def transfer_to_node(self, point):
        if isinstance(point, Node):
            return point
        elif len(list(point))==3:
            x = (point[0]-self.init_world_pos[0])/self.voxel_size+self.quadtree_width/2
            y = (point[1]-self.init_world_pos[1])/self.voxel_size+self.quadtree_height/2
            return Node(x, y, point[2])
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
    
    def navigate_p2p(self, current, target, verbose=False):
        if self.args.planners.method == 'rrt_star':
            return self.navigate_p2p_rrt_star(current, target, verbose=verbose)
        elif self.args.planners.method == 'a_star':
            return self.navigate_p2p_a_star(current, target, verbose=verbose)
        else:
            raise NotImplementedError(f"Navigation method {self.args.planners.method} is not implemented.")
    
    def navigate_p2p_a_star(self, start, goal, verbose = False) -> list:
        """
        Sends the target to P2P Navigation for pathfinding.
        return final_path, result_type. result_type: 0->success, 1->path found but goal occupied, 2->no path found
        """
        # Code to navigate to the next target point
        current = self.transfer_to_node(start)
        target = self.transfer_to_node(goal)
        
        # refresh the map before navigation
        quad_tree = deepcopy(self.quad_tree_root)
        radius = int(np.ceil(self.args.maps.agent_radius))
        area_bottom_left_x, area_bottom_left_y = int(current.x - radius), int(current.y - radius)
        area_width, area_height = int(2*radius), int(2*radius)
        quadtree_map = self.occupancy_map

        quadtree_map[area_bottom_left_y: area_bottom_left_y + area_height, area_bottom_left_x: area_bottom_left_x + area_width] = np.zeros((area_height, area_width)) 

        quad_tree.update(quadtree_map, area_bottom_left_x, area_bottom_left_y, area_width, area_height)

        
        paths, find_flag = self.path_planner.planning(current.y, current.x, target.y, target.x, # x and y are reversed in AStarPlanner
                                      obs_map=quadtree_map,
                                      min_final_meter=self.planner_config.last_scope,
                                      img_save_path=os.path.join(self.args.log_image_dir, "path_"+str(self.step_time)+".jpg"))
        
        transfer_paths = []
        for node in paths:
            world_coords = self.node_to_sim([node[1], node[0]], current.z)
            transfer_paths.append([world_coords[0], world_coords[1], current.z])
        
        if verbose:
            # save the occupancy and coordinates for verification
            coord_data = {'start_point':[current.y,current.x], 'end_point':[target.y,target.x]}
            with open(self.args.root_dir+'/vln/src/local_nav/coords.json', 'w') as f:
                json.dump(coord_data, f)
            np.save(self.args.root_dir+'/vln/src/local_nav/obs_map.npy', np.logical_not(quadtree_map))
        
        return transfer_paths, find_flag

    def navigate_p2p_rrt_star(self, current, target, verbose = False) -> list:
        """
        Sends the target to P2P Navigation for pathfinding.
        return final_path, result_type. result_type: 0->success, 1->path found but goal occupied, 2->no path found
        """
        # Code to navigate to the next target point
        current = self.transfer_to_node(current)
        target = self.transfer_to_node(target)
        # refresh the map before navigation
        quad_tree = deepcopy(self.quad_tree_root)
        radius = int(np.ceil(self.planner_config.agent_radius))
        area_bottom_left_x, area_bottom_left_y = int(current.x - radius), int(current.y - radius)
        area_width, area_height = int(2*radius), int(2*radius)
        quadtree_map = 1 - (self.occupancy_map == 0) # 0->0, others -> 1

        quadtree_map[area_bottom_left_y: area_bottom_left_y + area_height, area_bottom_left_x: area_bottom_left_x + area_width] = np.ones((area_height, area_width)) # !!!
        quad_tree.update(quadtree_map, area_bottom_left_x, area_bottom_left_y, area_width, area_height)

        path_planner = RRTstarPathPlanning(occupy_map=quad_tree, origin_map=quadtree_map, 
                            agent_radius=self.planner_config.agent_radius,
                            last_scope=self.planner_config.last_scope,
                            goal_sampling_rate=self.planner_config.goal_sampling_rate,
                            max_iter=self.planner_config.rrt_star_max_iter,
                            extend_length=self.planner_config.extend_length,
                            consider_range=self.planner_config.consider_range) # Navigation method
        # path_planner = PathPlanning(self.bev_map.quad_tree_root, 1-(self.bev_map.occupancy_map==0), **self.planner_config) # Navigation method
        node, node_type= path_planner.rrt_star(current, target)
        if node_type == 2:
            log.info("Path planning fails to find the path from the current point to the next point.")
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
        return not self.collision_free(current, target, self.planner_config.agent_radius, self.quad_tree_root)
    
    def collision_free(self, from_node, to_node, agent_radius, occupy_map):
        """
        Check if the path from from_node to to_node collides with any obstacles in the environment.
        
        Parameters:
        from_node -- The starting Node object with coordinates (x, y)
        to_node -- The ending Node object with coordinates (x, y)
        
        Returns:
        True if no collision occurs, otherwise False
        """
        if from_node == to_node or (from_node.x==to_node.x and from_node.y==to_node.y):
            return False
        # Calculate direction vector from from_node to to_node
        direction_vector = np.array([to_node.y - from_node.y, to_node.x - from_node.x])
        
        # Normalize the direction vector
        norm_direction_vector = direction_vector / np.linalg.norm(direction_vector)
        
        # Extend to_node by agent_radius in the direction of to_node
        extended_to_node = np.array([to_node.y, to_node.x]) + norm_direction_vector * agent_radius
        
        # Calculate perpendicular (normal) vector to the direction
        norm_vector = np.array([-norm_direction_vector[1], norm_direction_vector[0]])
        
        # Calculate the four vertices of the parallelogram, ordered clockwise
        vertices = [
            (from_node.y + norm_vector[0] * agent_radius, from_node.x + norm_vector[1] * agent_radius),
            (extended_to_node[0] + norm_vector[0] * agent_radius, extended_to_node[1] + norm_vector[1] * agent_radius),
            (extended_to_node[0] - norm_vector[0] * agent_radius, extended_to_node[1] - norm_vector[1] * agent_radius),
            (from_node.y - norm_vector[0] * agent_radius, from_node.x - norm_vector[1] * agent_radius),
            (from_node.y + norm_vector[0] * agent_radius, from_node.x + norm_vector[1] * agent_radius)
        ]

        # Step 2: Use quadtree for coarse collision detection
        # self.occupy_map.draw_polygon_on_map(self.origin_map, vertices)
        collision_free, bounding_box = occupy_map.query(vertices)
        # print(collision_free)
        if collision_free:
            # Quadtree query indicates no collision will occur
            return True
        return False
    