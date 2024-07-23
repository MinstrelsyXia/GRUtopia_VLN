import os
import numpy as np
import pandas as pd
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.cluster import DBSCAN, KMeans
from transformers.image_transforms import rgb_to_id
from transformers import DetrFeatureExtractor, DetrForSegmentation, SegformerFeatureExtractor, SegformerForSemanticSegmentation
from skimage.morphology import square, binary_erosion, binary_dilation

from grutopia.core.util.log import log

from .path_planner import QuadTreeNode, Node, PathPlanning

class BEVMap:
    def __init__(self, args, robot_z=(0.05, 0.5), voxel_size=0.01):
        self.args = args
        
        self.step_time = 0
        # Attributes for occupancy_map
        quadtree_config = args.maps.quadtree_config
        quadtree_config['width'], quadtree_config['height'] = int(quadtree_config['width']/voxel_size), int(quadtree_config['height']/voxel_size)
        self.quadtree_config = quadtree_config
        self.quadtree_width = self.quadtree_config['width']
        self.quadtree_height = self.quadtree_config['height']
        self.voxel_size = voxel_size  # Resolution to present the map
        self.robot_z = robot_z  # The height(m) range of robot
        self.occupancy_map = np.ones((self.quadtree_height, self.quadtree_width))  # 2D occupancy map
        self.quad_tree_root = QuadTreeNode(0, 0, map_data = self.occupancy_map, **self.quadtree_config)  # quadtrees

        # Attributes for path_planner
        self.planner_config = args.maps.planner_config
        
        log.info("BEVMap initialized")
    
    def reset(self):
        self.occupancy_map = np.ones((self.quadtree_height, self.quadtree_width))
        self.quad_tree_root = QuadTreeNode(0, 0, map_data = self.occupancy_map, **self.quadtree_config)
        self.step_time = 0
        
    ######################## update_occupancy_map ########################
    def update_occupancy_map(self, point_cloud, verbose = False):
        """
        Updates the occupancy map based on the new point cloud data. Optionally updates using all stored
        point clouds if update_with_global is True. 
        
        Args:
            point_cloud (numpy.ndarray): The nx3 array containing new point cloud data (x, y, z).
            update_with_global (bool): If True, updates the map using all stored point clouds.
        """
        # Store the new point cloud after downsampling
        if point_cloud is not None:
            point_cloud = pd.DataFrame(point_cloud)
            if not point_cloud.isna().all().all():
                point_cloud = point_cloud.dropna().to_numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_cloud)
                downsampled_cloud = np.asarray(pcd.voxel_down_sample(voxel_size=self.voxel_size).points)

                adjusted_coords = (downsampled_cloud[:, :2]/self.voxel_size + [self.quadtree_width/2, self.quadtree_height/2]).astype(int) 
                adjusted_coords_with_z = np.hstack((adjusted_coords, downsampled_cloud[:,2].reshape(-1,1)))
                point_to_consider = adjusted_coords_with_z[(adjusted_coords_with_z[:, 0] >= 0) & (adjusted_coords_with_z[:, 0] < self.quadtree_height) & (adjusted_coords_with_z[:, 1] >= 0) & (adjusted_coords_with_z[:, 1] < self.quadtree_width)]

                point_0 = point_to_consider[(point_to_consider[:,2]>(self.robot_z[0])) & (point_to_consider[:,2]<(self.robot_z[1]))].astype(int)

                unique_data_0 = np.unique(point_0[:, :2], axis=0)
                unique_data_all = np.unique(point_to_consider[:, :2], axis=0).astype(int)
                unique_data_1 = np.array(list(set(map(tuple, unique_data_all)) - set(map(tuple, unique_data_0)))).astype(int)

                last_map = 1 - (self.occupancy_map == 0)
                if unique_data_1.size > 0:
                    self.occupancy_map[unique_data_1[:,1], unique_data_1[:,0]]=2
                if unique_data_0.size > 0: 
                    self.occupancy_map[unique_data_0[:,1],unique_data_0[:,0]]=0
                self.occupancy_map = self.occupancy_map*last_map
                x, y = np.min(adjusted_coords, axis = 0)
                width, height = 1 + np.max(adjusted_coords, axis = 0) - (x, y)
                quadtree_map = 1 - (self.occupancy_map == 0)
                self.quad_tree_root.update(quadtree_map, x, y, width, height)
                if verbose:
                    plt.imsave(os.path.join(self.args.log_image_dir, "occupancy_"+str(self.step_time)+".jpg"), self.occupancy_map, cmap = "gray")
    
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
            return [(node.x-self.quadtree_width/2)*self.voxel_size, (node.y-self.quadtree_height/2)*self.voxel_size, node.z]
        if len(list(node))==2:
            return [(node[1]-self.quadtree_width/2)*self.voxel_size, (node[0]-self.quadtree_height/2)*self.voxel_size, node.z]
        else:
            raise TypeError(f"Point must be a Node or has length of 2 or 3, but got {type(node).__name__}")

    def transfer_to_node(self, point):
        if isinstance(point, Node):
            return point
        elif len(list(point))==3:
            return Node(point[0]/self.voxel_size+self.quadtree_width/2, point[1]/self.voxel_size+self.quadtree_height/2, point[2])
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
        radius = int(np.ceil(self.planner_config['agent_radius']))
        x, y = int(max(current.x - radius, 0)), int(max(current.y - radius, 0))
        width, height = int(current.x + radius - x), int(current.y + radius - y)
        quadtree_map = 1 - (self.occupancy_map == 0)
        quadtree_map[y: y + height, x: x + width] = np.ones((height, width))
        quad_tree.update(quadtree_map, x, y, width, height)

        path_planner = PathPlanning(quad_tree, quadtree_map, **self.planner_config) # Navigation method
        # path_planner = PathPlanning(self.bev_map.quad_tree_root, 1-(self.bev_map.occupancy_map==0), **self.planner_config) # Navigation method
        node, node_type= path_planner.rrt_star(current, target)
        if verbose:
            path_planner.plot_path(node, current, target, os.path.join(self.args.log_image_dir, 'path_'+str(self.step_time)+'.jpg'))
        path = []
        while node.parent is not None:
            path.append(self.node_to_sim(node))
            node = node.parent
        
        final_path = []
        path.reverse()
        start_point = self.node_to_sim(current)
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
    
    