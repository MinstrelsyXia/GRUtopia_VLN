import io
import os
import re
import torch
import random
random.seed(2024)
import numpy as np
import pandas as pd
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from transformers.image_transforms import rgb_to_id
from transformers import DetrFeatureExtractor, DetrForSegmentation, SegformerFeatureExtractor, SegformerForSemanticSegmentation
from skimage.morphology import square, binary_erosion, binary_dilation

from llm_agent.utils.path_planner import QuadTreeNode
from llm_agent.utils.utils import annotate_original_map, visualize_panoptic_segmentation_loc
from llm_agent.utils.utils_omni import get_camera_data
# os.environ['HF_HOME'] = 'empty'
# detection = {'num': {'fe': DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic"), 'model': DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")}, 'location': {'fe': SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512"), 'model': SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")}}

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# detection = {'num': {'fe': DetrFeatureExtractor.from_pretrained("local_model/facebook_detr_resnet_50_panoptic_fe", local_files_only= True), 'model': DetrForSegmentation.from_pretrained("local_model/facebook_detr_resnet_50_panoptic_model", local_files_only= True)}, 'location': {'fe': SegformerFeatureExtractor.from_pretrained("local_model/nvidia_segformer_b3_finetuned_ade_fe", local_files_only= True), 'model': SegformerForSemanticSegmentation.from_pretrained("local_model/nvidia_segformer_b3_finetuned_ade_model", local_files_only= True)}}
detection = {
    'num': {'fe': DetrFeatureExtractor.from_pretrained(os.path.join(ROOT_DIR,"local_model/facebook_detr_resnet_50_panoptic_fe"), local_files_only= True), 
    'model': DetrForSegmentation.from_pretrained(os.path.join(ROOT_DIR, "local_model/facebook_detr_resnet_50_panoptic_model"), local_files_only= True)}, 
    'location': {'fe': SegformerFeatureExtractor.from_pretrained(os.path.join(ROOT_DIR, "local_model/nvidia_segformer_b3_finetuned_ade_fe"), local_files_only= True), 
    'model': SegformerForSemanticSegmentation.from_pretrained(os.path.join(ROOT_DIR, "local_model/nvidia_segformer_b3_finetuned_ade_model"), local_files_only= True)}}

class BEVMap:
    def __init__(self, llm, vlm, camera_parameters: list, quadtree_config: dict, robot_z = (0.05, 0.5), voxel_size = 0.01):
        # llm and vlm
        self.llm = llm
        self.vlm = vlm

        self.camera_parameters = camera_parameters

        # Attributes for occupancy_map
        quadtree_config['width'], quadtree_config['height'] = int(quadtree_config['width']/voxel_size), int(quadtree_config['height']/voxel_size)
        self.quadtree_config = quadtree_config
        self.quadtree_width = self.quadtree_config['width']
        self.quadtree_height = self.quadtree_config['height']
        self.voxel_size = voxel_size  # Resolution to present the map
        self.robot_z = robot_z  # The height(m) range of robot
        self.occupancy_map = np.ones((self.quadtree_height, self.quadtree_width))  # 2D occupancy map
        self.quad_tree_root = QuadTreeNode(0, 0, map_data = self.occupancy_map, **self.quadtree_config)  # quadtree
        
        # Attributes for candidates
        self.candidates = []  # List to store information about candidates        
        self.detection = detection
        self.category_data = pd.read_excel(os.path.join(ROOT_DIR, 'llm_agent/utils/detection_info/result.xlsx'), engine='openpyxl')  # map from category in ty to the category in the detection model   
        self.detection_id = None
        self.goal = None  # The category of the candidates
        
    def reset(self):
        self.occupancy_map = np.ones((self.quadtree_height, self.quadtree_width))
        self.quad_tree_root = QuadTreeNode(0, 0, map_data = self.occupancy_map, **self.quadtree_config)
        self.candidates = []
        self.detection_id = None
        self.goal = None
        
    def set_goal(self, goal):
        if goal:
            self.goal = ('_'.join(goal.split())).lower()
        else:
            self.goal = -1
        try:
            self.detection_id = {'num': int(self.category_data.loc[self.category_data['Name']==self.goal,'num ID'].iloc[0]), 'loc': int(self.category_data.loc[self.category_data['Name']==self.goal,'location ID'].iloc[0])}
        except:
            self.detection_id = {'num': -1, 'loc': -1}

    def update_occupancy_and_candidates(self, update_candidates = True, verbose = False):
        """
        Updates the BEV map content
        """
        camera_data = get_camera_data(self.camera_parameters['camera'], self.camera_parameters['resolution'], ["pointcloud", "rgba", "depth", "bbox"])
        additional_view = get_camera_data(self.camera_parameters['camera_add'], self.camera_parameters['resolution'], ["pointcloud", "rgba", "bbox"])["pointcloud"]['data']
        if camera_data['rgba'] is not None:
            rgb_image = camera_data['rgba'][..., :3]
            if verbose:
                if not os.path.exists(os.path.join(ROOT_DIR, 'images', str(self.step_time))):
                    os.makedirs(os.path.join(ROOT_DIR, 'images', str(self.step_time)))
                plt.imsave(os.path.join(ROOT_DIR, 'images', str(self.step_time), 'rgb_'+str(self.step_time)+'.jpg'), rgb_image)
                
        if camera_data['rgba'] is not None and camera_data['pointcloud']['data'] is not None and additional_view is not None and camera_data["pointcloud"]['data'].size > 0 and camera_data['rgba'].size > 0 and additional_view.size > 0:
            rgb_image = camera_data['rgba'][..., :3]
            depth_image = camera_data['depth']
            if verbose:
                if not os.path.exists(os.path.join(ROOT_DIR, 'images', str(self.step_time))):
                    os.makedirs(os.path.join(ROOT_DIR, 'images', str(self.step_time)))
                plt.imsave(os.path.join(ROOT_DIR, 'images', str(self.step_time), 'rgb_'+str(self.step_time)+'.jpg'), rgb_image)
                plt.imsave(os.path.join(ROOT_DIR, 'images', str(self.step_time), 'depth_'+str(self.step_time)+'.jpg'), depth_image)
            point_cloud = np.vstack((camera_data["pointcloud"]['data'], additional_view))
            self.update_occupancy_map(point_cloud, verbose)
            if update_candidates:
                self.update_candidates(rgb_image, depth_image, verbose)
            return rgb_image, depth_image
        else:
            return None, None

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
                    if not os.path.exists(os.path.join(ROOT_DIR, 'images', str(self.step_time))):
                        os.makedirs(os.path.join(ROOT_DIR, 'images', str(self.step_time)))
                    plt.imsave(os.path.join(ROOT_DIR, 'images', str(self.step_time), "occupancy_"+str(self.step_time)+".jpg"), self.occupancy_map, cmap = "gray")

    ######################## update_candidates ########################
    def update_candidates(self, rgb_image, depth_image, verbose = False):
        """
        Updates the candidates and their descriptions using RGB and Depth images.
        """
        if depth_image is not None:
            assert isinstance(depth_image, np.ndarray)
            assert depth_image.size>0
            if depth_image.size>0 and not np.all(np.isinf(depth_image)):
                assert rgb_image.size > 0
                # Using semantic segmentation to get some bouding box of possible new candidates
                possible_candidates = self.seg_instance_on_image(rgb_image, verbose)
                if possible_candidates is None:
                    return
                # img = Image.open(possible_candidates['query_image_buf']).save('anno.png')
                # Using VLM to determine the possible candidates and get caption of them
                self.get_description(possible_candidates, verbose)

    def seg_instance_on_image(self, rgb_image, verbose = False) -> dict:
        """
        Updates the candidates and their descriptions using RGB and Depth images.
        """
        assert self.goal is not None
        rgb_image = Image.fromarray(rgb_image.astype(np.uint8)).convert("RGB")
        # num detection
        inputs_num = self.detection['num']['fe'](images=rgb_image, return_tensors="pt")
        outputs_num = self.detection['num']['model'](**inputs_num)
        processed_sizes = torch.as_tensor(inputs_num["pixel_values"].shape[-2:]).unsqueeze(0)
        result_num = self.detection['num']['fe'].post_process_panoptic(outputs_num, processed_sizes)[0]
        instances = [i for i in result_num['segments_info'] if i['category_id']==self.detection_id['num']]
        # location detection
        inputs_location = self.detection['location']['fe'](images=rgb_image, return_tensors="pt")
        outputs_location = self.detection['location']['model'](**inputs_location)
        result_location = self.detection['location']['fe'].post_process_semantic_segmentation(outputs_location,target_sizes=[(rgb_image.size[1], rgb_image.size[0])])
        if verbose:
            if not os.path.exists(os.path.join(ROOT_DIR, 'images', str(self.step_time))):
                os.makedirs(os.path.join('images', str(self.step_time)))
            visualize_panoptic_segmentation_loc(rgb_image, result_location[0], self.detection['location']['model'], os.path.join('images', str(self.step_time), "seg_"+str(self.step_time)+".jpg"))
        interest = self.detection_id['loc'] in np.unique(result_location[0])
        if interest and len(instances)>0:
            instance_num = len(instances)
            cluster_map = np.transpose(np.full(rgb_image.size, -1))
            goal_seg = result_location[0] == self.detection_id['loc']
            occupied_points = np.column_stack(np.where(goal_seg == 1))
            # get instances
            kmeans = KMeans(n_clusters=instance_num).fit(occupied_points)
            labels = kmeans.labels_
            cluster_map[occupied_points[:, 0], occupied_points[:, 1]] = labels
            # plt.scatter(occupied_points[:, 0], occupied_points[:, 1], c=labels, cmap='viridis')
            # plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red', marker='x') 
            # merge candidates
            new_candidates = self.merge_candidates(cluster_map)
            if len(new_candidates) == 0:
                return None
            id_to_keep = [t[0] for t in new_candidates]
            cluster_map = np.where(np.isin(cluster_map, id_to_keep), cluster_map, -1)
            anno_image = annotate_original_map(rgb_image, cluster_map)
            return anno_image, new_candidates
        elif interest and len(instances)==0:
            instance_num = len(instances)
            cluster_map = np.transpose(np.full(rgb_image.size, -1))
            goal_seg = result_location[0] == self.detection_id['loc']
            # get instances
            cluster_map += goal_seg.numpy()
            # merge candidates
            new_candidates = self.merge_candidates(cluster_map)
            if len(new_candidates) == 0:
                return None
            id_to_keep = [t[0] for t in new_candidates]
            cluster_map = np.where(np.isin(cluster_map, id_to_keep), cluster_map, -1)
            anno_image = annotate_original_map(rgb_image, cluster_map)
            return anno_image, new_candidates
        elif len(instances)>0 and not interest:
            panoptic_seg = Image.open(io.BytesIO(result_num["png_string"])).resize(rgb_image.size)
            panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
            # retrieve the ids corresponding to each mask
            panoptic_seg_id = rgb_to_id(panoptic_seg)
            # visualize_panoptic_segmentation_num(rgb_image, panoptic_seg_id, {i['id']:i['category_id'] for i in result_num['segments_info']}, self.detection['num']['model'])
            # get instances
            cluster_map = np.transpose(np.full(rgb_image.size, -1))
            for i, instance in enumerate(instances): 
                goal_seg = panoptic_seg_id == instance['id']
                selem = square(min(int((rgb_image.size[0]+rgb_image.size[1])/100), 20))
                goal_seg = binary_dilation(binary_erosion(goal_seg, selem), selem)
                occupied_points = np.column_stack(np.where(goal_seg == 1))
                cluster_map[occupied_points[:, 0], occupied_points[:, 1]] = i
            new_candidates = self.merge_candidates(cluster_map)
            if len(new_candidates) == 0:
                return None
            id_to_keep = [t[0] for t in new_candidates]
            cluster_map = np.where(np.isin(cluster_map, id_to_keep), cluster_map, -1)
            anno_image = annotate_original_map(rgb_image, cluster_map)
            return anno_image, new_candidates
        return None       

    def get_description(self, possible_candidates: tuple, verbose = False):        
        image_buf, new_candidates = possible_candidates
        if verbose:
            Image.open(image_buf).save('anno.png')
        #################################################### DEBUG #################################################### 
        Image.open(image_buf).save(os.path.join(self.save_path, str(self.step_time)+ '.png'))
        ###############################################################################################################
        result = self.vlm.get_answer('get_description', goal = self.goal, image = image_buf)
        #################################################### DEBUG #################################################### 
        result_save = result if result else 'None'
        with open(os.path.join(self.save_path, str(self.step_time)+ '.txt'),'w') as f:
            f.write(result_save) 
        ###############################################################################################################
        if result and '-1' not in result:
            lines = result.strip().split('\n')
            results = []
            for line in lines:
                try:
                    number = int(re.search(r'\d+', line.split(':')[0]).group())
                    description = line.split(':')[1]
                    results.append((number, description))
                except:
                    pass
            result = {int(region): description.strip() for region, description in results}
        else:
            result = {}

        for region, new_candidate in new_candidates:
            new_candidate['description'] = []

            if region in result:
                new_candidate['description'] = [result[region]]
                self.candidates.append(new_candidate)
            # self.candidates.append(new_candidate)
                
    
    def merge_candidates(self, instance_location: np.ndarray):
        instance_id = set(np.unique(instance_location)) - {-1}
        new_candidates = []
        for idx in instance_id:
            interest_region = (instance_location == idx)
            interest_pc = self.rgb_depth_to_point_cloud(interest_region)
            if interest_pc is not None:
                center = interest_pc[0, :]
                db = DBSCAN(eps=0.3, min_samples=10).fit(interest_pc)
                labels = db.labels_
                unique_clusters = set(labels) - {-1}
                
                # Initialize the closest cluster information
                closest_cluster = {'min_distance': np.inf, 'cluster': None}

                # Find the closest cluster to the candidate's center
                for k in unique_clusters:
                    points = interest_pc[labels == k]
                    centroid = np.mean(points, axis=0)
                    distances = np.linalg.norm(points - centroid, axis=1)
                    radius = distances.mean() + distances.std()
                    to_center = np.linalg.norm(center - centroid)

                    # Check if this cluster is closer to the center than the current closest
                    if to_center < closest_cluster['min_distance']:
                        closest_cluster = {'min_distance': to_center, 'cluster': {'points': points, 'centroid': centroid, 'radius': radius}}
                if closest_cluster['min_distance']==np.inf:
                    continue
                # Check against existing candidates
                new_flag = True
                for candidate in self.candidates:
                    distance = np.linalg.norm(candidate['centroids'] - closest_cluster['cluster']['centroid'])
                    if distance < candidate['radius'] or distance < closest_cluster['cluster']['radius'] or distance < 1:
                        new_flag = False
                        candidate['point_clouds'].append(closest_cluster['cluster']['points'])
                        all_points = np.concatenate(candidate['point_clouds'], axis=0)
                        candidate['centroids'] = np.mean(all_points, axis=0)
                        candidate['radius'] = np.mean(np.linalg.norm(all_points - candidate['centroids'], axis=1)) + np.std(np.linalg.norm(all_points - candidate['centroids'], axis=1))
                        break  # If merged, no need to check other candidates

                # If not merged, add as a new candidate
                if new_flag:
                    new_candidates.append((idx, {'not_the_goal': False, 'point_clouds': [closest_cluster['cluster']['points']], 'centroids': closest_cluster['cluster']['centroid'], 'radius': closest_cluster['cluster']['radius']}))
        return new_candidates

    def rgb_depth_to_point_cloud(self, mask):
        camera_data = get_camera_data(self.camera_parameters['camera'], self.camera_parameters['resolution'], ['pointcloud', 'depth'])
        point_cloud = camera_data['pointcloud']['data']
        depth = camera_data['depth']
        y_center, x_center = np.mean(np.where(mask==1),  axis=1)
        center_index = int(np.ceil(y_center-1)*mask.shape[1]+x_center) - np.sum((((depth.ravel()!=np.inf) & (depth.ravel()!=np.nan))==False)[:int(np.ceil(y_center-1)*mask.shape[1]+x_center)])
        try:
            point_center = point_cloud[center_index, :]
            mask = mask.ravel()[(depth.ravel()!=np.inf) & (depth.ravel()!=np.nan)]
            point_cloud = point_cloud[mask==1]
        except:
            return None
        point_cloud = np.vstack((point_center, point_cloud))
        return point_cloud 
    
    def get_frontier(self):
        candidates = [[0, 0], [self.quadtree_height/2, 0], [self.quadtree_height, 0], [self.quadtree_height, self.quadtree_width/2], [self.quadtree_height, self.quadtree_width], [self.quadtree_height/2, self.quadtree_width], [0, self.quadtree_width], [0, self.quadtree_width/2]]
        candidates.append(np.mean(np.where(self.occupancy_map==1),  axis=1))
        return random.choice(candidates)
    
if __name__ == "__main__":
    import yaml
    with open('llm_agent/agent_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    map_config = config['map_config']
    bevmap = BEVMap(**map_config)
    img = np.array(Image.open('rgb.png').convert("RGB"))
    bevmap.goal = 'nothing'
    bevmap.detection_id = -1
    bevmap.seg_instance_on_image(img)