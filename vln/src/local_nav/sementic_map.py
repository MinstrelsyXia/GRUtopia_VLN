import torch
import numpy as np
import torch.nn as nn
# class Semantic_Mapping(nn.Module):

#     """
#     Semantic_Mapping
#     Input: obs, pose_obs, maps_last, poses_last
#     Output: fp_map_pred, map_pred, pose_pred, current_poses
#     """

#     def __init__(self, args):
#         super(Semantic_Mapping, self).__init__()

#         self.device = args.device
#         self.screen_h = args.frame_height
#         self.screen_w = args.frame_width
#         self.resolution = args.map_resolution
#         self.z_resolution = args.map_resolution
#         self.map_size_cm = args.map_size_cm // args.global_downscaling
#         self.n_channels = 3
#         self.vision_range = args.vision_range
#         self.dropout = 0.5
#         self.fov = args.hfov
#         self.du_scale = args.du_scale
#         self.cat_pred_threshold = args.cat_pred_threshold
#         self.exp_pred_threshold = args.exp_pred_threshold
#         self.map_pred_threshold = args.map_pred_threshold
#         self.num_sem_categories = args.num_sem_categories

#         self.max_height = int(360 / self.z_resolution)
#         self.min_height = int(-40 / self.z_resolution)
#         self.agent_height = args.camera_height * 100.
#         self.shift_loc = [self.vision_range *
#                           self.resolution // 2, 0, np.pi / 2.0]
#         self.camera_matrix = du.get_camera_matrix(
#             self.screen_w, self.screen_h, self.fov)

#         self.pool = ChannelPool(1)

#         vr = self.vision_range

#         self.init_grid = torch.zeros(
#             args.num_processes, 1 + self.num_sem_categories, vr, vr,
#             self.max_height - self.min_height
#         ).float().to(self.device)
#         self.feat = torch.ones(
#             args.num_processes, 1 + self.num_sem_categories,
#             self.screen_h // self.du_scale * self.screen_w // self.du_scale
#         ).float().to(self.device)

#     def forward(self, obs, pose_obs, maps_last, poses_last):
#         bs, c, h, w = obs.size()
#         depth = obs[:, 3, :, :]

#         point_cloud_t = du.get_point_cloud_from_z_t(
#             depth, self.camera_matrix, self.device, scale=self.du_scale)

#         agent_view_t = du.transform_camera_view_t(
#             point_cloud_t, self.agent_height, 0, self.device)

#         agent_view_centered_t = du.transform_pose_t(
#             agent_view_t, self.shift_loc, self.device)

#         max_h = self.max_height
#         min_h = self.min_height
#         xy_resolution = self.resolution
#         z_resolution = self.z_resolution
#         vision_range = self.vision_range
#         XYZ_cm_std = agent_view_centered_t.float()
#         XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
#         XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
#                                vision_range // 2.) / vision_range * 2.
#         XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
#         XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
#                               (max_h + min_h) // 2.) / (max_h - min_h) * 2.
#         self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
#             obs[:, 4:, :, :]
#         ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

#         XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
#         XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
#                                      XYZ_cm_std.shape[1],
#                                      XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

#         voxels = du.splat_feat_nd(
#             self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)

#         min_z = int(25 / z_resolution - min_h)
#         max_z = int((self.agent_height + 1) / z_resolution - min_h)

#         agent_height_proj = voxels[..., min_z:max_z].sum(4)
#         all_height_proj = voxels.sum(4)

#         fp_map_pred = agent_height_proj[:, 0:1, :, :]
#         fp_exp_pred = all_height_proj[:, 0:1, :, :]
#         fp_map_pred = fp_map_pred / self.map_pred_threshold
#         fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
#         fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
#         fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

#         pose_pred = poses_last

#         agent_view = torch.zeros(bs, c,
#                                  self.map_size_cm // self.resolution,
#                                  self.map_size_cm // self.resolution
#                                  ).to(self.device)

#         x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
#         x2 = x1 + self.vision_range
#         y1 = self.map_size_cm // (self.resolution * 2)
#         y2 = y1 + self.vision_range
#         agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
#         agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
#         agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
#             agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
#             min=0.0, max=1.0)

#         corrected_pose = pose_obs

#         def get_new_pose_batch(pose, rel_pose_change):

#             pose[:, 1] += rel_pose_change[:, 0] * \
#                 torch.sin(pose[:, 2] / 57.29577951308232) \
#                 + rel_pose_change[:, 1] * \
#                 torch.cos(pose[:, 2] / 57.29577951308232)
#             pose[:, 0] += rel_pose_change[:, 0] * \
#                 torch.cos(pose[:, 2] / 57.29577951308232) \
#                 - rel_pose_change[:, 1] * \
#                 torch.sin(pose[:, 2] / 57.29577951308232)
#             pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

#             pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
#             pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

#             return pose

#         current_poses = get_new_pose_batch(poses_last, corrected_pose)
#         st_pose = current_poses.clone().detach()

#         st_pose[:, :2] = - (st_pose[:, :2]
#                             * 100.0 / self.resolution
#                             - self.map_size_cm // (self.resolution * 2)) /\
#             (self.map_size_cm // (self.resolution * 2))
#         st_pose[:, 2] = 90. - (st_pose[:, 2])

#         rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
#                                       self.device)

#         rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
#         translated = F.grid_sample(rotated, trans_mat, align_corners=True)

#         maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

#         map_pred, _ = torch.max(maps2, 1)

#         return fp_map_pred, map_pred, pose_pred, current_poses

# # Usage:
# # obs:(4+C) x H x W
# sem_mapping = Semantic_Mapping(args)
# rgb = 
# pose_obs = np.concatenate((rgb, depth, sem_seg_pred),
#                                axis=2).transpose(2, 0, 1)
# _,map_pred,_,current_poses = sem_mapping(obs, pose_obs, maps_last, poses_last)




# ####################

# def get_segmentation(image):
#     '''
#     Input: rgb image the camera sees
#     Output: [nc,]
#     point cloud: [N,[x,y,z,d,r,]]
#     '''
#     return segmentation


# #####################
# from ..src.dataset.data_utils import VLNDataLoader

# class VLNSemanticMap(VLNDataLoader):
#     def __init__(self, args):
#         super(VLNSemanticMap, self).__init__(args)

#     def get_semantic_occupation_map(self,camera_list:list, data_types:list):
#         obs = self.get_observations(data_type=data_types)
#         for camera in camera_list:
#             rgb = obs[camera]['rgb']
#             depth = obs[camera]['depth']
#             sem_seg_pred = get_segmentation(rgb)



from .path_planner import QuadTreeNode
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from grutopia.core.util.log import log
import torch.nn.functional as F
from matplotlib.lines import Line2D
from .BEVmap import BEVMap
from .pointcloud import create_pointcloud_from_depth, compute_intrinsic_matrix

import open3d as o3d



class BEVSemMap:
    def __init__(self,args,robot_init_pose):
        # super().__init__(args,robot_init_pose)
        self.args = args

        self.step_time = 0
        quadtree_config = args.maps.quadtree_config
        self.voxel_size = args.maps.voxel_size  # Resolution to present the map
        quadtree_config.width, quadtree_config.height = int(quadtree_config.width/self.voxel_size), int(quadtree_config.height/self.voxel_size)
        self.quadtree_config = quadtree_config
        self.quadtree_width = self.quadtree_config.width
        self.quadtree_height = self.quadtree_config.height
        print(f"quadtree width for semantic map: {self.quadtree_width}, quadtree height: {self.quadtree_height}")

        self.semantic_map = np.zeros((self.quadtree_height, self.quadtree_width, args.segmentation.num_categories))

        self.quad_tree_root = QuadTreeNode(x=0, y=0, width=self.quadtree_width, height=self.quadtree_height,
                                           map_data = self.semantic_map, 
                                           max_depth=self.quadtree_config.max_depth, threshold=self.quadtree_config.threshold)  
        self.init_world_pos = np.array(robot_init_pose)
        self.robot_z = args.maps.robot_z  # The height(m) range of robot

        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")


    
    def get_seg_feat(self,image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_mask = logits.argmax(dim=1)[0]  # Shape: [128, 128]

        # Add batch dimension and channel dimension for the interpolate function
        predicted_mask = predicted_mask.unsqueeze(0).unsqueeze(0).float()  # Shape: [1, 1, 128, 128]

        # Define the target size
        target_size = (224, 224)

        # Resize the mask to the original input size
        resized_mask = F.interpolate(predicted_mask, size=target_size, mode='nearest')  # Mode 'nearest' for categorical data
        resized_mask = resized_mask.squeeze().long()  # Shape: [224, 224]


        return resized_mask


    def depth_to_world_xy(self, depth_map, cameraProjection, cameraViewTransform):
        np.save('vln/semantic_map/depth_map.npy', depth_map)
        np.save('vln/semantic_map/cameraProjection.npy', cameraProjection)
        np.save('vln/semantic_map/cameraViewTransform.npy', cameraViewTransform)

        cameraProjection_inverse = np.linalg.inv(cameraProjection)
        cameraViewTransform_inverse = np.linalg.inv(cameraViewTransform)

        height, width = depth_map.shape
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

        u_ndc = (2.0 * u_coords / width) - 1.0
        v_ndc = 1.0 - (2.0 * v_coords / height)

        u_ndc_flat = u_ndc.flatten()
        v_ndc_flat = v_ndc.flatten()
        z_camera_flat = depth_map.flatten()

        ndc_points = np.stack([u_ndc_flat, v_ndc_flat, np.ones_like(z_camera_flat), -z_camera_flat], axis=1)

        camera_points_homogeneous = np.dot(cameraProjection_inverse, ndc_points.T).T
        camera_points = camera_points_homogeneous[:, :3] / camera_points_homogeneous[:, 3:4]
        camera_points_homogeneous = np.column_stack([camera_points, np.ones_like(z_camera_flat)])

        world_points_homogeneous = np.dot(cameraViewTransform_inverse, camera_points_homogeneous.T).T

        X_world = world_points_homogeneous[:, 0].reshape(height, width)
        Y_world = world_points_homogeneous[:, 1].reshape(height, width)
        
        return X_world, Y_world

    def convert_world_to_map(self, point_cloud):
        # Note that the pointclouds have the world corrdinates that some values are very negative
        # We need to convert it into the map coordinates
        if len(point_cloud)==0:
            log.error(f"The shape of point cloud is not correct. The shape is {point_cloud.shape}.")
            return None
        point_cloud[...,:2] = point_cloud[..., :2] - self.init_world_pos[:2]

        return point_cloud
    
    def framewise_update_semantic_map(self, depth_map, focal_length, aperture, position, quaternion, semantic_segmentation,robot_bottom_z):
        """
        Update the semantic map with new depth and semantic segmentation data.
        
        :param depth_image: 2D numpy array of depth values
        :param camera_pose: 4x4 transformation matrix (camera to world)
        :param camera_matrix: 3x3 camera intrinsic matrix
        :param dist_coeffs: Distortion coefficients
        :param semantic_segmentation: 2D numpy array of semantic segmentation labels
        """
        # Get world coordinates from depth image
        # cameraProjection = cameraProjection.reshape(4,4)
        # cameraViewTransform = cameraViewTransform.reshape(4,4)

        intrinsic_matrix = compute_intrinsic_matrix(focal_length=focal_length,aperture=aperture,image_shape=depth_map.shape)
        points = create_pointcloud_from_depth(intrinsic_matrix=intrinsic_matrix, depth=depth_map, position=position, orientation=quaternion,keep_invalid=False)
        if isinstance(points, list):
            pos_point_cloud = [self.convert_world_to_map(p) for p in points]
            pos_point_cloud = [p for p in pos_point_cloud if p is not None]
            pos_point_cloud = np.vstack(pos_point_cloud)
        else:
            pos_point_cloud = self.convert_world_to_map(points)

        adjusted_coords = (pos_point_cloud[:, :2]/self.voxel_size + [self.quadtree_width/2, self.quadtree_height/2]).astype(int) 
        adjusted_coords_with_z = np.hstack((adjusted_coords, pos_point_cloud[:,2].reshape(-1,1)))
        X_world, Y_world = adjusted_coords[:,0],adjusted_coords[:,1]
        
        # self.visualize_pc(adjusted_coords)
        # Get semantic labels
        semantic_labels = semantic_segmentation.flatten()

        # Get image dimensions
        height, width = depth_map.shape

        # # Reshape world coordinates
        # X_world = X_world.flatten()
        # Y_world = Y_world.flatten()
        if len(X_world) != len(semantic_labels):
            raise ValueError("X_world 和 semantic_labels 的长度不一致")

        # Get valid indices
        point_to_consider = np.where((X_world >= 0) & (X_world < self.quadtree_height) & (Y_world >= 0) & (Y_world <  self.quadtree_width) & adjusted_coords_with_z[:,2]>=(robot_bottom_z+self.robot_z[0] & adjusted_coords_with_z[:,2]<(robot_bottom_z+self.robot_z[1])) )[0]
        # TODO: check why the range are different
        point_within_robot_z = point_to_consider[(point_to_consider[:,2]>=(robot_bottom_z+self.robot_z[0])) & (point_to_consider[:,2]<=(robot_bottom_z+self.robot_z[1]))].astype(int)

        # Update semantic map
        for i in point_to_consider:
            x = int(X_world[i])
            y = int(Y_world[i])
            category = int(semantic_labels[i])
            self.semantic_map[x, y, category] = 1
            # TODO: check whether creating  a QuadTreeNode is necessary
        return adjusted_coords_with_z
    
    def update_semantic_map(self,obs_tr,camera_poses,camera_dict:dict,robot_bottom_z,verbose=False,global_bev=False):
        # single robot
        for camera in camera_dict:
            cur_obs = obs_tr[camera]
            rgb_obs = cur_obs['rgba'][...,:3]
            depth_obs = cur_obs['depth']
            max_depth = 10
            depth_obs[depth_obs > max_depth] = 0
            camera_params = cur_obs['camera_params']
            semantic_segmentation = self.get_seg_feat(rgb_obs)
            camera_pose = camera_poses[camera]
            camera_position, camera_orientation = camera_pose[0], camera_pose[1]


            pc=self.framewise_update_semantic_map(depth_map=depth_obs, focal_length=camera_params['cameraFocalLength'], aperture=camera_params['cameraAperture'], semantic_segmentation=semantic_segmentation,position=camera_position,quaternion=camera_orientation)
        
        if verbose:
            if global_bev:
                img_save_path = os.path.join(self.args.log_image_dir, "semantic_"+str(self.step_time)+".jpg")
            else:
                img_save_path = os.path.join(self.args.log_image_dir, "semantic_"+str(self.step_time)+".jpg")

            # draw the robot's position using the red 'x' mark
            # self.plot_semantic_map(img_save_path)
            # log.info(f"Semantic map saved at {img_save_path}")
            # img_save_path = os.path.join(self.args.log_image_dir, "segformer_"+str(self.step_time)+".jpg")
            # self.plot_segmentation_result(rgb_obs, semantic_segmentation, img_save_path)
            self.plot_rgb_segmentation_semantic_pc(depth_obs,rgb_obs, semantic_segmentation, self.semantic_map, pc,img_save_path,robot_bottom_z)
        return True
        
    def visualize_world_points(self,world_points_homogeneous):
        # 提取 x, y, z 坐标
        x = world_points_homogeneous[:, 0]
        y = world_points_homogeneous[:, 1]
        z = world_points_homogeneous[:, 2]

        # 过滤 y 值比 400 小的点
        mask = y <100
        x_filtered = x[mask]
        y_filtered = y[mask]
        z_filtered = z[mask]

        # 创建一个新的图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制散点图
        scatter = ax.scatter(x_filtered, y_filtered, z_filtered, c=z_filtered, cmap='viridis', s=1)

        # 设置轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 添加颜色条
        plt.colorbar(scatter)

        # 设置标题
        plt.title('World Points Visualization')

        # 显示图形
        plt.savefig('vln/semantic_map/world_points.png')


    def plot_semantic_map(self, img_save_path):
        '''
        Visualize and save the semantic map at img_save_path. Draw the map for each channel ([H,W]) with different colors and transparency 0.3 (so that overlapping is visible) on the same map with different color. Draw the caption of each channel referring to each color
        Input: semantic_map: [H,W,Channel]
        '''
        H, W, C = self.semantic_map.shape
    
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Define a colormap with 40 colors
        num_colors = 40
        colormap = plt.get_cmap('tab20c', num_colors)  # You can use other colormaps like 'tab20', 'viridis', etc.

        # Track which channels have non-zero values
        valid_channels = []

        # Plot each channel
        for i in range(C):
            channel = self.semantic_map[:, :, i]
            if np.any(channel > 0):
                valid_channels.append(i)
                color = colormap(i % num_colors)  # Use color mapping for up to 40 colors
                color_rgba = list(color)
                color_rgba[3] = 1  # Set alpha to 0.3 for transparency
                ax.imshow(np.ma.masked_where(channel == 0, channel), cmap=plt.cm.colors.ListedColormap([color_rgba]), interpolation='nearest')

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a legend for valid channels only
        legend_elements = []
        for i in valid_channels:
            color = colormap(i % num_colors)
            color_rgba = list(color)
            color_rgba[3] = 0.3
            legend_elements.append(Line2D([0], [0], color=color_rgba, lw=4, alpha=0.3, label=f'Channel {i+1}'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
        
        # Set the background color to white
        fig.patch.set_facecolor('white')
        
        # Adjust the plot to remove extra whitespace
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
        plt.close()

        
    def plot_segmentation_result(self, rgb, semantic_segmentation, save_path):
        '''
        Visualize and save the RGB image and semantic segmentation map side by side.
        Args:
            rgb: The RGB image array of shape [H, W, 3].
            semantic_segmentation: The semantic segmentation map array of shape [H, W].
            save_path: Path to save the resulting figure.
        '''
        # Create a figure and axis
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot RGB image
        ax[0].imshow(rgb)
        ax[0].set_title("RGB Image")
        ax[0].axis('off')  # Hide axes for RGB image
        
        # Define a colormap with 40 colors
        num_colors = 40
        colormap = plt.get_cmap('tab20c', num_colors)  # or any other colormap suitable for your number of classes

        # Ensure semantic_segmentation is integer type
        if semantic_segmentation.dtype != np.uint8:
            semantic_segmentation = semantic_segmentation.astype(np.uint8)

        # Plot semantic segmentation with color map
        ax[1].imshow(semantic_segmentation, cmap=colormap, interpolation='nearest')
        ax[1].set_title("Semantic Segmentation")
        ax[1].axis('off')  # Hide axes for segmentation map
        
        # Add a legend to the semantic segmentation plot
        # Only add legend entries for non-empty classes
        num_classes = np.max(semantic_segmentation) + 1
        valid_classes = np.unique(semantic_segmentation)
        
        legend_elements = []
        for i in valid_classes:
            color = colormap(i % num_colors)
            color_rgba = list(color)
            color_rgba[3] = 0.7  # Adjust alpha for legend visibility
            legend_elements.append(Line2D([0], [0], color=color_rgba, lw=4, label=f'Class {i}'))
        
        if legend_elements:
            ax[1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        import matplotlib.pyplot as plt


    
    def plot_rgb_segmentation_semantic_pc(self, depth, rgb, semantic_segmentation, semantic_map, pc, robot_coords, save_path):
        '''
        Visualize and save the RGB image, depth image, semantic segmentation map, semantic map, and point cloud side by side.
        
        Args:
            depth: The depth image array of shape [H, W].
            rgb: The RGB image array of shape [H, W, 3].
            semantic_segmentation: The semantic segmentation map array of shape [H, W].
            semantic_map: The semantic map array of shape [H, W, Channel] with binary values.
            pc: The point cloud data as a numpy array of shape [N, 3].
            save_path: Path to save the resulting figure.
        '''
        # 创建一个5个子图的图形和坐标轴 
        fig, ax = plt.subplots(1, 5, figsize=(30, 6))
        
        # 绘制RGB图像
        ax[0].imshow(rgb)
        ax[0].set_title("RGB Image")
        ax[0].axis('off')
        
        # 绘制深度图像
        ax[1].imshow(depth, cmap='gray')
        ax[1].set_title("Depth Image")
        ax[1].axis('off')
        
        # 定义一个具有40种颜色的colormap，以涵盖最多160类
        num_colors = 40
        colormap = plt.get_cmap('tab20c', num_colors)
        
        # 使用colormap绘制语义分割图像
        ax[2].imshow(semantic_segmentation, cmap=colormap, interpolation='nearest')
        ax[2].set_title("Semantic Segmentation")
        ax[2].axis('off')
        
        # 绘制语义图
        white_background = np.ones_like(semantic_map[:, :, 0])  # 创建一个白色图像
        ax[3].imshow(white_background, cmap='gray', vmin=0, vmax=1)
        ax[3].set_title("Semantic Map")
        ax[3].axis('off')
        
        valid_labels = []
        # 叠加每个channel的语义图
        for i in range(semantic_map.shape[2]):
            channel = semantic_map[:, :, i]
            if np.any(channel == 1):  # 检查该channel是否存在
                color = colormap(i % num_colors)
                color_rgba = list(color)
                color_rgba[3] = 1.0  # 设置alpha为1.0，完全不透明
                ax[3].imshow(np.ma.masked_where(channel == 0, channel), cmap=ListedColormap([color_rgba]), interpolation='nearest')
                valid_labels.append(i)
        
        # 绘制点云图像
        ax[4] = fig.add_subplot(1, 5, 5, projection='3d')
        ax[4].scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 2], cmap='viridis', s=1)
        ax[4].set_title("Point Cloud")
        ax[4].set_xlabel('X')
        ax[4].set_ylabel('Y')
        ax[4].set_zlabel('Z')
        convert_robot_coords = ((robot_coords[:2]- self.init_world_pos[:2])/self.voxel_size + [self.quadtree_width/2, self.quadtree_height/2])
        ax[4].scatter(convert_robot_coords[0], convert_robot_coords[1], color='blue', marker='o', label="current position: (%.2f, %.2f)"%(convert_robot_coords[0], convert_robot_coords[1]))
        ax[4].axis('off')

        # 保存最终的图形
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()