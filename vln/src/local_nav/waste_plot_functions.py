

############################## from class BevSemMap ########################################
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


############################### also from class BevSemMap ########################################
############################### cloud point test functions ########################################

def framewise_update_semantic_map(depth_map, position, quaternion):
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

        # intrinsic_matrix = compute_intrinsic_matrix(focal_length=focal_length,aperture=aperture,image_shape=depth_map.shape)
        intrinsic_matrix = np.array([221,0,128,0,221,128,0,0,1] ).reshape([3,3])
        pos_point_cloud = create_pointcloud_from_depth(intrinsic_matrix=intrinsic_matrix, depth=depth_map, position=position, orientation=quaternion,keep_invalid=False)
        shuffle_mask = np.arange(pos_point_cloud.shape[0])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::50]
        pc = pos_point_cloud[shuffle_mask,:]
        return pc

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

    # intrinsic_matrix = compute_intrinsic_matrix(focal_length=focal_length,aperture=aperture,image_shape=depth_map.shape)
    intrinsic_matrix = np.array([221,0,128,0,221,128,0,0,1] ).reshape([3,3])
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
    
def preprocess_pose_iter(camera_pose,position_neg_iter,position_axis_iter,ori_iter=[0,1,2,3]):
    '''
    match poses: 
        Input: (x,y,z,pw,px,py,pz): x right, y up, z backward
        Output: (x,y,z,px,py,pz,pw); x right, y down z forward

    '''
    position = camera_pose[:, :3]
    orientation = camera_pose[:, 3:]
    
    # Apply the transformations directly
    vlmap_pos = position * position_neg_iter
    vlmap_pos = vlmap_pos[:, position_axis_iter] # 0,1,2; 1,2,0; 1,0,2;0,2,1;2,0,1;
    vlmap_ori = orientation[:, ori_iter]
    return np.hstack((vlmap_pos, vlmap_ori))

if __name__ == "__main__":
    
    depth_map1 = np.load("/ssd/xiaxinyuan/code/w61-grutopia/logs/sample_episodes/s8pcmisQ38h/id_2606/depth/pano_camera_0_depth_step_8.npy")
    depth_map2 = np.load("/ssd/xiaxinyuan/code/w61-grutopia/logs/sample_episodes/s8pcmisQ38h/id_2606/depth/pano_camera_0_depth_step_11.npy")
    focal_length = 18
    aperture = 20
    pose = np.loadtxt("/ssd/xiaxinyuan/code/w61-grutopia/logs/sample_episodes/s8pcmisQ38h/id_2606/poses.txt")
    camera_pose_tf_iter=np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]])
    camera_pose_axis_iter = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]])
    for i,c_pose_tf in enumerate(camera_pose_tf_iter):
        for j,c_pose_axis in enumerate(camera_pose_axis_iter):
            pose = preprocess_pose_iter(pose,c_pose_tf,c_pose_axis)
            position = pose[0, :3]
            quaternion = pose[0, 3:]
            position = np.array([position[0],position[2],position[1]])
            quaternion = np.array([-quaternion[3], quaternion[0], quaternion[2], -quaternion[1]])

            pc1 = framewise_update_semantic_map(depth_map1,position, quaternion)

            position = pose[1, :3]
            quaternion = pose[1, 3:]
            position = np.array([position[0],position[2],position[1]])
            quaternion = np.array([-quaternion[3], quaternion[0], quaternion[2], -quaternion[1]])
            pc2 = framewise_update_semantic_map(depth_map2,position, quaternion)

            pc = np.vstack((pc1, pc2))
            pcd_global = o3d.geometry.PointCloud()
            pcd_global.points = o3d.utility.Vector3dVector(pc)
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1.0, origin=[0, 0, 0]
)
            o3d.visualization.draw_geometries([pcd_global,coordinate_frame])


################################## also from class BevSemMap ############################
################################## Segformer Version ####################################
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
# ! not used
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

    # ! not used
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



############################### from data util, inverse transform to obtain extrinsic matrix #################################
def depth2pc_real_world(depth, cam_mat):
    """
    Return 3xN array
    """

    h, w = depth.shape
    cam_mat_inv = np.linalg.inv(cam_mat)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # x = x[int(h/2)].reshape((1, -1))
    # y = y[int(h/2)].reshape((1, -1))
    # z = depth[int(h/2)].reshape((1, -1))

    x = x.reshape((1, -1))[:, :]
    y = y.reshape((1, -1))[:, :]
    z = depth.reshape((1, -1))[:, :]

    p_2d = np.vstack([x, y, np.ones_like(x)])
    pc = cam_mat_inv @ p_2d
    pc = pc * z
    mask_1 = pc[2, :] > 0.1
    # mask = mask_1
    mask_2 = pc[2, :] < 4
    mask = np.logical_and(mask_1, mask_2)
    # pc = pc[:, mask]
    return pc, mask

def compute_transformation_matrix(PC, PW):
    """
    计算变换矩阵 E，使得 PC = E @ PW
    :param PC: 目标点云，形状为 (3, N)
    :param PW: 原始点云，形状为 (3, N)
    :return: 变换矩阵 E，形状为 (3, 3)
    """
    # 确保输入是 NumPy 数组
    PC = np.asarray(PC)
    PW = np.asarray(PW)
    
    # 检查输入形状
    assert PC.shape[0] == 3 and PW.shape[0] == 3, "PC 和 PW 必须是形状为 (3, N) 的矩阵"
    
    # 转置矩阵以适应 np.linalg.lstsq 的输入要求
    PW_T = PW.T  # 形状为 (N, 3)
    PC_T = PC.T  # 形状为 (N, 3)
    
    # 使用最小二乘法求解 E
    E, residuals, rank, s = np.linalg.lstsq(PW_T, PC_T, rcond=None)
    
    # 转置 E 以获得正确的形状 (3, 3)
    E = E.T
    
    return E
def compute_transformation_matrix(PC, PW):
    """
    计算变换矩阵 E，使得 PC = E @ PW
    :param PC: 目标点云，形状为 (3, N)
    :param PW: 原始点云，形状为 (3, N)
    :return: 变换矩阵 E，形状为 (3, 3)
    """
    # 确保输入是 NumPy 数组
    PC = np.asarray(PC)
    PW = np.asarray(PW)
    
    # 检查输入形状
    assert PC.shape[0] == 3 and PW.shape[0] == 3, "PC 和 PW 必须是形状为 (3, N) 的矩阵"
    
    # 转置矩阵以适应 np.linalg.lstsq 的输入要求
    PW_T = PW.T  # 形状为 (N, 3)
    PC_T = PC.T  # 形状为 (N, 3)
    
    # 使用最小二乘法求解 E
    E, residuals, rank, s = np.linalg.lstsq(PW_T, PC_T, rcond=None)
    
    # 转置 E 以获得正确的形状 (3, 3)
    E = E.T
    
    return E