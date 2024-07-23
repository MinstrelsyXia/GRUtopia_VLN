import os
import open3d as o3d
import numpy as np
import cv2
        
from grutopia.core.util.log import log
from ..utils.utils import euler_angles_to_quat, quat_to_euler_angles, compute_rel_orientations


# Function to transform a point cloud given a transformation matrix
def transform_point_cloud(point_cloud, transformation):
    pc = np.asarray(point_cloud)
    ones = np.ones((pc.shape[0], 1))
    pc_homogeneous = np.hstack((pc, ones))
    transformed_pc = transformation @ pc_homogeneous.T
    transformed_pc = transformed_pc.T[:, :3]
    point_cloud = o3d.utility.Vector3dVector(transformed_pc)
    return point_cloud

# Function to get transformation matrix from position and orientation (e.g., Euler angles)
def get_transformation_matrix(position, orientation):
    # Assuming orientation is given as (roll, pitch, yaw) in radians
    roll, pitch, yaw = quat_to_euler_angles(orientation)
    c, s = np.cos, np.sin
    Rx = np.array([[1, 0, 0],
                   [0, c(roll), -s(roll)],
                   [0, s(roll), c(roll)]])
    Ry = np.array([[c(pitch), 0, s(pitch)],
                   [0, 1, 0],
                   [-s(pitch), 0, c(pitch)]])
    Rz = np.array([[c(yaw), -s(yaw), 0],
                   [s(yaw), c(yaw), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.array(position).reshape((3, 1))
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = T.flatten()
    return transformation

def generate_pano_pointcloud(camera_positions, camera_orientations, pointclouds, draw=False, log_dir="logs/images/"):
    # Transform point clouds to the world coordinate system
    transformed_point_clouds = []
    for i, pcd in enumerate(pointclouds):
        transformation = get_transformation_matrix(camera_positions[i], camera_orientations[i])
        transformed_pcd = transform_point_cloud(pcd, transformation)
        transformed_point_clouds.append(transformed_pcd)

    # Combine point clouds
    combined_points = np.vstack([np.asarray(pcd) for pcd in transformed_point_clouds])
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)

    if draw:
        # Save the visualization as an image
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)  # Create a window, but don't show it
        vis.add_geometry(combined_pcd)
        vis.update_geometry(combined_pcd)
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        # Convert float buffer to uint8 image
        image_np = np.asarray(image) * 255
        image_np = image_np.astype(np.uint8)

        # Save image using OpenCV
        import cv2
        cv2.imwrite("/home/pjlab/w61/GRUtopia/logs/images/pano_pointcloud.png", image_np[..., ::-1])  # Convert RGB to BGR for OpenCV
        log.info("Saved the combined point cloud to " % os.path.join(log_dir, "pano_pointcloud.png"))

    return combined_pcd

def generate_pano_pointcloud_local(pointclouds, draw=False, log_dir="logs/images/"):
    ''' supposed that pointclouds have been converted to local coordinate system
    '''
    # Transform point clouds to the world coordinate system
    transformed_point_clouds = []

    # Combine point clouds
    combined_points = np.vstack([np.asarray(pcd) for pcd in pointclouds])

    
    # Filter points that are farther than a certain threshold distance (e.g., 2.0 units)
    threshold_distance = 2.0
    distances = np.linalg.norm(combined_points, axis=1)
    combined_points = combined_points[distances <= threshold_distance]
    
    
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    
    # Visualize the filtered point cloud
    # o3d.visualization.draw_geometries([combined_pcd])

    if draw:
        # Save the visualization as an image
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)  # Create a window, but don't show it
        vis.add_geometry(combined_pcd)
        vis.update_geometry(combined_pcd)
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        # Convert float buffer to uint8 image
        image_np = np.asarray(image) * 255
        image_np = image_np.astype(np.uint8)

        # Save image using OpenCV

        cv2.imwrite("/home/pjlab/w61/GRUtopia/logs/images/pano_pointcloud.png", image_np[..., ::-1])  # Convert RGB to BGR for OpenCV
        log.info("Saved the combined point cloud to %s" % os.path.join(log_dir, "pano_pointcloud.png"))

    return combined_pcd

def pc_to_local_pose(cur_obs):
    pc_data = cur_obs['pointcloud']['data']
    cam_transform_row_major = np.array(cur_obs['camera_params']['cameraViewTransform']).reshape(4,4)
    
    # Convert world_pose to local_pose
    
    # Homogenize the point cloud data (x, y, z) -> (x, y, z, 1) for multiplication with the camera transform
    pc_homogenized = np.hstack((pc_data, np.ones((pc_data.shape[0], 1))))
    # Transform to camera frame (no need to transpose the transformation matrix since it is column-major)
    pc_camera_frame = pc_homogenized @ cam_transform_row_major
    # De-homogenize the point cloud data (x, y, z, 1) -> (x, y, z)
    pc_camera_frame = pc_camera_frame[:, :3] 
    
    return pc_camera_frame

# Example positions and orientations for 3 cameras
# positions = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
# orientations = [(0, 0, 0), (0, 0, np.pi/4), (0, np.pi/4, 0)]

