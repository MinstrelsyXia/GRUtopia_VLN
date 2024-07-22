import os
import open3d as o3d
import numpy as np

from grutopia.core.util.log import log


# Function to transform a point cloud given a transformation matrix
def transform_point_cloud(point_cloud, transformation):
    pc = np.asarray(point_cloud.points)
    ones = np.ones((pc.shape[0], 1))
    pc_homogeneous = np.hstack((pc, ones))
    transformed_pc = transformation @ pc_homogeneous.T
    transformed_pc = transformed_pc.T[:, :3]
    point_cloud.points = o3d.utility.Vector3dVector(transformed_pc)
    return point_cloud

# Function to get transformation matrix from position and orientation (e.g., Euler angles)
def get_transformation_matrix(position, orientation):
    # Assuming orientation is given as (roll, pitch, yaw) in radians
    roll, pitch, yaw = orientation
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
    combined_pcd = transformed_point_clouds[0]
    for pcd in transformed_point_clouds[1:]:
        combined_pcd += pcd

    if draw:
        # Visualize the combined point cloud
        o3d.visualization.draw_geometries([combined_pcd])
        # Save the picture
        o3d.io.write_image(os.path.join(log_dir, "pano_pointcloud.png"), combined_pcd)
        log.info("Saved the combined point cloud to %s" % os.path.join(log_dir, "pano_pointcloud"))

    return combined_pcd

# Example positions and orientations for 3 cameras
# positions = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
# orientations = [(0, 0, 0), (0, 0, np.pi/4), (0, np.pi/4, 0)]

