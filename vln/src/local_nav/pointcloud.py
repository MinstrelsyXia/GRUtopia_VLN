import os
import open3d as o3d
import numpy as np
import cv2
        
from grutopia.core.util.log import log
# from ..utils.utils import euler_angles_to_quat, quat_to_euler_angles, compute_rel_orientations


# Function to transform a point cloud given a transformation matrix
def transform_point_cloud(point_cloud, transformation):
    pc = np.asarray(point_cloud)
    ones = np.ones((pc.shape[0], 1))
    pc_homogeneous = np.hstack((pc, ones))
    transformed_pc = transformation @ pc_homogeneous.T
    transformed_pc = transformed_pc.T[:, :3]
    point_cloud = o3d.utility.Vector3dVector(transformed_pc)
    return point_cloud

def downsample_pc(pc, depth_sample_rate):
    '''
    INput: points:(N,3); rate:downsample rate:int
    Output: downsampled_points:(N/rate,3)
    '''
    shuffle_mask = np.arange(pc.shape[0])
    np.random.shuffle(shuffle_mask)
    shuffle_mask = shuffle_mask[::depth_sample_rate]
    pc = pc[shuffle_mask,:]
    return pc

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



import torch
import warp as wp
from collections.abc import Sequence
from omni.isaac.lab.utils.array import TensorData, convert_to_torch
import omni.isaac.lab.utils.math as math_utils
import math

def create_pointcloud_from_depth(
    intrinsic_matrix: np.ndarray | torch.Tensor | wp.array,
    depth: np.ndarray | torch.Tensor | wp.array,
    keep_invalid: bool = False,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray | torch.Tensor:
    r"""Creates pointcloud from input depth image and camera intrinsic matrix.

    This function creates a pointcloud from a depth image and camera intrinsic matrix. The pointcloud is
    computed using the following equation:

    .. math::
        p_{camera} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`K` is the camera intrinsic matrix, :math:`u` and :math:`v` are the pixel coordinates and
    :math:`d` is the depth value at the pixel.

    Additionally, the pointcloud can be transformed from the camera frame to a target frame by providing
    the position ``t`` and orientation ``R`` of the camera in the target frame:

    .. math::
        p_{target} = R_{target} \times p_{camera} + t_{target}

    Args:
        intrinsic_matrix: A (3, 3) array providing camera's calibration matrix.
        depth: An array of shape (H, W) with values encoding the depth measurement.
        keep_invalid: Whether to keep invalid points in the cloud or not. Invalid points
            correspond to pixels with depth values 0.0 or NaN. Defaults to False.
        position: The position of the camera in a target frame. Defaults to None.
        orientation: The orientation (w, x, y, z) of the camera in a target frame. Defaults to None.
        device: The device for torch where the computation should be executed.
            Defaults to None, i.e. takes the device that matches the depth image.

    Returns:
        An array/tensor of shape (N, 3) comprising of 3D coordinates of points.
        The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
        is returned.
    """
    # We use PyTorch here for matrix multiplication since it is compiled with Intel MKL while numpy
    # by default uses OpenBLAS. With PyTorch (CPU), we could process a depth image of size (480, 640)
    # in 0.0051 secs, while with numpy it took 0.0292 secs.

    # convert to numpy matrix
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    depth = convert_to_torch(depth, dtype=torch.float32, device=device)
    # update the device with the device of the depth image
    # note: this is needed since warp does not provide the device directly
    device = depth.device
    # convert inputs to torch tensors
    intrinsic_matrix = convert_to_torch(intrinsic_matrix, dtype=torch.float32, device=device)
    if position is not None:
        position = convert_to_torch(position, dtype=torch.float32, device=device)
    if orientation is not None:
        orientation = convert_to_torch(orientation, dtype=torch.float32, device=device)
    # compute pointcloud
    depth_cloud = math_utils.unproject_depth(depth, intrinsic_matrix)
    # convert 3D points to world frame
    depth_cloud = math_utils.transform_points(depth_cloud, position, orientation)

    # keep only valid entries if flag is set
    if not keep_invalid:
        pts_idx_to_keep = torch.all(torch.logical_and(~torch.isnan(depth_cloud), ~torch.isinf(depth_cloud)), dim=1)
        depth_cloud = depth_cloud[pts_idx_to_keep, ...]

    # return everything according to input type
    if is_numpy:
        return depth_cloud.detach().cpu().numpy()
    else:
        return depth_cloud
    
def compute_intrinsic_matrix(aperture, focal_length, image_shape):
    """Compute camera's matrix of intrinsic parameters.

    This matrix works for linear depth images. We consider both horizontal and vertical apertures.

    Parameters:
    aperture (array-like): Array containing horizontal and vertical aperture values in mm
    focal_length (float): Focal length of the camera in mm
    image_shape (tuple): Shape of the image (height, width)

    Returns:
    numpy.ndarray: 3x3 camera intrinsic matrix
    """
    intrinsic_matrix = np.zeros((3, 3))
    height, width = image_shape
    horiz_aperture, vert_aperture = aperture

    # Calculate horizontal and vertical field of view
    fov_h = 2 * math.atan(horiz_aperture / (2 * focal_length))
    fov_v = 2 * math.atan(vert_aperture / (2 * focal_length))

    # Calculate focal length in pixels for both directions
    focal_px_x = width * 0.5 / math.tan(fov_h / 2)
    focal_px_y = height * 0.5 / math.tan(fov_v / 2)
    
    # Create intrinsic matrix
    intrinsic_matrix[0, 0] = focal_px_x
    intrinsic_matrix[1, 1] = focal_px_y
    intrinsic_matrix[0, 2] = width * 0.5
    intrinsic_matrix[1, 2] = height * 0.5
    intrinsic_matrix[2, 2] = 1

    return intrinsic_matrix

