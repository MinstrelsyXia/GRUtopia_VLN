import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_world_points(world_points_homogeneous):
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

# 示例使用
# 假设 world_points_homogeneous 是你的齐次坐标点云数据

import plotly.graph_objects as go

def visualize_world_points_move(world_points_homogeneous):
    # 提取 x, y, z 坐标
    x = world_points_homogeneous[:, 0]
    y = world_points_homogeneous[:, 1]
    z = world_points_homogeneous[:, 2]

    # 创建3D散点图
    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=z,  # 设置颜色为z值
            colorscale='Viridis',  # 颜色映射
            opacity=0.8
        )
    )

    # 设置布局
    layout = go.Layout(
        title='World Points Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    # 创建图形
    fig = go.Figure(data=[scatter], layout=layout)

    # 显示图形
    fig.show()

import matplotlib.pyplot as plt

def visualize_depth_map(depth_map):
    # 创建一个新的图形
    plt.figure(figsize=(10, 8))

    # 绘制深度图
    plt.imshow(depth_map, cmap='viridis')
    
    # 添加颜色条
    plt.colorbar()

    # 设置标题
    plt.title('Depth Map Visualization')

    # 显示图形
    plt.savefig('vln/semantic_map/depth_map.png')

# def depth_to_world_xy(depth_map, cameraProjection, cameraViewTransform):
#     cameraProjection_inverse = np.linalg.inv(cameraProjection)
#     cameraViewTransform_inverse = np.linalg.inv(cameraViewTransform.T)

#     height, width = depth_map.shape
#     u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

#     # Adjust NDC calculation for left-handed coordinate system
#     u_ndc = (2.0 * u_coords / width) - 1.0
#     v_ndc = 1.0 - (2.0 * v_coords / height)

#     u_ndc_flat = u_ndc.flatten()
#     v_ndc_flat = v_ndc.flatten()
#     z_camera_flat = depth_map.flatten()
#     z_camera_flat[z_camera_flat > 3] = 3
#     # Create NDC points
#     # ndc_points = np.stack([u_ndc_flat, v_ndc_flat, np.ones_like(z_camera_flat), np.ones_like(z_camera_flat)], axis=1)

#     # # Transform to camera space
#     # camera_points = np.dot(cameraProjection_inverse, ndc_points.T).T
    
#     # # Apply depth
#     # camera_points[:, 0] *= z_camera_flat
#     # camera_points[:, 1] *= z_camera_flat
#     # camera_points[:, 2] *= -z_camera_flat  # Negative because of left-handed system
    
#     # # Transform to world space
#     # world_points = np.dot(cameraViewTransform_inverse, camera_points.T).T
#     # world_points = world_points[:, :3] / world_points[:, 3:]

#     # X_world = world_points[:, 0].reshape(height, width)
#     # Y_world = world_points[:, 1].reshape(height, width)
#     # Z_world = world_points[:, 2].reshape(height, width)

#     # visualize_world_points(world_points)
    
#     # return X_world, Y_world
#     ndc_points = np.stack([u_ndc_flat, v_ndc_flat, np.ones_like(z_camera_flat), -z_camera_flat], axis=1)

#     camera_points_homogeneous = np.dot(cameraProjection_inverse, ndc_points.T).T
#     camera_points = camera_points_homogeneous[:, :3] / camera_points_homogeneous[:, 3:4]
#     camera_points_homogeneous = np.column_stack([camera_points, np.ones_like(z_camera_flat)])

#     world_points_homogeneous = np.dot(cameraViewTransform_inverse, camera_points_homogeneous.T).T

#     X_world = world_points_homogeneous[:, 0].reshape(height, width)
#     Y_world = world_points_homogeneous[:, 1].reshape(height, width)
#     visualize_depth_map(depth_map)
#     visualize_world_points(world_points_homogeneous)    
#     return X_world, Y_world


# def depth_to_world_xy( depth_map, cameraProjection, cameraViewTransform):
#     # np.save('vln/semantic_map/depth_map.npy', depth_map)
#     # np.save('vln/semantic_map/cameraProjection.npy', cameraProjection)
#     # np.save('vln/semantic_map/cameraViewTransform.npy', cameraViewTransform)

#     cameraProjection_inverse = np.linalg.inv(cameraProjection)
#     cameraViewTransform_inverse = np.linalg.inv(cameraViewTransform)

#     height, width = depth_map.shape
#     u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

#     u_ndc = (2.0 * u_coords / width) - 1.0
#     v_ndc = 1.0 - (2.0 * v_coords / height)

#     u_ndc_flat = u_ndc.flatten()
#     v_ndc_flat = v_ndc.flatten()
#     z_camera_flat = depth_map.flatten()
#     z_camera_flat[z_camera_flat > 3] = 0
#     ndc_points = np.stack([u_ndc_flat, v_ndc_flat, -z_camera_flat, np.ones_like(z_camera_flat)], axis=1)

#     camera_points_homogeneous = np.dot(cameraProjection_inverse, ndc_points.T).T
#     camera_points = camera_points_homogeneous[:, :3] / camera_points_homogeneous[:, 3:4]
#     camera_points_homogeneous = np.column_stack([camera_points, np.ones_like(z_camera_flat)])

#     world_points_homogeneous = np.dot(cameraViewTransform_inverse, camera_points_homogeneous.T).T

#     X_world = world_points_homogeneous[:, 0].reshape(height, width)
#     Y_world = world_points_homogeneous[:, 1].reshape(height, width)
    
#     visualize_depth_map(depth_map)
#     visualize_world_points(world_points_homogeneous)
#     return X_world, Y_world

import torch
import warp as wp
from collections.abc import Sequence
from omni.isaac.lab.utils.array import TensorData, convert_to_torch
import omni.isaac.lab.utils.math as math_utils

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
depth_map = np.load('vln/semantic_map/depth_map.npy')
cameraProjection = np.load('vln/semantic_map/cameraProjection.npy')


import numpy as np
import math

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


def set_intrinsic_matrices(
        self, matrices: torch.Tensor, focal_length: float = 1.0, env_ids: Sequence[int] | None = None
    ):
        """Set parameters of the USD camera from its intrinsic matrix.

        The intrinsic matrix and focal length are used to set the following parameters to the USD camera:

        - ``focal_length``: The focal length of the camera.
        - ``horizontal_aperture``: The horizontal aperture of the camera.
        - ``vertical_aperture``: The vertical aperture of the camera.
        - ``horizontal_aperture_offset``: The horizontal offset of the camera.
        - ``vertical_aperture_offset``: The vertical offset of the camera.

        .. warning::

            Due to limitations of Omniverse camera, we need to assume that the camera is a spherical lens,
            i.e. has square pixels, and the optical center is centered at the camera eye. If this assumption
            is not true in the input intrinsic matrix, then the camera will not set up correctly.

        Args:
            matrices: The intrinsic matrices for the camera. Shape is (N, 3, 3).
            focal_length: Focal length to use when computing aperture values. Defaults to 1.0.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # convert matrices to numpy tensors
        if isinstance(matrices, torch.Tensor):
            matrices = matrices.cpu().numpy()
        else:
            matrices = np.asarray(matrices, dtype=float)
        # iterate over env_ids
        for i, intrinsic_matrix in zip(env_ids, matrices):
            # extract parameters from matrix
            f_x = intrinsic_matrix[0, 0]
            c_x = intrinsic_matrix[0, 2]
            f_y = intrinsic_matrix[1, 1]
            c_y = intrinsic_matrix[1, 2]
            # get viewport parameters
            height, width = self.image_shape
            height, width = float(height), float(width)
            # resolve parameters for usd camera
            params = {
                "focal_length": focal_length,
                "horizontal_aperture": width * focal_length / f_x,
                "vertical_aperture": height * focal_length / f_y,
                "horizontal_aperture_offset": (c_x - width / 2) / f_x,
                "vertical_aperture_offset": (c_y - height / 2) / f_y,
            }
            # change data for corresponding camera index
            sensor_prim = self._sensor_prims[i]
            # set parameters for camera
            for param_name, param_value in params.items():
                # convert to camel case (CC)
                param_name = to_camel_case(param_name, to="CC")
                # get attribute from the class
                param_attr = getattr(sensor_prim, f"Get{param_name}Attr")
                # set value
                # note: We have to do it this way because the camera might be on a different
                #   layer (default cameras are on session layer), and this is the simplest
                #   way to set the property on the right layer.
                omni.usd.set_prop_val(param_attr(), param_value)
        # update the internal buffers
        self._update_intrinsic_matrices(env_ids)

import numpy as np

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])
    return R

def compute_camera_view_transform(quaternion, position):
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(quaternion)
    
    # Invert the Z-axis for forward direction
    R[:, 2] = -R[:, 2]  # Flip the Z axis to match the forward as -Z convention

    # Camera position
    T = np.array(position)
    
    # Compute the view matrix
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R.T  # Transpose of rotation matrix
    
    # Adjust translation with the flipped Z axis in rotation matrix
    view_matrix[:3, 3] = -R.T @ T
    
    return view_matrix

# # Example quaternion and position
# quaternion = [0.14632428, 0.21607629, 0.02670759, 0.96498028]  # Example values
# position = [5.9468007, -28.225435, 1.3916627]  # Example values

# # cameraViewTransform = compute_camera_view_transform(quaternion, position)
# # print(cameraViewTransform)

# cameraViewTransform = np.load('vln/semantic_map/cameraViewTransform.npy')
# aperture = np.array([20.955, 15.2908])
# focal_length = 18.147560119628906
# # X_world, Y_world = depth_to_world_xy(depth_map, cameraProjection, cameraViewTransform)

# intrinsic_matrix = compute_intrinsic_matrix(focal_length=focal_length,aperture=aperture,image_shape=depth_map.shape)
# print(intrinsic_matrix)
# points = create_pointcloud_from_depth(intrinsic_matrix=intrinsic_matrix, depth=depth_map, position=position, orientation=quaternion,keep_invalid=False)
# print(points.shape)
# # points = np.reshape(points, (depth_map.shape[0],depth_map.shape[1],3))
# visualize_world_points(points)】


# import numpy as np
# import open3d as o3d
# import os


# def generate_random_point_cloud(num_points=100):
#     # 生成随机点云
#     points = np.random.rand(num_points, 3)
#     return points

# def test_draw_geometries():
#     # 生成随机点云
#     pc = generate_random_point_cloud()
#     # 使用 open3d 可视化点云
#     o3d.visualization.draw_plotly([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))])

# if __name__ == "__main__":
#     test_draw_geometries()

import numpy as np
import open3d as o3d

import os
os.environ['DISPLAY'] = ':0.0'
def generate_random_point_cloud(num_points=100):
    # 生成随机点云
    points = np.random.rand(num_points, 3)
    return points

def save_point_cloud_image(pc, save_path="point_cloud.jpg"):
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    
    # 设置无头渲染
    vis = o3d.visualization.Visualizer()
    vis.create_window()  # 创建一个不可见的窗口

    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # 捕获当前视图并保存为图像
    vis.capture_screen_image(save_path)
    vis.destroy_window()

if __name__ == "__main__":
    pc = generate_random_point_cloud()
    save_point_cloud_image(pc, "point_cloud.jpg")
