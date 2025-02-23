# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt


from vlmaps.vlfm.frontier_detection_v2 import detect_frontier_waypoints
from vlmaps.vlfm.fog_of_war_v2 import reveal_fog_of_war, get_current_angle

from vlmaps.vlfm.base_map import BaseMap
import os
from typing import List
from scipy.ndimage import binary_dilation
# from depth_camera_filtering import filter_depth
# from agent_utils.geometry_utils import extract_camera_pos_zyxrot, get_extrinsic_matrix, get_world_points_from_image_coords
# from agent_utils.img_utils import fill_small_holes
from vlmaps.vlfm.traj_visualizer import TrajectoryVisualizer

def create_dilation_structure(voxel_size, radius):
    """
    Creates a dilation structure based on the robot's radius.
    """
    radius_cells = int(np.ceil(radius / voxel_size))
    # Create a structuring element for dilation (a disk of the robot's radius)
    dilation_structure = np.zeros((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
    cy, cx = radius_cells, radius_cells
    for y in range(2 * radius_cells + 1):
        for x in range(2 * radius_cells + 1):
            if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius_cells:
                dilation_structure[y, x] = True
    return dilation_structure

class ObstacleMap:
    """Generates two maps; one representing the area that the robot has explored so far,
    and another representing the obstacles that the robot has seen so far.
    self._map: 1-obstacle
    self._navigable_map: 1-freemap, 0-obstacle(dilated_version)
    self.explored_area: 1-explored, 0-unexplored
    """

    _map_dtype: np.dtype = np.dtype(bool)
    _frontiers_px: np.ndarray = np.array([])
    frontiers: np.ndarray = np.array([])
    radius_padding_color: tuple = (100, 100, 100)

    def __init__(
        self,
        min_height: float,
        max_height: float,
        agent_radius: float,
        area_thresh: float = 3.0,  # square meters
        hole_area_thresh: int = 100000,  # square pixels
        size: int = 100,
        pixels_per_meter: int = 20,
        log_image_dir: str = None,
        dilate_iters: int = 1
    ):
        self.pixels_per_meter = pixels_per_meter
        self.cs = 1 / pixels_per_meter
        self.size = size
        self._traj_vis = TrajectoryVisualizer(np.array([0,0]), self.pixels_per_meter)
        self.explored_area = np.zeros((size, size), dtype=bool)
        self._map = np.zeros((size, size), dtype=bool)
        self._navigable_map = np.zeros((size, size), dtype=bool)
        self._min_height = min_height
        self._max_height = max_height
        self._area_thresh_in_pixels = area_thresh * (self.pixels_per_meter**2)
        self._hole_area_thresh = hole_area_thresh
        kernel_size = self.pixels_per_meter * agent_radius * 2
        self.robot_radius = agent_radius
        # round kernel_size to nearest odd number
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.save_dir = log_image_dir + '/obstacle_map'
        if os.path.exists(self.save_dir) == False:
            os.makedirs(self.save_dir)
        self._dilate_iters = dilate_iters
        self.nav_map_visual = None
        self.dilation_structure = create_dilation_structure(1/self.pixels_per_meter, agent_radius)
        self.pcd_min = np.array([0, 0])
        self.pcd_max = np.array([0, 0])

    def freemap_to_accupancy_map(self, add_dilation=True):
        '''
        Maps:
            self._map: 0-freemap, 1-obstacle
            self._navigable_map: 1-obstacle, 0-freemap(dilated_version)
            self.explored_area: 1-explored, 0-unexplored
        Output: 
            occupancy_map: 255-obstacle, 2-free, 0-unexplored
        '''
        occupancy_map = np.zeros_like(self._navigable_map, dtype=np.uint8)
        occupancy_map[np.logical_and(self._map == 0, self.explored_area == 1)] = 2
        
        occupancy_map[self._map == 1] = 255
        if add_dilation:
            for i in range(1, self._dilate_iters+1):
                ob_mask = np.logical_and(occupancy_map!=0, occupancy_map!=2)
                expanded_ob_mask = binary_dilation(ob_mask, structure=self.dilation_structure, iterations=self._dilate_iters)
                occupancy_map[expanded_ob_mask&(np.logical_or(occupancy_map==0,occupancy_map==2))] = 255 - i*10
        return occupancy_map

    def reset(self) -> None:
        super().reset()
        self._map.fill(0)
        self._navigable_map.fill(0)
        self.explored_area.fill(0)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])

    def get_random_free_point(self):
        free_points = np.argwhere(self.explored_area == 1)
        if len(free_points) == 0:
            return None
        idx = np.random.randint(0, len(free_points))
        return free_points[idx]
    
    def get_forward_pos_v2(self, curr_pos, curr_angle, meters):
        """在扇形区域内寻找最优前进点
        
        Args:
            curr_pos: 当前位置 [x, y]
            curr_angle: 当前朝向(弧度)
            meters: 期望前进距离
            fov: 视场角(弧度)，默认180度
        
        Returns:
            np.array: 最优前进点的坐标 [x, y]
        """
        fov = self.fov
        i, j = curr_pos[0], curr_pos[1]
        pix = int(meters * self.pixels_per_meter)
        half_fov = fov / 4
        
        # 待检查的角度列表：优先检查当前角度，一直到左右四分之一视场角，每次角度差为5
        angles_to_check = np.linspace(curr_angle - half_fov, curr_angle + half_fov, int(fov/(5/180*np.pi)))
        
        best_distance = 0  # 最大可行进距离
        best_angle_diff = float('inf')  # 与当前角度的最小差值
        best_point = np.array([i, j])  # 最优点
        
        best_score = float('-inf')  # 最高启发值
        angle_weight = 0.2  # 角度权重
        distance_weight = 0.8  # 距离权重
        required_distance = meters * self.pixels_per_meter
        for angle in angles_to_check:
            # 计算方向向量
            cos_rad = np.cos(angle)
            sin_rad = np.sin(angle)
            
            # 生成路径上的所有点
            steps = np.arange(pix)
            path_i = i + steps * cos_rad
            path_j = j + steps * sin_rad
            
            # 将坐标转换为整数
            path_i = path_i.astype(np.int32)
            path_j = path_j.astype(np.int32)
            
            # 确保所有坐标都在地图范围内
            mask = (
                (path_i >= 0) & 
                (path_i < self._map.shape[0]) & 
                (path_j >= 0) & 
                (path_j < self._map.shape[1])
            )
            
            if not np.any(mask):
                continue
            
            # 获取有效路径点
            valid_i = path_i[mask]
            valid_j = path_j[mask]
            
            # 检查路径上的点是否在已探索区域内且可通行
            explored = self.explored_area[valid_i, valid_j]
            navigable = self._navigable_map[valid_i, valid_j]
            valid_points = np.logical_and(explored == 1, navigable == 1)
            valid_indices = np.where(valid_points)[0]
            
            # if len(valid_indices) == 0:
            #     continue
                
            # # 计算最远可行点
            # last_valid_idx = valid_indices[-1]
            # distance = np.sqrt((valid_i[last_valid_idx] - i)**2 + 
            #                 (valid_j[last_valid_idx] - j)**2)
            # angle_diff = abs(angle - curr_angle)
            
            # # 更新最优点
            # if distance > best_distance or (distance == best_distance and angle_diff < best_angle_diff):
            #     best_distance = distance
            #     best_angle_diff = angle_diff
            #     best_point = np.array([valid_i[last_valid_idx], valid_j[last_valid_idx]])
            if len(valid_indices) == 0:
                continue
                
            # 计算最远可行点
            last_valid_idx = valid_indices[-1]
            distance = np.sqrt((valid_i[last_valid_idx] - i)**2 + 
                            (valid_j[last_valid_idx] - j)**2)
            
            # 计算角度差异（归一化到[0,1]范围）
            angle_diff = abs(angle - curr_angle)
            normalized_angle_score = 1 - (angle_diff / half_fov)  # 角度差越小分数越高
            
            # 计算距离得分（归一化到[0,1]范围）
            normalized_distance_score = 1 - abs(distance - required_distance) / required_distance
            if distance < required_distance / 2:
                normalized_distance_score *= 0.5  # 惩罚过短的距离
            
            # 计算综合得分
            score = angle_weight * normalized_angle_score + distance_weight * normalized_distance_score
            
            # 更新最优点
            if score > best_score:
                best_score = score
                best_point = np.array([valid_i[last_valid_idx], valid_j[last_valid_idx]])
    
        
        # min_required_distance = 0.5 * self.pixels_per_meter   # 最小要求距离为期望距离的一半
    
        # if best_distance < min_required_distance:
        #     print("facing the wall, not moving")
        #     return np.array([i, j])  # 返回当前位置
        if best_score < 0.5:  # 可以调整这个阈值
            print("facing the wall, not moving")
            return np.array([i, j])

        return best_point
    
    def get_forward_pos(self, curr_pos: List[float], curr_angle: float, meters: float) -> List[float]:
        '''
        在已探索区域中找到指定方向上最远的可达点，确保路径上所有点都是已探索的
        
        Args:
            curr_pos: 在地图坐标系中的当前位置 [i, j]
            curr_angle: 在地图坐标系中的角度(弧度)
            meters: 前进距离(米)
        
        Returns:
            List[float]: 新位置坐标 [i, j]，返回路径上最远的已探索点
        '''
        i, j = curr_pos[1], curr_pos[0]
        rad = curr_angle
        pix = int(meters * self.pixels_per_meter)
        
        # 计算方向向量
        cos_rad = np.cos(rad)
        sin_rad = np.sin(rad)
        
        # 生成路径上的所有点
        steps = np.arange(pix)
        path_i = i + steps * sin_rad
        path_j = j + steps * cos_rad
        # 将坐标转换为整数
        path_i = path_i.astype(np.int32)
        path_j = path_j.astype(np.int32)
        
        # 确保所有坐标都在地图范围内
        mask = (
            (path_i >= 0) & 
            (path_i < self._map.shape[0]) & 
            (path_j >= 0) & 
            (path_j < self._map.shape[1])
        )
        
        if not np.any(mask):
            return [i, j]  # 如果所有点都超出范围，返回当前位置
        
        # 获取有效路径点
        valid_i = path_i[mask]
        valid_j = path_j[mask]
        
        # 检查路径上的点是否在已探索区域内
        explored = self.explored_area[valid_i, valid_j]
        explored_indices = np.where(explored == 1)[0]
        
        if len(explored_indices) == 0:
            return [i, j]  # 如果没有已探索点，返回当前位置
        
        # 找到最远的已探索点的索引
        last_explored_idx = explored_indices[-1]
        
        # 返回最远的已探索点
        return [valid_j[last_explored_idx], valid_i[last_explored_idx]]
    
    
    
    def clear_robot_surrounding(self, robot_pos, robot_radius, num_points=36):
        '''
        input: 
        - robot_pos: tuple (x, y) representing the robot's position
        - robot_radius: float, radius around the robot to clear
        - num_points: int, number of points to generate around the robot (default is 36)
        
        output: 
        - List of surrounding points (x, y) within the specified radius
        '''
        x, y = robot_pos  # 机器人当前位置
        surrounding_points = []

        # 生成num_points个在半径为robot_radius的圆周上的点
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        
        for angle in angles:
            # 计算圆周上点的坐标
            point_x = x + robot_radius * np.cos(angle)
            point_y = y + robot_radius * np.sin(angle)
            surrounding_points.append((point_x, point_y))

        return np.array(surrounding_points)


    def _get_current_angle_on_map(self,camera_orientation):
        '''
        camera_orientation: in row, pitch, yaw format, 1-dim
        '''
        return get_current_angle(camera_orientation)

    def _xy_to_px(self, points: Union[np.ndarray, list, tuple]) -> np.ndarray:
        """将世界坐标转换为像素坐标。

        Args:
            points: 坐标输入，支持以下格式：
                - list/tuple: [x, y] 或 [x, y, z]
                - np.ndarray: 
                    - shape (2,) 或 (3,): 单个点 [x,y] 或 [x,y,z]
                    - shape (n,2) 或 (n,3): 多个点 [[x,y],...] 或 [[x,y,z],...]

        Returns:
            np.ndarray: shape (n,2) 的像素坐标数组
        """
        # 转换输入为numpy数组
        points = np.asarray(points)
        
        # 处理单个点的情况
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        # 只取前两个坐标(x,y)
        xy_points = points[:, :2]
        
        # 转换为像素坐标
        px = np.rint((xy_points - self.pcd_min) * self.pixels_per_meter).astype(int)
        
        return px.astype(int)

    def _px_to_xy(self, px: Union[np.ndarray, list, tuple], z_axes: float = None) -> np.ndarray:
        """将像素坐标转换为世界坐标。
        Args:
            px: 像素坐标输入，支持以下格式：
                - list/tuple: [px, py]
                - np.ndarray: 
                    - shape (2,): 单个点 [px,py]
                    - shape (n,2): 多个点 [[px,py],...]
            z_axes: bool, 是否包含z轴坐标
        Returns:
            np.ndarray: shape (n,2) 的世界坐标数组
        
        Raises:
            ValueError: 当输入点的维度不正确时
        """
        try:
            px = np.asarray(px, dtype=np.float32)
        except:
            raise ValueError("输入点无法转换为numpy数组")
        
        # 验证输入维度
        if px.ndim == 1:
            if len(px) != 2:
                raise ValueError("单点输入必须包含两个坐标")
            px = px.reshape(1, -1)
        elif px.ndim == 2:
            if px.shape[1] != 2:
                raise ValueError("每个点必须是二维坐标")
        else:
            raise ValueError(f"不支持的输入维度: {px.ndim}")
        
        # 转换为世界坐标
        points = px / self.pixels_per_meter + self.pcd_min
        if z_axes is not None:
            points = np.hstack([points, np.full((points.shape[0], 1), z_axes)])
        return points
    
    def _yaw_to_pz(self, yaw: float) -> float:
        """将地图上的偏航角转换为真实世界中的方向向量。
        
        从地图坐标系的yaw[0,2π]转到真实世界中的yaw[-π,π],需要 -π 操作

        Args:
            yaw: 地图上的偏航角(弧度)[0,2π]

        Returns:
            float: 真实世界中的z轴方向分量[-π,π]
        """
        pz = yaw  # 将[0,2π]映射到[-π,π]
        return float(pz)

    def _pz_to_yaw(self, orientation: float) -> float:
        """将真实世界中的方向向量转换为地图上的偏航角。
        
        从真实世界的yaw[-π,π]转到地图坐标系的yaw[0,2π],需要 +π 操作
        
        Args:
            orientation (float): 真实世界中的方向角度，单位为弧度([-π,π])

        Returns:
            float: 地图上的偏航角[0,2π]
        """
        yaw = orientation #! direct mapping !
        return float(yaw)
    
    def expand_map(self, xy_points: np.ndarray, curr_position) -> np.ndarray:
        """扩展地图以容纳新的世界坐标点，并更新坐标系边界。

        Args:
            xy_points: (np.ndarray), shape:[n,2] - 世界坐标系下的点
                xy_points[:,0] 是 x 坐标
                xy_points[:,1] 是 y 坐标

        """
        # 获取当前点云的边界

        min_x = min(xy_points[:, 0].min(),curr_position[0])
        max_x = max(xy_points[:, 0].max(),curr_position[0])
        min_y = min(xy_points[:, 1].min(),curr_position[1])
        max_y = max(xy_points[:, 1].max(),curr_position[1])
        
        # 计算是否需要更新pcd_min和pcd_max，并保存旧值
        need_update = False
        old_pcd_min = self.pcd_min.copy()
        new_pcd_min = np.array([
            min(self.pcd_min[0], min_x),
            min(self.pcd_min[1], min_y)
        ])
        new_pcd_max = np.array([
            max(self.pcd_max[0], max_x),
            max(self.pcd_max[1], max_y)
        ])
        
        if any(new_pcd_min != self.pcd_min) or any(new_pcd_max != self.pcd_max):
            need_update = True
        
        # 计算新地图需要的尺寸
        world_width = new_pcd_max[0] - new_pcd_min[0]
        world_height = new_pcd_max[1] - new_pcd_min[1]
        
        pixel_width = int(np.ceil(world_width * self.pixels_per_meter))
        pixel_height = int(np.ceil(world_height * self.pixels_per_meter))
        
        # 确保尺寸是100的倍数，且不小于初始尺寸(self.size)
        pixel_width = max(self.size, int(np.ceil(pixel_width / 100) * 100))
        pixel_height = max(self.size, int(np.ceil(pixel_height / 100) * 100))
        
        if need_update or pixel_width > self._map.shape[0] or pixel_height > self._map.shape[1]:
            # 更新边界值
            self.pcd_min = new_pcd_min
            self.pcd_max = new_pcd_max
            
            # 创建新地图
            new_shape = [pixel_width, pixel_height]
            new_map = np.zeros(new_shape, dtype=self._map.dtype)
            new_explored_area = np.zeros(new_shape, dtype=self.explored_area.dtype)
        
            
            # 计算原地图对应的世界坐标范围
            old_map_world_coords = np.array([
                old_pcd_min,  # 原地图左下角
                old_pcd_min + np.array([  # 原地图右上角
                    self._map.shape[0] / self.pixels_per_meter,
                    self._map.shape[1] / self.pixels_per_meter
                ])
            ])
            
            # 计算原地图在新地图中的像素位置
            old_map_pixels = self._xy_to_px(old_map_world_coords)
            
            # 计算原地图在新地图中的范围
            dst_x_start = max(0, old_map_pixels[0, 0])
            dst_y_start = max(0, old_map_pixels[0, 1])
            
            # 计算在原地图中的起始位置（处理负值情况）
            src_x_start = max(0, -old_map_pixels[0, 0])
            src_y_start = max(0, -old_map_pixels[0, 1])
            
            # 计算需要复制的宽度和高度
            copy_width = min(
                self._map.shape[0] - src_x_start,  # 原地图中可用的宽度
                new_shape[0] - dst_x_start  # 新地图中可用的宽度
            )
            copy_height = min(
                self._map.shape[1] - src_y_start,  # 原地图中可用的高度
                new_shape[1] - dst_y_start  # 新地图中可用的高度
            )
            
            # 确保复制区域有效
            if copy_width > 0 and copy_height > 0:
                # 复制原有数据到新位置
                new_map[
                    dst_x_start:dst_x_start + copy_width,
                    dst_y_start:dst_y_start + copy_height
                ] = self._map[
                    src_x_start:src_x_start + copy_width,
                    src_y_start:src_y_start + copy_height
                ]
                
                new_explored_area[
                    dst_x_start:dst_x_start + copy_width,
                    dst_y_start:dst_y_start + copy_height
                ] = self.explored_area[
                    src_x_start:src_x_start + copy_width,
                    src_y_start:src_y_start + copy_height
                ]
            
            # 更新地图
            self._map = new_map
            self.explored_area = new_explored_area
        
        # 返回新坐标系下的点位置
        return self._xy_to_px(xy_points)
    
    def update_map_with_pc(
        self,
        pc: np.ndarray,
        camera_position: np.ndarray,
        camera_orientation : np.ndarray,
        max_depth: float,
        topdown_fov: float,
        explore: bool = True,
        update_obstacles: bool = True,
        verbose: bool = False,
        step: int = 0,
        get_grad: bool = False
    ) -> None:
        """
        Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.

        Args:
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).

            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.
            topdown_fov (float): The field of view of the depth camera projected onto
                the topdown map.
            explore (bool): Whether to update the explored area.
            update_obstacles (bool): Whether to update the obstacle map.
            camera_orientation : in (rot, pitch, yaw) format
        """
        # update obstacle map
        if update_obstacles:

            # if self._hole_area_thresh == -1:
            #     filled_depth = depth.copy()
            #     filled_depth[depth == 0] = 1.0
            # else:
            #     filled_depth = fill_small_holes(depth, self._hole_area_thresh)
            # mask = (depth < max_depth) * (depth > min_depth)
            # point_cloud_episodic_frame = get_world_points_from_image_coords(depth, mask, camera_ex, camera_in)
            # obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, self._min_height, self._max_height)

            obstacle_cloud = pc
            if len(obstacle_cloud) == 0:
                return
            camera_xy_location = camera_position[:2]
            camera_rotation = self._pz_to_yaw(camera_orientation[2])
            
            max_depth_limit = np.min([max_depth, 10])

            
            # Populate topdown map with obstacle locations
            xy_points = obstacle_cloud[:, :2]

            # dynamically change map size:
            self.expand_map(xy_points,camera_xy_location)
            agent_pixel_location = self._xy_to_px(camera_xy_location)[0] 

            new_pixel_points = self._xy_to_px(xy_points)
            self._map[new_pixel_points[:, 0], new_pixel_points[:, 1]] = 1

            
            # agent_surrounding = self.clear_robot_surrounding(camera_position[:2], self.robot_radius*2)
            # agent_surrounding_on_map = self._xy_to_px(agent_surrounding)
            # # self._map[agent_pixel_location[1], agent_pixel_location[0]] = 0
            # self._map[agent_surrounding_on_map[:, 1], agent_surrounding_on_map[:, 0]] = 0

            # Update the navigable area, which is an inverse of the obstacle map after a
            # dilation operation to accommodate the robot's radius.
            self._navigable_map = 1 - cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=self._dilate_iters,
            ).astype(bool)
        
        # if verbose: 
        #     obs_map_save_path = os.path.join(self.save_dir,f'obstacle_map_{step}.jpg')
        #     plt.imsave(obs_map_save_path, self._map)
        #     navigatable_map_save_path = os.path.join(self.save_dir,f'navigatable_map_{step}.jpg')
        #     plt.imsave(navigatable_map_save_path, self._navigable_map)
        if not explore:
            return

        # Update the explored area
        # camera_position, camera_rotation = extract_camera_pos_zyxrot(camera_transform)
        # camera_xy_location = camera_position[:2].reshape(1, 2)
        self.fov = topdown_fov
        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location,
            current_angle = camera_rotation , # modified!
            fov=np.rad2deg(topdown_fov),
            max_line_len= max_depth_limit * self.pixels_per_meter,
            enable_debug_visualization=True
        )
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)
        self.explored_area[new_explored_area > 0] = 1
        self.explored_area[self._navigable_map == 0] = 0
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) > 1:
            min_dist = np.inf
            best_idx = 0
            for idx, cnt in enumerate(contours):
                dist = cv2.pointPolygonTest(cnt, tuple([int(i) for i in agent_pixel_location]), True)
                if dist >= 0:
                    best_idx = idx
                    break
                elif abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_idx = idx
            new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
            cv2.drawContours(new_area, contours, best_idx, 1, -1)  # type: ignore
            self.explored_area = new_area.astype(bool)

        
        # Compute frontier locations
        self._frontiers_px, self._frontiers_angles_obs = self._get_frontiers(get_grad=get_grad)


        if verbose:
            '''(1) 设置背景颜色'''
            #! 可视化效果和矩阵对齐
            # 创建可视化用的 navigable_map 副本,将 navigable_map 转为黑白图像 (0-255)
            navigable_map_visual = self._navigable_map.astype(np.uint8) * 255 
            # 将 navigable_map 扩展为三通道 (灰度图变为RGB图像)
            navigable_map_visual = cv2.cvtColor(navigable_map_visual, cv2.COLOR_GRAY2BGR)
            # > 0 处应该为free处，设置为深灰色（即vis过+未vis过的外层都是）
            navigable_map_visual[self._navigable_map > 0] = (60,60,60) 
            # = 0 处为 obs，设置为白色
            navigable_map_visual[self._navigable_map == 0] = (255 ,255, 255) 
            # explored_area > 0 处为现在explored area 连通区域，设置为浅灰色
            navigable_map_visual[self.explored_area > 0] = (128, 128, 128) 
            '''(2) 画出frontiers+ frontiers_grad'''
            # 在 visual_map 上圈出 frontiers，用红色标记
            navigable_map_visual = self._traj_vis.draw_frontiers(img = navigable_map_visual, positions=self._frontiers_px, angles = self._frontiers_angles_obs)
            '''(3) 画出agent位置'''
            navigable_map_visual = self._traj_vis.draw_agent(navigable_map_visual, agent_pixel_location, camera_rotation)
            # cv2.circle(navigable_map_visual, tuple(agent_pixel_location[::-1]), 3, (255, 192, 15), -1)  # 蓝色 (BGR) 表示 frontiers
            # # 保存最终结果

            '''(4) 保存可视化结果'''
            save_path = os.path.join(self.save_dir, f'explored_with_frontiers_{step}.jpg')
            cv2.imwrite(save_path, navigable_map_visual)
            new_save_path = os.path.join(
            os.path.dirname(os.path.dirname(save_path)),  # 上级目录
                'explored_with_frontiers.jpg'
            )
            cv2.imwrite(new_save_path, navigable_map_visual)


        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
            self.frontiers_angles = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)
            
        self.nav_map_visual = navigable_map_visual
        return navigable_map_visual

    

    def _get_frontiers(self, get_grad: bool = False) -> np.ndarray:
        """Returns the frontiers of the map."""
        # Dilate the explored area slightly to prevent small gaps between the explored
        # area and the unnavigable area from being detected as frontiers.
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        frontiers, frontiers_angles = detect_frontier_waypoints(
            self._navigable_map.astype(np.uint8),
            explored_area,
            self._area_thresh_in_pixels,
            get_grad=get_grad
        )
        return frontiers, frontiers_angles

    def _path_is_blocked(self, path):
        '''
        already updated obstacle map:

        '''
        # draw a straight line between start and end:

        # determine whether points intersect with obstacle map

    def visualize(self) -> np.ndarray:
        """Visualizes the map."""
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        # Draw explored area in light green
        vis_img[self.explored_area == 1] = (200, 255, 200)
        # Draw unnavigable areas in gray
        vis_img[self._navigable_map == 0] = self.radius_padding_color
        # Draw obstacles in black
        vis_img[self._map == 1] = (0, 0, 0)
        # Draw frontiers in blue (200, 0, 0)
        for frontier in self._frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in frontier]), 5, (200, 0, 0), 2)

        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img


def filter_points_by_height(points: np.ndarray, min_height: float, max_height: float) -> np.ndarray:
    return points[(points[:, 2] >= min_height) & (points[:, 2] <= max_height)]
