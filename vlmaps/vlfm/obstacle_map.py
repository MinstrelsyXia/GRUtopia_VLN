# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt


from vlmaps.vlfm.frontier_detection import detect_frontier_waypoints
from vlmaps.vlfm.fog_of_war import reveal_fog_of_war, get_current_angle

from vlmaps.vlfm.base_map import BaseMap
import os
# from depth_camera_filtering import filter_depth
# from agent_utils.geometry_utils import extract_camera_pos_zyxrot, get_extrinsic_matrix, get_world_points_from_image_coords
# from agent_utils.img_utils import fill_small_holes


class ObstacleMap(BaseMap):
    """Generates two maps; one representing the area that the robot has explored so far,
    and another representing the obstacles that the robot has seen so far.
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
        size: int = 1000,
        pixels_per_meter: int = 20,
        log_image_dir: str = None,
        dilate_iters: int = 1
    ):
        super().__init__(size, pixels_per_meter)
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


    def reset(self) -> None:
        super().reset()
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
        step: int = 0
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
            camera_xy_location = camera_position[:2]
            camera_rotation = camera_orientation 
            agent_pixel_location = self._xy_to_px(np.array([camera_xy_location]))[0]

            max_depth_limit = np.min([max_depth, 10])

            
            # Populate topdown map with obstacle locations
            xy_points = obstacle_cloud[:, :2]
            pixel_points = self._xy_to_px(xy_points) #! didn't align with semantic map
            
            self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1

            
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

        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle= camera_rotation[2], # modified!
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
        self._frontiers_px = self._get_frontiers()
        # if verbose:
        #     explored_area_uint8 = self.explored_area.astype(np.uint8)
        #     for frontier in self._frontiers_px:
        #         cv2.circle(explored_area_uint8, tuple([int(i) for i in frontier]), 5, (255,0,0), -1)
        #     save_path = os.path.join(self.save_dir,f'explored_with_frontiers_{step}.jpg')
        #     plt.imsave(save_path, explored_area_uint8)
        #     save_path = os.path.join(self.save_dir,f'explored_map_{step}.jpg')
        #     plt.imsave(save_path, self.explored_area)

        if verbose:
            # 创建可视化用的 navigable_map 副本
            navigable_map_visual = self._navigable_map.astype(np.uint8) * 255  # 将 navigable_map 转为黑白图像 (0-255)

            # 将 navigable_map 扩展为三通道 (灰度图变为RGB图像)
            navigable_map_visual = cv2.cvtColor(navigable_map_visual, cv2.COLOR_GRAY2BGR)

            navigable_map_visual[self._navigable_map>0] = (60,60,60) 
            navigable_map_visual[self._navigable_map == 0] = (255 ,255, 255) 
            # 创建 explored_area 的三通道灰色图像 (灰色为 [128,128,128])
            navigable_map_visual[self.explored_area > 0] = (128, 128, 128)  # 灰色 (BGR) 表示 explored_area
            # 在 visual_map 上圈出 frontiers，用红色标记
            for frontier in self._frontiers_px:
                cv2.circle(navigable_map_visual, tuple([int(i) for i in frontier]), 3, (0, 0, 255), -1)  # 红色 (BGR) 表示 frontiers

            # 创建自己此时位置：
            cv2.circle(navigable_map_visual, tuple(agent_pixel_location), 3, (255, 192, 15), -1)  # 蓝色 (BGR) 表示 frontiers
            # 保存最终结果
            save_path = os.path.join(self.save_dir, f'explored_with_frontiers_{step}.jpg')
            cv2.imwrite(save_path, navigable_map_visual)
            cv2.imwrite('/ssd/xiaxinyuan/code/w61-grutopia/tmp/explored_with_frontiers.jpg', navigable_map_visual)

            # 另存 explored_area，仅为灰度图，探索区域为灰色，未探索区域为黑色
            # explored_area_gray = self.explored_area.astype(np.uint8) * 255  # 将 explored_area 转为黑白图像
            # save_path = os.path.join(self.save_dir, f'explored_map_{step}.jpg')
            # cv2.imwrite(save_path, explored_area_gray)

        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)
        self.nav_map_visual = navigable_map_visual


    def update_map(
        self,
        depth: np.ndarray,
        camera_in: np.ndarray,
        camera_transform: np.ndarray,
        min_depth: float,
        max_depth: float,
        topdown_fov: float, 
        explore: bool = True,
        update_obstacles: bool = True,
        verbose: bool = False
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
        """
        # extract info from observations
        depth = filter_depth(depth, blur_type=None)
        camera_ex = get_extrinsic_matrix(camera_transform)

        # update obstacle map
        if update_obstacles:
            if self._hole_area_thresh == -1:
                filled_depth = depth.copy()
                filled_depth[depth == 0] = 1.0
            else:
                filled_depth = fill_small_holes(depth, self._hole_area_thresh)
            mask = (depth < max_depth) * (depth > min_depth)
            point_cloud_episodic_frame = get_world_points_from_image_coords(depth, mask, camera_ex, camera_in)
            obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, self._min_height, self._max_height)

            # Populate topdown map with obstacle locations
            xy_points = obstacle_cloud[:, :2]
            pixel_points = self._xy_to_px(xy_points)
            self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1

            # Update the navigable area, which is an inverse of the obstacle map after a
            # dilation operation to accommodate the robot's radius.
            self._navigable_map = 1 - cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1,
            ).astype(bool)
        
        if verbose: 
            plt.imsave('GRUtopia/grutopia_extension/agents/social_navigation_agent/images/obstacle_map.jpg', self._map)
            plt.imsave('GRUtopia/grutopia_extension/agents/social_navigation_agent/images/navigatable_map.jpg', self._navigable_map)
        if not explore:
            return

        # Update the explored area
        camera_position, camera_rotation = extract_camera_pos_zyxrot(camera_transform)
        camera_xy_location = camera_position[:2].reshape(1, 2)
        agent_pixel_location = self._xy_to_px(camera_xy_location)[0]
        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle= -np.pi/2 - camera_rotation[0], 
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
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
        self._frontiers_px = self._get_frontiers()
        if verbose:
            explored_area_uint8 = self.explored_area.astype(np.uint8)
            for frontier in self._frontiers_px:
                cv2.circle(explored_area_uint8, tuple([int(i) for i in frontier]), 5, 1, -1)
            plt.imsave('GRUtopia/grutopia_extension/agents/social_navigation_agent/images/explored_with_frontiers.jpg', explored_area_uint8)
            plt.imsave('GRUtopia/grutopia_extension/agents/social_navigation_agent/images/explored_map.jpg', self.explored_area)
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)

    def _get_frontiers(self) -> np.ndarray:
        """Returns the frontiers of the map."""
        # Dilate the explored area slightly to prevent small gaps between the explored
        # area and the unnavigable area from being detected as frontiers.
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        frontiers = detect_frontier_waypoints(
            self._navigable_map.astype(np.uint8),
            explored_area,
            self._area_thresh_in_pixels,
        )
        return frontiers

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
