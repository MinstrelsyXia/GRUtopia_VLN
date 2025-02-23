# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# copied from vlfm/mapping/traj_visualizer.py
from typing import Any, List, Union

import cv2
import numpy as np


class TrajectoryVisualizer:
    _num_drawn_points: int = 1
    _cached_path_mask: Union[np.ndarray, None] = None
    _origin_in_img: Union[np.ndarray, None] = None
    _pixels_per_meter: Union[float, None] = None

    agent_body_radius: int = 3
    agent_line_length: int = 4
    agent_line_thickness: int = 1
    agent_color: tuple = (255, 192, 15) # beautiful blue
    agent_ori: tuple = (0, 0, 0)

    frontier_color: tuple = (0, 255, 0) # green
    frontier_ori: tuple = (0, 0, 0)

    path_color: tuple = (255, 0, 0) # blue
    path_thickness: int = 1
    scale_factor: float = 1.0
    arrow_length: int = 100

    def __init__(self, origin_in_img: np.ndarray, pixels_per_meter: float):
        self._origin_in_img = origin_in_img
        self._pixels_per_meter = pixels_per_meter

    def reset(self) -> None:
        self._num_drawn_points = 1
        self._cached_path_mask = None

    def draw_trajectory(
        self,
        img: np.ndarray,
        camera_positions: Union[np.ndarray, List[np.ndarray]],
        camera_yaw: float,
    ) -> np.ndarray:
        """Draws the trajectory on the image and returns it"""
        img = self._draw_path(img, camera_positions)
        # img = self.draw_agent(img, camera_positions[-1], camera_yaw)
        return img

    def _draw_path_waste(self, img: np.ndarray, camera_positions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Draws the path on the image and returns it"""
        if len(camera_positions) < 2:
            return img
        if self._cached_path_mask is not None:
            path_mask = self._cached_path_mask.copy()
        else:
            path_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        for i in range(self._num_drawn_points - 1, len(camera_positions) - 1):
            path_mask = self._draw_line(path_mask, camera_positions[i], camera_positions[i + 1])

        img[path_mask == 255] = self.path_color

        self._cached_path_mask = path_mask
        self._num_drawn_points = len(camera_positions)

        return img

    def draw_path(self, img: np.ndarray, camera_positions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """绘制路径"""
        img_copy = img.copy()
        
        if len(camera_positions) < 2:
            return img_copy
            
        for i in range(len(camera_positions) - 1):
            pt_a = camera_positions[i]
            pt_b = camera_positions[i + 1]
            img_copy = self._draw_line(img_copy, pt_a, pt_b)
        return img_copy

    def _draw_line(self, img: np.ndarray, pt_a: np.ndarray, pt_b: np.ndarray) -> np.ndarray:
        """画线"""
        img_copy = img.copy()
        
        px_a = self._metric_to_pixel(pt_a)
        px_b = self._metric_to_pixel(pt_b)

        if np.array_equal(px_a, px_b):
            return img_copy

        cv2.line(
            img_copy,
            tuple(px_a[::-1]),
            tuple(px_b[::-1]),
            255,
            int(self.path_thickness * self.scale_factor),
        )
        return img_copy

    def _draw_arrow(
        self, 
        img: np.ndarray, 
        pt_a: np.ndarray, 
        pt_b: np.ndarray, 
        color: tuple = (255, 255, 255),
        thickness: int = None,
        tipLength: float = 0.3
    ) -> np.ndarray:
        """画箭头"""
        img_copy = img.copy()
        
        px_a = self._metric_to_pixel(pt_a)
        px_b = self._metric_to_pixel(pt_b)

        if np.array_equal(px_a, px_b):
            return img_copy

        if thickness is None:
            thickness = int(self.path_thickness * self.scale_factor)

        cv2.arrowedLine(
            img_copy,
            tuple(px_a[::-1]),
            tuple(px_b[::-1]),
            color,
            thickness,
            tipLength=tipLength
        )
        return img_copy

    def draw_grad_direction(
        self, 
        img: np.ndarray, 
        positions: Union[np.ndarray, List[np.ndarray]],
        direction: Union[np.ndarray, List[float]],
        color: tuple = (0, 255, 0),
        thickness: int = 1,
    ) -> np.ndarray:
        """绘制梯度方向"""
        img_copy = img.copy()
        
        if isinstance(positions, list):
            positions = np.array(positions)
        if isinstance(direction, list):
            direction = np.array(direction)
        
        if positions.size == 0:
            return img_copy
            
        arrow_length = self.arrow_length
        
        for pos, angle in zip(positions, direction):
            angle_cv2 = -angle + np.pi/2
            end_point = pos - arrow_length/self._pixels_per_meter * np.array([
                np.cos(angle_cv2), 
                np.sin(angle_cv2)
            ])
            
            img_copy = self._draw_arrow(
                img_copy,
                pos,
                end_point,
                color=color,
                thickness=thickness,
                tipLength=0.3
            )
        return img_copy


    def draw_frontiers(self, positions, angles, img: np.ndarray) -> np.ndarray:
        img_copy = img.copy()
        for position, orientation in zip(positions, angles):
            img_copy = self._draw_agent(img_copy, position, orientation, self.frontier_color, self.frontier_ori)
        return img_copy
    
    def draw_agent(self, img, camera_position, camera_yaw):
        img_copy = self._draw_agent(img, camera_position, camera_yaw, self.agent_color, self.agent_ori)
        return img_copy
    
    def _draw_agent(self, img: np.ndarray, camera_position: np.ndarray, camera_yaw: float, agent_color: tuple, ori_color: tuple ) -> np.ndarray:
        """Draws the agent on the image and returns it"""
        img_copy = img.copy()
        px_position = self._metric_to_pixel(camera_position)
        cv2.circle(
            img_copy,
            tuple(px_position[::-1]),
            int(self.agent_body_radius * self.scale_factor),
            agent_color,
            -1,
        )
        heading_end_pt = (
            int(px_position[0] + self.agent_line_length * self.scale_factor * np.cos(camera_yaw)),
            int(px_position[1] + self.agent_line_length * self.scale_factor * np.sin(camera_yaw)),
        )
        cv2.line(
            img_copy,
            tuple(px_position[::-1]),
            tuple(heading_end_pt[::-1]),
            ori_color,
            int(self.agent_line_thickness * self.scale_factor),
        )

        return img_copy

    def draw_circle(self, img: np.ndarray, position: np.ndarray, radius, color) -> np.ndarray:
        """Draws the point as a circle on the image and returns it"""
        img_copy = img.copy()
        if position.size == 0:
            return img_copy
        # 处理单个点的情况
        if position.ndim == 1:
            position = position.reshape(1, -1)
        
        # 批量绘制所有点
        for pos in position:
            px_position = self._metric_to_pixel(pos)
            cv2.circle(img_copy, tuple(px_position[::-1]), radius, color, -1)
            
        return img_copy

    def _metric_to_pixel(self, pt: np.ndarray) -> np.ndarray:
        """Converts a metric coordinate to a pixel coordinate"""
        # Need to flip y-axis because pixel coordinates start from top left
        px = pt
        # px = pt * self._pixels_per_meter + self._origin_in_img
        px = px.astype(np.int32)
        return px