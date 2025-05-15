import math
from shapely.geometry import LineString
# from vln.src.v2.util.common_log_util import common_logger as log
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString

from collections import deque
from typing import Tuple, List, Dict
from vlmaps.vlfm.traj_visualizer import TrajectoryVisualizer
import cv2
import os
class AStarPlanner:
    def __init__(
        self, 
        map_width=500, 
        map_height=500, 
        max_step=10000,
        trajectory_visualizer: TrajectoryVisualizer = None,
    ):
        """
        Initialize grid map for a star planning.
        Note that this class does not consider the robot's radius. So the given obs_map should be expanded
        """
        self.resolution = 1
        self.max_step = max_step
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = map_width, map_height # init, = self.x_width
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        self.visualize_init()
        self._traj_vis = trajectory_visualizer
        
    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)
    ############ adapt to obstacleMap ############
    def plan_to(
        self, start: Tuple[float, float], goal: Tuple[float, float], vis: bool = False, navigable_map_visual = None, obs_map = None, save_path ='tmp/planned_path.jpg'
    ) -> np.ndarray:
        """
        Take full map start (row, col) and full map goal (row, col) as input
        Return a list of full map path points (row, col) as the palnned path
        """
        paths, _, _ = self.planning(start[0], start[1], goal[0], goal[1], obs_map)
        
        # 修复1: 验证路径并只保留有效部分
        valid_paths = self.find_valid_path(obs_map, paths)
        if len(valid_paths) == 0:
            print("No valid path found")
            return []
        
        if len(valid_paths) > 1:
            valid_paths = np.array([valid_paths[i] for i in range(len(valid_paths)) 
                                  if i == 0 or not np.array_equal(valid_paths[i], valid_paths[i-1])])
        
        if vis:
            obs_map_vis = self._traj_vis.draw_path(navigable_map_visual, valid_paths)
            cv2.imwrite(save_path, obs_map_vis)
            new_save_path = os.path.join(
                os.path.dirname(os.path.dirname(save_path)),  # 上级目录
                'planned_path.jpg'
            )
            cv2.imwrite(new_save_path, obs_map_vis)

        return valid_paths
    
    def _check_if_start_in_graph_obstacle(self, start: Tuple[float, float],obs_map: np.ndarray):
        if obs_map[start[0], start[1]] == 1:
            return True
        return False
    
    
    def check_path_blocked(self,start, goal, obs_map):
        '''
        start, goal: (row, col) in full map
        obs_map: 2D list or array representing the map, where 0 is obstacle and 1 is free
        '''
        goal = [int(goal[0]), int(goal[1])]
        line_points = bresenham_line(start[0], start[1], goal[0], goal[1])
        
        for point in line_points:
            row, col = point
            if obs_map[row][col] == 0:  # 0 indicates a free cell
                return True
        return False
        
    def get_angle_cost(self,gx,gy,current,x,y):
        import math

        # 计算从当前点到目标点的向量
        vector_current_to_goal = (gx - current.x, gy - current.y)

        # 计算从当前点到新点的向量
        vector_current_to_new = (x - current.x, y - current.y)

        # 计算向量的点积
        dot_product = vector_current_to_goal[0] * vector_current_to_new[0] + vector_current_to_goal[1] * vector_current_to_new[1]

        # 计算向量的模长
        magnitude_current_to_goal = math.sqrt(vector_current_to_goal[0]**2 + vector_current_to_goal[1]**2)
        magnitude_current_to_new = math.sqrt(vector_current_to_new[0]**2 + vector_current_to_new[1]**2)

        # 计算夹角的余弦值
        cos_theta = dot_product / (magnitude_current_to_goal * magnitude_current_to_new)

        # 保证夹角余弦值在 -1 和 1 之间，防止数值误差
        cos_theta = max(-1.0, min(1.0, cos_theta))

        # 计算夹角
        theta = math.acos(cos_theta)

        # 将夹角转换为代价：夹角越大，代价越高
        angle_cost = 100*theta  # 你可以根据需要调整这个权重

        return angle_cost
    
    def   find_valid_path(self, obs_map, path):
        """查找有效路径段"""
        if len(path) == 0:
            return []
            
        valid_path = [path[0]]  # 保留起点
        
        for i in range(1, len(path)):
            current = path[i]
            previous = valid_path[-1]
            
            # 检查两点之间的路径是否有效
            line_points = bresenham_line(
                int(previous[0]), int(previous[1]),
                int(current[0]), int(current[1])
            )
            
            path_valid = True
            for x, y in line_points:
                if not (0 <= x < obs_map.shape[0] and 0 <= y < obs_map.shape[1]):
                    path_valid = False
                    break
                if obs_map[x][y] == 255 or obs_map[x][y] == 0:
                    path_valid = False
                    break
                    
            if path_valid:
                valid_path.append(current)
            else:
                break  # 遇到无效路径段就停止
                
        return valid_path

    def find_nearest_free_node(self, obs_map, goal_node):
        if obs_map[goal_node.x, goal_node.y] != 255 and obs_map[goal_node.x, goal_node.y] != 0:
            return goal_node  # Goal node is not in an obstacle

        free_nodes = np.argwhere(np.logical_and(obs_map != 255, obs_map != 0))
        goal_position = np.array([goal_node.x, goal_node.y])

        distances = np.linalg.norm(free_nodes - goal_position, axis=1)
        nearest_free_node_index = np.argmin(distances)
        nearest_free_node = free_nodes[nearest_free_node_index]

        new_goal_node = self.Node(nearest_free_node[0], nearest_free_node[1], 0.0, -1)
        return new_goal_node
    ############ ending addaption ############
    
    def planning(self, sx, sy, gx, gy, obs_map, min_final_meter=1, use_new_cost=True,coord = 'obs') -> tuple[list[tuple[float, float]], bool]:
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
            min_final_meter: 像素点的个数

        output:
            paths: ndarray, 
        
        obs_map:
            255: obstacle
            2: free area, cost = 0
            0: unexplored area, cost = 240
            others: dilated area, with larger cost
            occupancy_map: 255-obstacle, 2-free, 0-unexplored, 240...: dilated obstacle

        """
        if coord == 'obs':
            self.x_width = obs_map.shape[0]
            self.y_width = obs_map.shape[1]
            self.max_x = self.x_width
            self.max_y = self.y_width
            start_node = self.Node(sx,sy,0.0,-1)
            goal_node = self.Node(gx,gy,0.0,-1)
            motion = self.get_motion_model()
            reason = None
# if obs_map[goal_node.x, goal_node.y] == 255 or obs_map[goal_node.x, goal_node.y]==0:
            #     reason = 'goal_in_obstacle'
            #     # return [], [], False, reason
            #     new_goal_node = self.find_nearest_free_node(obs_map, goal_node)
            #     goal_node = new_goal_node

            open_set, closed_set = dict(), dict()
            open_set[self.calc_grid_index(start_node)] = start_node
        else:
            raise ValueError("coord should be 'obs'")

        step = 0
        while step < self.max_step:
            step += 1
            if len(open_set) == 0:
                reason = 'open_set_empty'
                break
            c_id = min(open_set,key=lambda o: open_set[o].cost)
            current = open_set[c_id]
            to_final_dis = self.calc_heuristic(current, goal_node)
            if to_final_dis <= min_final_meter:
                # print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(motion):
                x = current.x + motion[i][0]
                y = current.y + motion[i][1]
                if use_new_cost:
                    obs_cost = self.get_cost_new(x,y,obs_map) 
                    obs_cost += self.get_angle_cost(gx,gy,current,x,y)
                else:
                    obs_cost = self.get_cost_old(x,y,obs_map)

                node = self.Node(x,
                                 y,
                                 current.cost + motion[i][2] + obs_cost,
                                 c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node, obs_map):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        find_flag = True
        if step == self.max_step:
            reason = 'plan_max_step'
            goal_node = current
            find_flag = False

        rx, ry = self.calc_final_path(goal_node, closed_set)
        points_list = list(zip(rx, ry))
        points_list.append((gx, gy))
        if len(points_list) > 1:
            explorable_points = self.find_valid_path(obs_map, points_list)
            points = self.simplify_path(explorable_points, tolerance=0.01, obs_map=obs_map)
            # points.append((gx, gy))
        else:
# log.warning(f"Path planning results only contain {len(points_list)} points.")
            points = []
            points.append((gx, gy))
        
        
        self.vis_whole_path(obs_map, 'tmp/whole_path.png', [points], for_llm=True, vis_latest_path=True,start_position = [sx,sy])
        points = np.array(points)
        return points, find_flag, reason

    def get_cost_new(self, x, y, obs_map):
        # 如果点在地图范围外，直接返回最大值
        if x < 0 or x >= self.max_x or y < 0 or y >= self.max_y:
            return 255

        # 初始化总cost和计数器
        total_cost = 0
        count = 0

        # 遍历以(x, y)为中心的5x5区域
        for dx in range(-2, 3):  # 从-2到2（包含）
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.max_x and 0 <= ny < self.max_y:  # 确保在地图范围内
                    if obs_map[nx][ny] == 0: # unexplored
                        cost = 240
                    elif obs_map[nx][ny] == 2: # explored
                        cost = 0
                    else:
                        cost = obs_map[nx][ny]
                    total_cost += cost
                    count += 1

        # 防止count为0，理论上不会发生，防御性编程
        if count == 0:
            return 255
        
        # 返回平均值
        return total_cost // count

    def get_cost_old(self, x, y, obs_map):
        if x < self.max_x and y < self.max_y:
            if obs_map[x][y] == 0:
                cost = 240
            elif obs_map[x][y] == 2:
                cost = 0
            else:
                cost = obs_map[x][y]
            return cost
        else:
            return 255

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        rx.reverse() # from begin to end
        ry.reverse()

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)
    
    

    def verify_node(self, node, obs_map):
        """验证节点是否有效"""
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)
        
        # 检查边界
        if (px < self.min_x or py < self.min_y or 
            px >= self.max_x or py >= self.max_y):
            return False
        
        # 修复2: 正确检查障碍物
        # 255表示障碍物, 0表示未探索区域
        # 只绕过障碍物，不考虑未探索区域
        if obs_map[node.x][node.y] == 255:
            return False
        
        # # 检查节点周围的安全区域
        # radius = 2  # 安全半径
        # x_start = max(0, node.x - radius)
        # x_end = min(self.max_x, node.x + radius + 1)
        # y_start = max(0, node.y - radius)
        # y_end = min(self.max_y, node.y + radius + 1)
        
        # area = obs_map[x_start:x_end, y_start:y_end]
        # if np.any(area == 255):  # 如果周围有障碍物，返回False
        #     return False
            
        return True
    
    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

    # def simplify_path(self, points, tolerance=0.01):
    #     ''' The tolerance sets sampling distance. The smaller the tolerance, the more points in the simplified line.
    #     '''
    #     line = LineString(points)
    #     simplified_line = line.simplify(tolerance, preserve_topology=False)
    #     return list(simplified_line.coords)

    # def simplify_path(self, points, tolerance=0.01, obs_map=None):
    #     '''
    #     简化路径，同时确保简化后的点不在障碍物中
        
    #     Args:
    #         points: 原始路径点列表
    #         tolerance: 简化容差，越小保留的点越多
    #         obs_map: 障碍物地图，用于检查点是否可行
            
    #     Returns:
    #         简化后且无障碍的路径点列表
    #     '''
    #     # 如果没有提供障碍物地图，使用原始的简化方法
    #     if obs_map is None:
    #         line = LineString(points)
    #         simplified_line = line.simplify(tolerance, preserve_topology=False)
    #         return list(simplified_line.coords)
        
    #     # 第一步：使用LineString进行基本简化
    #     line = LineString(points)
    #     simplified_line = line.simplify(tolerance, preserve_topology=False)
    #     simplified_points = list(simplified_line.coords)
        
    #     # 第二步：检查简化后的路径上所有点是否都可行
    #     valid_points = [points[0]]  # 始终保留起点
        
    #     for i in range(1, len(simplified_points)):
    #         current_point = simplified_points[i]
    #         x, y = int(current_point[0]), int(current_point[1])
            
    #         # 检查当前点是否在地图范围内且不是障碍物
    #         if (0 <= x < self.max_x and 0 <= y < self.max_y and 
    #             obs_map[x][y] != 255 and obs_map[x][y] != 0):
    #             # 检查当前点与上一个有效点之间的连线是否穿过障碍物
    #             prev_point = valid_points[-1]
    #             line_points = bresenham_line(prev_point[0], prev_point[1], x, y)
                
    #             is_path_valid = True
    #             for lx, ly in line_points:
    #                 if (0 <= lx < self.max_x and 0 <= ly < self.max_y and 
    #                     (obs_map[lx][ly] == 255 or obs_map[lx][ly] == 0)):
    #                     is_path_valid = False
    #                     break
                
    #             if is_path_valid:
    #                 valid_points.append(current_point)
    #             else:
    #                 # 如果连线不可行，尝试添加原始路径中的中间点
    #                 idx_start = points.index(prev_point) if prev_point in points else 0
    #                 idx_current = -1
                    
    #                 # 找到当前简化点在原始路径中的最近点
    #                 min_dist = float('inf')
    #                 for j, p in enumerate(points):
    #                     if j <= idx_start:
    #                         continue
    #                     dist = math.hypot(p[0] - current_point[0], p[1] - current_point[1])
    #                     if dist < min_dist:
    #                         min_dist = dist
    #                         idx_current = j
                    
    #                 # 添加原始路径中的中间点，直到找到可行路径
    #                 for j in range(idx_start + 1, idx_current + 1):
    #                     ox, oy = int(points[j][0]), int(points[j][1])
    #                     if (0 <= ox < self.max_x and 0 <= oy < self.max_y and 
    #                         obs_map[ox][oy] != 255 and obs_map[ox][oy] != 0):
    #                         valid_points.append(points[j])
        
    #     # 确保终点被包含
    #     if points[-1] not in valid_points:
    #         valid_points.append(points[-1])
            
    #     return valid_points


    def simplify_path(self, points, tolerance=0.01, obs_map=None):
        '''
        优化版简化路径函数，减少点数同时确保路径无障碍
        
        Args:
            points: 原始路径点列表
            tolerance: 简化容差，越小保留的点越多
            obs_map: 障碍物地图，用于检查点是否可行
            
        Returns:
            简化后且无障碍的路径点列表
        '''
        if obs_map is None: # Use internal if available, for convenience
            obs_map = self.obs_map_data

        # If no points or too few points, or no obs_map, return as is or basic simplify
        if not points or len(points) <= 2:
            if obs_map is None and points and len(points) > 2 : # Fallback if no obs_map
                 # This import would be at the top of the file
                 from shapely.geometry import LineString 
                 line = LineString(points)
                 simplified_line = line.simplify(tolerance, preserve_topology=False)
                 return list(simplified_line.coords)
            return points # Return original if obs_map is also None or too few points

        # 1. 首先进行初步简化以减少点数 (Shapely LineString assumed)
        # Ensure you have shapely installed: pip install shapely
        try:
            from shapely.geometry import LineString
        except ImportError:
            print("Shapely library is not installed. Skipping initial simplification.")
            simplified_points = list(points) # Use original points if Shapely is not available
        else:
            line = LineString(points)
            simplified_line = line.simplify(tolerance, preserve_topology=False)
            simplified_points = list(simplified_line.coords)
            if not simplified_points: # Handle case where simplification results in no points
                simplified_points = [points[0], points[-1]] if len(points) >=2 else list(points)


        # 2. 自适应路径检查和修复
        if not simplified_points: # If simplification somehow failed
             return [points[0], points[-1]] if len(points) >=2 else list(points)

        valid_points = [points[0]]  # 始终保留起点 (original start point)
        
        # Ensure the first point of simplified_points aligns with points[0] or is valid
        # For simplicity, we assume simplified_points[0] is close to points[0]
        # If simplified_points[0] is not points[0], we might need to adjust valid_points initialization
        # or ensure the first point of simplified_points is itself valid.
        # Let's use the first point from the original path as the guaranteed start.

        for i in range(len(simplified_points)): # Iterate through all simplified points including the first
            current_point_tuple = simplified_points[i]
            x_coord, y_coord = int(current_point_tuple[0]), int(current_point_tuple[1])
            
            # Check if current_point_tuple is valid (not an obstacle cell)
            current_point_is_valid_cell = (
                0 <= x_coord < self.max_x and 
                0 <= y_coord < self.max_y and 
                obs_map[x_coord][y_coord] != 255 and 
                obs_map[x_coord][y_coord] != 0
            )
            
            # If current_point_tuple is in an obstacle, try to find a nearby valid cell
            if not current_point_is_valid_cell:
                found_valid_replacement_for_current = False
                # Search 5x5 (or other defined range)
                for dx_search in range(-2, 3): 
                    for dy_search in range(-2, 3):
                        if dx_search == 0 and dy_search == 0: continue # Skip self
                        nx, ny = x_coord + dx_search, y_coord + dy_search
                        if (0 <= nx < self.max_x and 0 <= ny < self.max_y and 
                            obs_map[nx][ny] != 255 and obs_map[nx][ny] != 0):
                            current_point_tuple = (float(nx), float(ny)) # Update current_point_tuple
                            x_coord, y_coord = nx, ny # Update integer coords for checks
                            current_point_is_valid_cell = True
                            found_valid_replacement_for_current = True
                            break
                    if found_valid_replacement_for_current:
                        break
                
                # If no valid replacement found nearby, skip this simplified point
                if not current_point_is_valid_cell:
                    continue 
            
            # current_point_tuple is now guaranteed to be a valid cell or was skipped.
            # Ensure we don't add duplicate points if current_point_tuple is same as last valid_point
            if valid_points and \
               math.isclose(current_point_tuple[0], valid_points[-1][0]) and \
               math.isclose(current_point_tuple[1], valid_points[-1][1]):
                continue

            prev_point_tuple = valid_points[-1]
            
            # Check direct line from prev_point_tuple to current_point_tuple
            line_segment_points = bresenham_line(
                int(prev_point_tuple[0]), int(prev_point_tuple[1]), 
                x_coord, y_coord # Use integer coords of (potentially adjusted) current_point_tuple
            )
            is_direct_segment_valid = True
            for lx, ly in line_segment_points:
                if not (0 <= lx < self.max_x and 0 <= ly < self.max_y and 
                        (obs_map[lx][ly] != 255 and obs_map[lx][ly] != 0)): # Check if point is on obstacle
                    is_direct_segment_valid = False
                    break
            
            if is_direct_segment_valid:
                valid_points.append(current_point_tuple)
            else:
                # Path prev_point_tuple -> current_point_tuple is blocked. Try to repair.
                # Find corresponding indices in the original path (this part is heuristic)
                original_idx_start = 0
                original_idx_end = len(points) - 1
                
                # Find prev_point_tuple in original path (approximate)
                min_dist_start = float('inf')
                for j, p_orig in enumerate(points):
                    dist = math.hypot(p_orig[0] - prev_point_tuple[0], p_orig[1] - prev_point_tuple[1])
                    if dist < min_dist_start:
                        min_dist_start = dist
                        original_idx_start = j
                
                # Find current_point_tuple in original path (approximate)
                min_dist_end = float('inf')
                for j, p_orig in enumerate(points):
                    # Search from original_idx_start onwards for efficiency, assuming order
                    if j < original_idx_start: continue 
                    dist = math.hypot(p_orig[0] - current_point_tuple[0], p_orig[1] - current_point_tuple[1])
                    if dist < min_dist_end:
                        min_dist_end = dist
                        original_idx_end = j
                
                # Ensure start < end
                if original_idx_start > original_idx_end:
                    original_idx_start, original_idx_end = original_idx_end, original_idx_start
                if original_idx_start == original_idx_end and original_idx_end < len(points) -1 :
                    original_idx_end +=1 # Ensure there's a range if possible
                elif original_idx_start == original_idx_end and original_idx_start > 0:
                     original_idx_start -=1


                intermediate_nodes = self.find_valid_midpoints(
                    points, original_idx_start, original_idx_end, 
                    prev_point_tuple, current_point_tuple, obs_map
                )
                
                if intermediate_nodes:
                    # intermediate_nodes form a valid chain from prev_point_tuple to current_point_tuple
                    # (implicitly, prev_point_tuple -> intermediate_nodes[0] is fine,
                    #  and intermediate_nodes[-1] -> current_point_tuple is fine)
                    valid_points.extend(intermediate_nodes)
                    # current_point_tuple itself was already validated (current_point_is_valid_cell)
                    # and is the target of the chain from find_valid_midpoints
                    valid_points.append(current_point_tuple) 
                else:
                    # FIX: If find_valid_midpoints returns empty, it failed to find a repair.
                    # Do not add current_point_tuple as the path to it is blocked.
                    # The path will continue from prev_point_tuple to the *next* simplified point.
                    pass 
        
        # Ensure the original endpoint is included if it's valid and different from the last point
        if points and len(points) > 0:
            original_end_point = points[-1]
            oex, oey = int(original_end_point[0]), int(original_end_point[1])
            is_original_end_point_valid_cell = (
                0 <= oex < self.max_x and 0 <= oey < self.max_y and
                obs_map[oex][oey] != 255 and obs_map[oex][oey] != 0
            )

            if valid_points and is_original_end_point_valid_cell:
                last_added_point = valid_points[-1]
                if not (math.isclose(last_added_point[0], original_end_point[0]) and \
                        math.isclose(last_added_point[1], original_end_point[1])):
                    
                    # Check path from last_added_point to original_end_point
                    line_to_original_end = bresenham_line(
                        int(last_added_point[0]), int(last_added_point[1]),
                        oex, oey
                    )
                    can_connect_to_original_end = True
                    for lx, ly in line_to_original_end:
                        if not (0 <= lx < self.max_x and 0 <= ly < self.max_y and
                                (obs_map[lx][ly] != 255 and obs_map[lx][ly] != 0)):
                            can_connect_to_original_end = False
                            break
                    
                    if can_connect_to_original_end:
                        valid_points.append(original_end_point)
                    else:
                        # Try to repair path to original_end_point
                        # This is a simplified version, could use find_valid_midpoints again if complex
                        # For now, if direct fails, we might end up not reaching exact original endpoint
                        # if it's separated by an obstacle from the current valid path end.
                        # A more robust solution might involve a dedicated call to find_valid_midpoints
                        # to connect valid_points[-1] to original_end_point.
                        pass # Cannot connect, path might end slightly short of original goal.
            elif not valid_points and is_original_end_point_valid_cell: # e.g. if all simplified points were skipped
                 valid_points.append(original_end_point)


        # 4. 最后一次清理 - 移除冗余点 (collinear points on a clear path)
        if len(valid_points) > 2:
            final_cleaned_path = [valid_points[0]]
            i = 1
            while i < len(valid_points) - 1:
                pt_a = final_cleaned_path[-1]
                pt_b = valid_points[i] # Candidate for removal
                pt_c = valid_points[i+1]
                
                # Check if pt_a can connect directly to pt_c
                line_ac_points = bresenham_line(
                    int(pt_a[0]), int(pt_a[1]), 
                    int(pt_c[0]), int(pt_c[1])
                )
                is_ac_path_valid = True
                for lx, ly in line_ac_points:
                    if not (0 <= lx < self.max_x and 0 <= ly < self.max_y and 
                            (obs_map[lx][ly] != 255 and obs_map[lx][ly] != 0)):
                        is_ac_path_valid = False
                        break
                
                if is_ac_path_valid:
                    # pt_b is redundant, skip it. The next point to check against final_cleaned_path[-1] will be pt_c
                    # Effectively, we are checking if we can extend from final_cleaned_path[-1] to valid_points[i+1]
                    # The loop structure needs adjustment for proper removal.
                    # A better way for cleanup:
                    i += 1 # Move to check the next point as pt_b in the next iteration
                           # This means pt_b (current valid_points[i]) is kept for now.
                           # The original logic was: if valid, pop(i), else i++.
                           # This can be tricky. Let's use a safer build-up for final_cleaned_path.
                else:
                    # Cannot remove pt_b, so add it.
                    final_cleaned_path.append(pt_b)
                    i += 1
            
            # Add the last point
            if len(valid_points) > 0 : # Check if valid_points is not empty
                 if not final_cleaned_path or \
                    not (math.isclose(final_cleaned_path[-1][0], valid_points[-1][0]) and \
                         math.isclose(final_cleaned_path[-1][1], valid_points[-1][1])):
                    final_cleaned_path.append(valid_points[-1])
            
            valid_points = final_cleaned_path

        # Ensure at least start and end if possible
        if not valid_points and points:
            if len(points) == 1: return [points[0]]
            if len(points) >= 2:
                # Simplified logic: just return original start and end if everything else failed
                # but ideally, check their validity and connection
                start_p = points[0]
                end_p = points[-1]
                sx_int, sy_int = int(start_p[0]), int(start_p[1])
                ex_int, ey_int = int(end_p[0]), int(end_p[1])

                start_valid = (0 <= sx_int < self.max_x and 0 <= sy_int < self.max_y and obs_map[sx_int][sy_int] != 255 and obs_map[sx_int][sy_int] != 0)
                end_valid = (0 <= ex_int < self.max_x and 0 <= ey_int < self.max_y and obs_map[ex_int][ey_int] != 255 and obs_map[ex_int][ey_int] != 0)

                if start_valid and end_valid:
                    return [start_p, end_p]
                elif start_valid:
                    return [start_p]
                elif end_valid:
                    return [end_p]
                else:
                    return [] # Cannot even provide valid start/end

        return valid_points

    def find_valid_midpoints(self, points, start_idx, end_idx, start_node_tuple, end_node_tuple, obs_map):
        """
        使用二分法递归查找有效的中间点.
        Returns a list of intermediate points from the original 'points' list that form a valid path
        between start_node_tuple and end_node_tuple.
        start_node_tuple and end_node_tuple themselves are NOT included in the returned list.
        """
        # MODIFIED: Check direct connection between start_node_tuple and end_node_tuple first.
        # If this direct path is clear, no intermediate points from 'points' list are needed.
        line_direct_segment = bresenham_line(
            int(start_node_tuple[0]), int(start_node_tuple[1]),
            int(end_node_tuple[0]), int(end_node_tuple[1])
        )
        is_direct_path_clear = True
        for lx_direct, ly_direct in line_direct_segment:
            if not (0 <= lx_direct < self.max_x and 0 <= ly_direct < self.max_y and
                    obs_map[lx_direct][ly_direct] != 255 and obs_map[lx_direct][ly_direct] != 0):
                is_direct_path_clear = False
                break
        if is_direct_path_clear:
            return [] # No intermediate points needed

        # MODIFIED: If the segment of original points is too small to pick a distinct middle point.
        if end_idx - start_idx < 2: # e.g., start_idx=0, end_idx=1. No mid_idx possible.
            return [] # Cannot find further intermediate points from original list. Direct path already failed.

        mid_idx = (start_idx + end_idx) // 2
        
        # This is the candidate midpoint from the original high-resolution path.
        mid_point_original_candidate = points[mid_idx] 
        
        # This point, if used, must correspond to a valid (non-obstacle) map cell.
        # Try to use mid_point_original_candidate as is, or find a close valid substitute.
        point_to_insert_in_chain = None 
        
        mx_orig, my_orig = int(mid_point_original_candidate[0]), int(mid_point_original_candidate[1])
        
        is_original_mid_cell_valid = (
            0 <= mx_orig < self.max_x and 0 <= my_orig < self.max_y and
            obs_map[mx_orig][my_orig] < 230 and obs_map[mx_orig][my_orig] > 0
        )

        if is_original_mid_cell_valid:
            point_to_insert_in_chain = mid_point_original_candidate
        else:
            # Original mid-point is an obstacle. Search for a nearby valid cell.
            # FIX: Search radius was (-2,3), making it 5x5. User's code was (-2,3) for dx, (-2,3) for dy.
            # Let's use a small search, e.g., 3x3 (-1 to 1) or keep 5x5.
            # The user code had a nested loop for this search.
            found_valid_substitute_mid = False
            for dx_mid_search in range(-2, 3): # Search 5x5 area
                for dy_mid_search in range(-2, 3):
                    if dx_mid_search == 0 and dy_mid_search == 0: continue # Skip the invalid cell itself
                    
                    nmx, nmy = mx_orig + dx_mid_search, my_orig + dy_mid_search
                    if (0 <= nmx < self.max_x and 0 <= nmy < self.max_y and
                        obs_map[nmx][nmy] != 255 and obs_map[nmx][nmy] != 0):
                        point_to_insert_in_chain = (float(nmx), float(nmy)) # Use float for consistency
                        found_valid_substitute_mid = True
                        break
                if found_valid_substitute_mid:
                    break
        
        if point_to_insert_in_chain is None:
            # FIX: Midpoint from original path is an obstacle, and no valid substitute found.
            # This means this bisection attempt (via points[mid_idx]) has failed.
            # Return empty list to signal that no valid intermediate path was found through this midpoint.
            return []

        # Recursively find paths for the two new sub-segments:
        # 1. From start_node_tuple to point_to_insert_in_chain
        # 2. From point_to_insert_in_chain to end_node_tuple
        left_chain_nodes = self.find_valid_midpoints(
            points, start_idx, mid_idx, 
            start_node_tuple, point_to_insert_in_chain, 
            obs_map
        )
        
        right_chain_nodes = self.find_valid_midpoints(
            points, mid_idx, end_idx, 
            point_to_insert_in_chain, end_node_tuple, 
            obs_map
        )
        
        # The path is constructed by:
        # path_from_start_to_left_chain_end -> point_to_insert_in_chain -> path_from_right_chain_start_to_end
        # The returned list is the sequence of these intermediate points.
        return left_chain_nodes + [point_to_insert_in_chain] + right_chain_nodes
    ########### visualize utils ##############
    def visualize_init(self):
        """
        Visualizes the current state of the A* exploration.
        """
        # self.cmap = mcolors.ListedColormap(['white', 'green', 'gray', 'black'])  # Colors for 0, between 1-254, 2, 255
        self.cmap = mcolors.ListedColormap(['white', '#C1FFC1', 'gray', 'black']) # #C1FFC1 denotes the light green
        bounds = [0, 1, 3, 254, 256]  # Boundaries for the colors
        self.norm = mcolors.BoundaryNorm(bounds, self.cmap.N)
        
        figsize_x = self.x_width / 20
        figsize_y = self.y_width / 20
        self.fig = plt.figure(2, figsize=(figsize_x, figsize_y))  # Create and store a specific figure
        self.ax = self.fig.add_subplot(111)  # Add a subplot to the figure
        # self.ax.set_title("A* Path Planning")
        # self.ax.grid(True)
        self.ax.axis("equal")
        self.ax.set_xlim(0, self.max_x) 
        self.ax.set_ylim(0, self.max_y) 

        self.major_ticks_x = np.linspace(0, self.max_x, 40)
        # self.minor_ticks_x = np.linspace(0, self.max_x, 40)
        self.major_ticks_y = np.linspace(0, self.max_y, 40)
        self.windows_head = False

    def vis_path(self, obs_map, sx, sy, gx, gy, points, img_save_path, legend=True):
        obs_map_draw = obs_map.transpose(1,0) # transpose the map to match the plot
        # self.image_display.set_data(obs_map_draw)
        plt.imshow(obs_map_draw, cmap=self.cmap, norm=self.norm, aspect='auto')
        plt.plot(sx, sy, "or", label="start")
        plt.plot(gx, gy, "xr", label="end")
        plt.plot([x[0] for x in points], [x[1] for x in points], "-r")
        
        if legend:
            plt.legend()
            plt.grid()
        
        plt.savefig(img_save_path, pad_inches=0, bbox_inches='tight', dpi=100)
        # log.info("Path has been saved to {}".format(img_save_path))
        if self.windows_head:
            plt.show(block=False)
            plt.pause(0.001)
        
        plt.clf()
        plt.close()
    
    def figure_clear(self, for_llm=False):
        self.ax.clear()
        self.ax.set_xlim(0, self.max_x) 
        self.ax.set_ylim(0, self.max_y)

    def vis_whole_path(self, obs_map, img_save_path, whole_path, for_llm=False, vis_latest_path=True, start_position=None):
        """可视化整个路径，包括起点、路径点和终点
        
        Args:
            obs_map: 障碍物地图
            img_save_path: 图片保存路径
            whole_path: 完整路径点列表
            for_llm: 是否为LLM生成可视化
            vis_latest_path: 是否显示最新路径
            start_position: 起始位置
        """
        plt.close('all')
        
        # 创建与地图大小完全一致的图形
        height, width = obs_map.shape
        dpi = 100
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        ax = fig.add_subplot(111)
        
        # 设置坐标轴范围和比例
        ax.set_xlim(-0.5, width-0.5)
        ax.set_ylim(height-0.5, -0.5)  # 翻转y轴使得原点在左上角
        ax.set_aspect('equal')
        
        # 移除边框和轴
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 绘制地图
        ax.imshow(obs_map, cmap=self.cmap, norm=self.norm)
        
        # 绘制起始点
        if start_position is not None:
            ax.plot(start_position[1], start_position[0], "ob", markersize=8,
                    label=f"start position (0):({start_position[0]:.2f},{start_position[1]:.2f})")
            if for_llm:
                ax.text(start_position[1]-0.5, start_position[0]+0.5, '0', 
                        fontsize=12, color='blue', ha='right')
        
        # 处理路径点
        whole_points = whole_path if vis_latest_path else whole_path[:-1]
        
        # 绘制路径点和连线
        for i, points in enumerate(whole_points):
            if len(points) == 0:
                continue
            
            # 绘制路径点
            if i > 0:
                point = points[0]
                ax.plot(point[1], point[0], "ob", markersize=8,
                        label=f"visited waypoint ({i}):({point[0]:.2f},{point[1]:.2f})")
                if for_llm:
                    ax.text(point[1], point[0], str(i), 
                            fontsize=12, color='blue', ha='right')
            
            # 绘制路径线
            x_coords = [point[1] for point in points]
            y_coords = [point[0] for point in points]
            ax.plot(x_coords, y_coords, "-r", linewidth=2)
        
        # 绘制终点
        if len(whole_points) >= 1 and len(whole_points[-1]) >= 1:
            end_point = whole_points[-1][-1]
            ax.plot(end_point[1], end_point[0], "ob", markersize=8,
                    label=f"current position ({len(whole_points)}):({end_point[0]:.2f},{end_point[1]:.2f})")
            if for_llm:
                ax.text(end_point[1], end_point[0], str(len(whole_points)), 
                        fontsize=12, color='blue', ha='right')
        
        # 为LLM模式添加网格和标签
        if for_llm:
            ax.set_xticks(np.linspace(0, width, 20))
            ax.set_yticks(np.linspace(0, height, 20))
            ax.grid(True, linewidth=0.75, color='gray', alpha=0.75)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
        
        # 确保图像边界紧凑
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # 保存图片
        plt.savefig(img_save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        
        if self.windows_head:
            plt.show(block=False)
            plt.pause(0.001)
        
        plt.close()

def bresenham_line(x0, y0, x1, y1):
    """Bresenham's Line Algorithm to get points on a line between (x0, y0) and (x1, y1)."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1 or len(points) > abs(dx) + abs(dy) + 2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points