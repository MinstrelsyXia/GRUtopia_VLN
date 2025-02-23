
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
        max_step=10000
    ):
        """
        Initialize grid map for a star planning.
        Note that this class does not consider the robot's radius. So the given obs_map should be expanded
        """
        self.resolution = 1
        self.max_step = max_step
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = map_width, map_height
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = map_width, map_height
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        self.visualize_init()
        self._traj_vis = TrajectoryVisualizer(np.array([0,0]), self.pixels_per_meter)
        
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
    ) -> List[List[float]]:
        """
        Take full map start (row, col) and full map goal (row, col) as input
        Return a list of full map path points (row, col) as the palnned path
        """
        if self._check_if_start_in_graph_obstacle(start,obs_map):
            # self._rebuild_visgraph(start, vis)
            self.find_nearest_free_node(obs_map, start)
        paths = self.planning(start[0], start[1], goal[0], goal[1], obs_map)
        # paths = self.shift_path(paths, self.rowmin, self.colmin)
        if vis == True:
            obs_map_vis = self._traj_vis.draw_trajectory(navigable_map_visual, paths, 0)
            cv2.imwrite(save_path, obs_map_vis)
            new_save_path = os.path.join(
            os.path.dirname(os.path.dirname(save_path)),  # 上级目录
                'planned_path.jpg'
            )
            cv2.imwrite(new_save_path, obs_map_vis)

        return paths
    
    def _check_if_start_in_graph_obstacle(self, start: Tuple[float, float],obs_map: np.ndarray):
        if obs_map[start[0], start[1]] == 1:
            return True
        return False
    
    
    def check_path_blocked(self,start, goal):
        '''
        start, goal: (row, col) in full map
        grid: 2D list or array representing the map, where 0 is free and 1 is blocked
        '''
        goal = [int(goal[0]), int(goal[1])]
        line_points = bresenham_line(start[0], start[1], goal[0], goal[1])
        
        for point in line_points:
            row, col = point
            if self.obs_map[row][col] == 1:  # 1 indicates a blocked cell
                return False
        
        return True
        



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
    
    def find_nearest_free_node(self, obs_map, goal_node):
        if obs_map[goal_node.x, goal_node.y] != 255:
            return goal_node  # Goal node is not in an obstacle

        free_nodes = np.argwhere(obs_map != 255)
        goal_position = np.array([goal_node.x, goal_node.y])

        distances = np.linalg.norm(free_nodes - goal_position, axis=1)
        nearest_free_node_index = np.argmin(distances)
        nearest_free_node = free_nodes[nearest_free_node_index]

        new_goal_node = self.Node(self.calc_xy_index(nearest_free_node[0], self.min_x), self.calc_xy_index(nearest_free_node[1], self.min_y), 0.0, -1, 0)
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
            rx: x position list of the final path
            ry: y position list of the final path
        
        obs_map:
            255: obstacle
            2: free area, cost = 0
            0: unexplored area, cost = 240
            others: dilated area, with larger cost

        """
        if coord == 'xy':
            start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                                self.calc_xy_index(sy, self.min_y), 0.0, -1)
            goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                                self.calc_xy_index(gy, self.min_y), 0.0, -1)
                    
            motion = self.get_motion_model()
            reason = None
            if obs_map[goal_node.x, goal_node.y] == 255:
                reason = 'goal_in_obstacle'
                # return [], [], False, reason
                new_goal_node = self.find_nearest_free_node(obs_map, goal_node)
                goal_node = new_goal_node

            open_set, closed_set = dict(), dict()
            open_set[self.calc_grid_index(start_node)] = start_node
        if coord == 'obs':
            start_node = self.Node(sx,sy,0.0,-1)
            goal_node = self.Node(gx,gy,0.0,-1)
            motion = self.get_motion_model()
            reason = None
            if obs_map[goal_node.x, goal_node.y] == 255:
                reason = 'goal_in_obstacle'
                # return [], [], False, reason
                new_goal_node = self.find_nearest_free_node(obs_map, goal_node)
                goal_node = new_goal_node

            open_set, closed_set = dict(), dict()
            open_set[self.calc_grid_index(start_node)] = start_node


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

        if len(points_list) > 1:
            points = self.simplify_path(points_list)
            points.append((gx, gy))
        else:
            # log.warning(f"Path planning results only contain {len(points_list)} points.")
            points = []
            points.append((gx, gy))
        self.vis_whole_path(obs_map, 'tmp/whole_path.png', [points], for_llm=True, vis_latest_path=True,start_position = [sx,sy])

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
            for dy in range(-5, 5):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.max_x and 0 <= ny < self.max_y:  # 确保在地图范围内
                    if obs_map[nx][ny] == 0:
                        cost = 240
                    elif obs_map[nx][ny] == 2:
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
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if obs_map[node.x][node.y] == 255:
            return False

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

    def simplify_path(self, points, tolerance=0.01):
        ''' The tolerance sets sampling distance. The smaller the tolerance, the more points in the simplified line.
        '''
        line = LineString(points)
        simplified_line = line.simplify(tolerance, preserve_topology=False)
        return list(simplified_line.coords)
    
    ########### visualize utils ##############
    def visualize_init(self):
        """
        Visualizes the current state of the A* exploration.
        """
        # self.cmap = mcolors.ListedColormap(['white', 'green', 'gray', 'black'])  # Colors for 0, between 1-254, 2, 255
        self.cmap = mcolors.ListedColormap(['white', '#C1FFC1', 'gray', 'black']) # #C1FFC1 denotes the light green
        bounds = [0, 1, 3, 254, 256]  # Boundaries for the colors
        self.norm = mcolors.BoundaryNorm(bounds, self.cmap.N)
        
        self.fig = plt.figure(2, figsize=(10,10))  # Create and store a specific figurel
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

    def vis_whole_path(self, obs_map, img_save_path, whole_path, for_llm=False, vis_latest_path=True,start_position=None):
        # obs_map_draw = obs_map.transpose(1,0) # transpose the map to match the plot
        self.figure_clear()
        obs_map_draw = obs_map
        self.ax.imshow(obs_map_draw, cmap=self.cmap, norm=self.norm, aspect='auto')
        self.ax.plot(start_position[1], start_position[0], "ob", label="start position (0):(%.2f,%.2f)"%(start_position[0], start_position[1]))
        if for_llm:
            self.ax.text(start_position[1]-0.5, start_position[0]+0.5, '0', fontsize=12, color='blue', ha='right')
        
        if not vis_latest_path:
            whole_points = whole_path[:-1]
        else:
            whole_points = whole_path

        for i, points in enumerate(whole_points):
            for j, point in enumerate(points):
                if i == 0 and j == 0:
                    continue
                if j == 0:
                    self.ax.plot(point[1], point[0], "ob", label="visited waypoint (%d):(%.2f,%.2f)"%(i, point[1], point[0]))
                    if for_llm:
                        self.ax.text(point[1], point[0], str(i), fontsize=12, color='blue', ha='right')
        
        # Ensure the entire path is drawn by connecting all points
        for points in whole_points:
            x_coords = [point[1] for point in points]
            y_coords = [point[0] for point in points]
            self.ax.plot(x_coords, y_coords, "-r")
        
        if len(whole_points) >= 1:
            end_point = whole_points[-1][-1]
            self.ax.plot(end_point[1], end_point[0], "ob", label="current position (%d):(%.2f,%.2f)"%(len(whole_points), end_point[1], end_point[0]))
            if for_llm:
                self.ax.text(end_point[1], end_point[0], str(len(whole_points)), fontsize=12, color='blue', ha='right')

        if for_llm:
            # Set major and minor ticks
            self.ax.set_xticks(self.major_ticks_x)
            self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation=40)
            self.ax.set_yticks(self.major_ticks_y)
            # self.ax.set_xticks(self.minor_ticks_x, minor=True)
            # self.ax.set_yticks(self.minor_ticks_y, minor=True)

            # self.ax.grid(which='minor', linewidth='0.75', color='gray', alpha=0.6)
            self.ax.grid(which='major', linewidth='0.75', color='gray', alpha=0.75)

            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')

        self.ax.legend()

        self.ax.figure.savefig(img_save_path, pad_inches=0, bbox_inches='tight', dpi=100)
        # log.info("Whole path has been saved to {}".format(img_save_path))
        if self.windows_head:
            plt.show(block=False)
            plt.pause(0.001)

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