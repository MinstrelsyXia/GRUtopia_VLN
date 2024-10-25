import numpy as np
import pyvisgraph as vg
from vlmaps.vlmaps.utils.navigation_utils import build_visgraph_with_obs_map, plan_to_pos_v2
from typing import Tuple, List, Dict
import cv2

class Navigator:
    def __init__(self):
        pass

    def build_visgraph(self, obstacle_map: np.ndarray, rowmin: float = 0, colmin: float = 0, vis: bool = False):
        self.obs_map = obstacle_map
        self.visgraph = build_visgraph_with_obs_map(obstacle_map, vis=vis)
        self.rowmin = rowmin
        self.colmin = colmin
        

    def plan_to(
        self, start_full_map: Tuple[float, float], goal_full_map: Tuple[float, float], vis: bool = False, navigable_map_visual = None,save_path ='tmp2/tmp/planned_path.jpg'
    ) -> List[List[float]]:
        """
        Take full map start (row, col) and full map goal (row, col) as input
        Return a list of full map path points (row, col) as the palnned path
        """
        start = self._convert_full_map_pos_to_cropped_map_pos(start_full_map)
        goal = self._convert_full_map_pos_to_cropped_map_pos(goal_full_map)
        if self._check_if_start_in_graph_obstacle(start):
            self._rebuild_visgraph(start, vis)
        paths = plan_to_pos_v2(start_full_map, goal_full_map, self.obs_map, self.visgraph, vis,navigable_map_visual)
        # paths = self.shift_path(paths, self.rowmin, self.colmin)
        if vis == True:
            self.visualize_path(start = start_full_map, goal = goal_full_map, obstacles = navigable_map_visual, path = paths,save_path=save_path)
        return paths
    
    def visualize_path(self,start,goal,obstacles,path,save_path):
        # obs_map_vis = (obstacles[:, :, None] * 255).astype(np.uint8)
        # obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])

        # obs_map_vis = (obstacles[:, :, None] * 255).astype(np.uint8)
        # obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
        obs_map_vis = obstacles

        for i, point in enumerate(path):
            subgoal = (int(point[1]), int(point[0]))
            print(i, subgoal)
            obs_map_vis = cv2.circle(obs_map_vis, subgoal, 1, (255, 0, 0), -1)
            if i > 0:
                cv2.line(obs_map_vis, last_subgoal, subgoal, (255, 0, 0), 1)
            last_subgoal = subgoal
        obs_map_vis = cv2.circle(obs_map_vis, (int(start[1]), int(start[0])), 2, (0, 255, 0), -1)
        obs_map_vis = cv2.circle(obs_map_vis, (int(goal[1]), int(goal[0])), 2, (0, 0, 255), -1)
        # cv2.imshow("planned path", obs_map_vis)
        # cv2.waitKey(1)
        
        cv2.imwrite(save_path, obs_map_vis)


    def shift_path(self, paths: List[List[float]], row_shift: int, col_shift: int) -> List[List[float]]:
        shifted_paths = []
        for point in paths:
            shifted_paths.append([point[0] + row_shift, point[1] + col_shift])
        return shifted_paths

    def _check_if_start_in_graph_obstacle(self, start: Tuple[float, float]):
        startvg = vg.Point(start[0], start[1])
        poly_id = self.visgraph.point_in_polygon(startvg)
        if poly_id != -1 and self.obs_map[int(start[0]), int(start[1])] == 1:
            return True
        return False

    def _rebuild_visgraph(self, start: Tuple[float, float], vis: bool = False):
        self.visgraph = build_visgraph_with_obs_map(
            self.obs_map, use_internal_contour=True, internal_point=start, vis=vis
        )

    def _convert_full_map_pos_to_cropped_map_pos(self, full_map_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        full_map_pos: (row, col) in full map
        Return (row, col) in cropped_map
        """
        print("full_map_pos: ", full_map_pos)
        print("self.rowmin: ", self.rowmin)
        print("self.colmin: ", self.colmin)
        return [full_map_pos[0] - self.rowmin, full_map_pos[1] - self.colmin]

    def _convert_cropped_map_pos_to_full_map_pos(self, cropped_map_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        cropped_map_pos: (row, col) in cropped map
        Return (row, col) in full map
        """
        return [cropped_map_pos[0] + self.rowmin, cropped_map_pos[1] + self.colmin]


    def check_path_blocked(self,start, goal):
        '''
        start, goal: (row, col) in full map
        grid: 2D list or array representing the map, where 0 is free and 1 is blocked
        '''
        line_points = bresenham_line(start[0], start[1], goal[0], goal[1])
        
        for point in line_points:
            row, col = point
            if self.obs_map[row][col] == 1:  # 1 indicates a blocked cell
                return False
        
        return True


def bresenham_line(x0, y0, x1, y1):
    """Bresenham's Line Algorithm to get points on a line between (x0, y0) and (x1, y1)."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

        
