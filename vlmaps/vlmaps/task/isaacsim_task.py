from pathlib import Path
import json
from typing import Dict, List, Tuple, Union

from omegaconf import DictConfig
import numpy as np
import hydra
import matplotlib.pyplot as plt
import cv2
import os
import traceback
# from vlmaps.vlmaps.task.habitat_task import HabitatTask
# from vlmaps.vlmaps.utils.habitat_utils import agent_state2tf, get_position_floor_objects
# from vlmaps.vlmaps.utils.navigation_utils import get_dist_to_bbox_2d
# from vlmaps.vlmaps.utils.habitat_utils import display_sample

# def compute_metrics(self, fail_reason=''):
#     """计算VLN任务的评估指标
    
#     Args:
#         infos: 当前episode的信息,包含当前位置、目标位置等
        
#     Returns:
#         metrics: 包含各项指标的字典
#     """
#     metrics = {}
    
#     # 获取当前位置和目标位置
#     current_position = self.get_robot_poses()[self.env_idx][0]
#     goal_position = self.data_item['reference_path'][-1]

#     # 计算Navigation Error (NE) - 当前位置到目标的欧氏距离
#     ne = np.linalg.norm(current_position[:2] - goal_position[:2])
#     metrics['NE'] = ne 
    
#     # 计算Success Rate (SR) - 是���到达目标点
#         # 成功阈值通常设为3米
#     success = ne < self.success_distance
#     metrics['success'] = float(success)
    
#     # 计算Oracle Success Rate (OSR) - 轨迹中是否有点达到目标
#     min_distance = ne if ne < self.shortest_to_goal_distance else self.shortest_to_goal_distance # 如果需要轨迹中最小距离,需要在step中记录
#     metrics['osr'] = float(min_distance < self.success_distance)
    
#     # 计算Trajectory Length (TL) - 轨迹总长度
#     metrics['TL'] = self.current_path_length

#     # 计算SPL (Success weighted by Path Length)
#     if metrics['TL'] > 0:
#         spl = metrics['success'] * self.shortest_path_length / max(
#             metrics['TL'], self.shortest_path_length
#         )
#     else:
#         spl = 0
#     metrics['spl'] = spl
    
#     # 计算NDTW (Normalized Dynamic Time Warping)
#     # 计算当前轨迹与参考轨迹之间的DTW距离
#     # DTW参数
#     dtw_threshold = self.success_distance  # 通常设置为success_distance
#     # 计算路径间的累积DTW距离
#     dtw_distance = 0.0
#     trajectory = []
#     if len(self.pred_traj_list[self.env_idx]) > 0:
#         trajectory = np.array(self.pred_traj_list[self.env_idx])[:,:2] # 只取x,y坐标
#         reference_path = np.array(self.data_item['reference_path'])[:,:2]

#         for point in trajectory:
#             # 找到参考路径上最近的点
#             min_dist = float('inf')
#             for ref_point in reference_path:
#                 dist = np.linalg.norm(point - ref_point)
#                 min_dist = min(min_dist, dist)
#             # 累加DTW距离,使用高斯函数进行归一化
#             dtw_distance += np.exp(-min_dist**2 / (2 * dtw_threshold**2))
        
#     # 归一化DTW得分
#     ndtw = dtw_distance / len(trajectory) if len(trajectory) > 0 else 0.0
#     metrics['ndtw'] = ndtw
    
#     # 计算SDTW (Success weighted Dynamic Time Warping)
#     # metrics['sdtw'] = metrics['success'] * metrics['ndtw']
    
#     # 其他可能的指标
#     metrics['steps'] = self.current_step_list[self.env_idx]  # 步数
#     metrics['episode_id'] = self.data_item['episode_id']  # episode ID
#     metrics['trajectory_id'] = self.data_item['trajectory_id']  # 轨迹 ID

#     metrics['fail_reason'] = fail_reason
    
#     return [metrics] # batch size = 1


class IsaacSimSpatialGoalNavigationTask():
    def __init__(self,config):
        self.config = config
        


    def add_action_func(self,action_func):
        self.action_funcs.append(action_func)

    def get_past_path(self)->List[List[float]]:
        return np.array(self.simple_pos_list_all)
    
    def get_goals(self)->List[List[float]]:
        return np.array([list(goal) for goal in self.goals])
    
    def start_new_episode(self,step):
        # when fall, record as a new episode
        last_episode_info = {
            'step': step,
            'pos_list': np.array(self.pos_list),
        }
        self.pos_list_all.append(last_episode_info)
        self.pos_list = []

    def add_pos(self,pos):
        if pos[2] < self.ground_height-1.65 :
            return
        self.pos_list.append(list(pos))
        self.simple_pos_list_all.append(list(pos))
    
    def add_parsed_instruction(self,parsed_instruction):
        self.parsed_instruction = parsed_instruction

    def setup_task(self, item):
        # remember to transpose
        self.action_funcs = []
        self.pos_list = [] # xyz coord
        self.pos_list_all = [] # list of list
        self.simple_pos_list_all = [] # list of list
        self.instruction = item["instruction"]['instruction_text']
        self.goals = np.array(item["reference_path"])
        self.ground_height = np.mean(self.goals[:,2])
        self.episode_id = item['episode_id']
        self.scan = item['scan']
        self.n_subgoals_in_task = len(self.goals)
        self.success_subgoal_num = 0
        self.finished_subgoals = []
        self.distance_to_subgoals = []
        self.success = False
        self.success_distance = 3
        self.shortest_to_goal_distance = 999
        

    def calculate_metric(self,step):
        # 获取当前位置和目标位置
        metrics = {}
        current_position = self.simple_pos_list_all[-1][:2]
        goal_position = self.goals[-1][:2]
        # 计算Navigation Error (NE) - 当前位置到目标的欧氏距离
        ne = np.linalg.norm(current_position - goal_position)
        metrics['NE'] = ne
        # 计算Success Rate (SR) - 是否到达目标点
        success = ne < self.success_distance
        metrics['success'] = float(success)

        # 计算Oracle Success Rate (OSR) - 轨迹中是否有点达到目标
        min_distance = ne if ne < self.shortest_to_goal_distance else self.shortest_to_goal_distance # 如果需要轨迹中最小距离,需要在step中记录
        metrics['osr'] = float(min_distance < self.success_distance)
        
        # 计算Trajectory Length (TL) - 轨迹总长度
        tl = 0
        for episode in self.pos_list_all:
            if len(episode['pos_list']) > 0:
                tl += np.sum(np.sqrt(np.sum(np.diff(episode['pos_list'][:,:2], axis=0) ** 2, axis=1)))
        metrics['TL'] = float(tl)
        self.shortest_path_length = float(np.sum(np.sqrt(np.sum(np.diff(self.goals[:,:2], axis=0) ** 2, axis=1))))
        if metrics['TL'] > 0:
            spl = metrics['success'] * self.shortest_path_length / max(
                metrics['TL'], self.shortest_path_length
            )
        else:
            spl = 0
        metrics['spl'] = float(spl)

        # 计算NDTW (Normalized Dynamic Time Warping)
        # 计算当前轨迹与参考轨迹之间的DTW距离
        # DTW参数
        dtw_threshold = self.success_distance  # 通常设置为success_distance

        # 计算路径间的累积DTW距离
        dtw_distance = 0.0
        trajectory_count = 0  # 记录有效的trajectory数量

        for episode in self.pos_list_all:
            if len(episode['pos_list']) > 0:
                trajectory = np.array(episode['pos_list'])[:, :2]  # 只取x,y坐标
                reference_path = self.goals[:, :2]

                # 检查trajectory的完整性
                if len(trajectory) < 0:  # min_length是您定义的最小长度
                    continue  # 跳过不完整的episode

                for point in trajectory:
                    # 找到参考路径上最近的点
                    min_dist = float('inf')
                    for ref_point in reference_path:
                        dist = np.linalg.norm(point - ref_point)
                        min_dist = min(min_dist, dist)
                    # 累加DTW距离,使用高斯函数进行归一化
                    dtw_distance += np.exp(-min_dist**2 / (2 * dtw_threshold**2))
            
            trajectory_count += 1  # 记录有效的trajectory数量

        # 归一化DTW得分
        ndtw = dtw_distance / trajectory_count if trajectory_count > 0 else 0.0
        metrics['ndtw'] = float(ndtw)


        self.finished_subgoals = []
        self.distance_to_subgoals = []
        for goal_3 in self.goals:
            goal = goal_3[:2]
            min_dist = np.inf
            for pos_3 in self.simple_pos_list_all:
                pos = pos_3[:2]
                if (np.linalg.norm(np.array(goal)-np.array(pos)) < min_dist):
                    min_dist = np.linalg.norm(np.array(goal)-np.array(pos))
                if np.linalg.norm(np.array(goal)-np.array(pos)) < self.config["nav"]["valid_range"]:
                    self.finished_subgoals.append(goal)                
                    break
            self.distance_to_subgoals.append(min_dist)
        if len(self.finished_subgoals) == len(self.goals):
            self.success = True
        self.subgoal_success_rate = len(self.finished_subgoals) / len(self.goals)
        # distances = np.sqrt(np.sum(np.diff(self.pos_list, axis=0) ** 2, axis=1))

        # # Sum up all the distances to get the total path length
        # gt_length = np.sqrt(np.sum(np.diff(self.goals, axis=0)** 2, axis=1))
        # self.spl = np.sum(gt_length) / (np.sum(distances) + 1e-6)
        # 其他可能的指标
        metrics['steps'] = step  # 步数

        self.metrics = metrics

    
    def save_single_task_metric(
        self,
        save_path: Union[Path, str],
    ):
        results_dict = self.metrics
        results_dict["episode_id"] = self.episode_id
        results_dict["scan"] = self.scan
        # results_dict["num_subgoals"] = self.n_subgoals_in_task
        # results_dict["num_subgoal_success"] = self.n_success_subgoals
        results_dict["subgoal_success_rate"] = float(self.subgoal_success_rate)
        results_dict['distance_to_subgoals'] = [x.tolist() if isinstance(x, np.ndarray) else x for x in self.distance_to_subgoals]
        results_dict["finished_subgoal_ids"] = [x.tolist() if isinstance(x, np.ndarray) else x for x in self.finished_subgoals]
        results_dict["instruction"] = self.instruction
        results_dict["action_funcs"] = self.action_funcs
        results_dict["parsed_instruction"] = self.parsed_instruction
        with open(save_path, "w") as f:
            json.dump(results_dict, f, indent=4)
    
    def display_trajectory(self, save_path: str, occupancy_map: np.ndarray, 
                           traj_obs_list: List[np.ndarray], gt_obs: np.ndarray) -> None:
        """在占用栅格地图上绘制轨迹
        
        Args:
            save_path: 保存图片路径
            occupancy_map: 占用栅格地图,灰度图
            traj_obs_list: 实际轨迹点数组列表，每个元素为shape=(N,2)
            gt_obs: 参考轨迹点数组 shape=(N,2)
        """
        # 确保输入是numpy数组
        if not isinstance(occupancy_map, np.ndarray):
            occupancy_map = np.array(occupancy_map)

        # 转换为BGR彩色图像
        if len(occupancy_map.shape) == 2:
            vis_map = cv2.cvtColor(occupancy_map, cv2.COLOR_GRAY2BGR)
        else:
            vis_map = occupancy_map.copy()
        
        # 确保gt_obs是二维数组
        if gt_obs.ndim == 1:
            gt_obs = gt_obs.reshape(-1, 2)
        gt_obs = gt_obs.astype(np.int32)

        # 定义颜色列表 (BGR格式)
        colors = [
            (0, 165, 255),   # 橙色
            (0, 255, 255),   # 黄色
            (0, 255, 0),     # 绿色
            (255, 255, 0),   # 青色
            (255, 0, 0),     # 蓝色
            (255, 0, 255)    # 紫色
        ]
        # 绘制参考轨迹
        if len(gt_obs) > 1:
            for i in range(len(gt_obs) - 1):
                pt1 = tuple(gt_obs[i])
                pt2 = tuple(gt_obs[i + 1])
                cv2.line(vis_map, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)  # 参考轨迹始终为蓝色
        # 绘制每个episode的轨迹
        for idx, traj_obs in enumerate(traj_obs_list):
            # 确保traj_obs是二维数组
            if traj_obs.ndim == 1:
                traj_obs = traj_obs.reshape(-1, 2)
            traj_obs = traj_obs.astype(np.int32)

            # 为当前轨迹选择颜色
            color = colors[idx % len(colors)]

            # 绘制实际轨迹
            if len(traj_obs) > 1:
                for i in range(len(traj_obs) - 1):
                    pt1 = tuple(traj_obs[i])
                    pt2 = tuple(traj_obs[i + 1])
                    cv2.line(vis_map, pt1, pt2, color, 1, cv2.LINE_AA)

                # # 绘制起始点（绿色空心圆）
                # if len(traj_obs) > 0:
                #     start_point = tuple(traj_obs[0])
                #     cv2.circle(vis_map, start_point, radius=2, color=color, 
                #                thickness=1, lineType=cv2.LINE_AA)

                # # 绘制终点（蓝色空心圆）
                # if len(traj_obs) > 0:
                #     end_point = tuple(traj_obs[-1])
                #     cv2.circle(vis_map, end_point, radius=2, color=color, 
                #                thickness=1, lineType=cv2.LINE_AA)



        # 保存图像
        cv2.imwrite(save_path, vis_map)