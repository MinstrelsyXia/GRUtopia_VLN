from pathlib import Path
import json
from typing import Dict, List, Tuple, Union

from omegaconf import DictConfig
import numpy as np
import hydra

# from vlmaps.vlmaps.task.habitat_task import HabitatTask
# from vlmaps.vlmaps.utils.habitat_utils import agent_state2tf, get_position_floor_objects
# from vlmaps.vlmaps.utils.navigation_utils import get_dist_to_bbox_2d
# from vlmaps.vlmaps.utils.habitat_utils import display_sample

class IsaacSimSpatialGoalNavigationTask():
    def __init__(self,config):
        self.actions = []
        self.pos_list = []
        self.config = config
    
    def add_action(self,action):
        self.actions.append(action)

    def get_past_path(self):
        return self.pos_list
    
    def add_pos(self,pos):
        self.pos_list.append(pos)
    
    def setup_task(self, item):
        pass
        # remember to transpose
        self.instruction = item["instruction"]['instruction_text']

        self.goals = item["reference_path"]
        self.episode_id = item['episode_id']
        self.scan = item['scan']
        self.n_subgoals_in_task = len(self.goals)
        self.success_subgoal_num = 0
        self.finished_subgoals = []
        self.distance_to_subgoals = []
        self.success = False
    
    def calculate_metric(self):
        for goal_3 in self.goals:
            goal = goal_3[:2]
            min_dist = np.inf
            for pos_3 in self.pos_list:
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
        distances = np.sqrt(np.sum(np.diff(self.pos_list, axis=0) ** 2, axis=1))

        # Sum up all the distances to get the total path length
        gt_length = np.sqrt(np.sum(np.diff(self.goals, axis=0)** 2, axis=1))
        self.spl = np.sum(gt_length) / (np.sum(distances) + 1e-6)

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
        
    #     # 计算Success Rate (SR) - 是否到达目标点
    #       # 成功阈值通常设为3米
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
    def save_single_task_metric(
        self,
        save_path: Union[Path, str],
    ):
        results_dict = {}
        results_dict["episode_id"] = self.episode_id
        results_dict["scan"] = self.scan
        # results_dict["num_subgoals"] = self.n_subgoals_in_task
        # results_dict["num_subgoal_success"] = self.n_success_subgoals
        results_dict["subgoal_success_rate"] = float(self.subgoal_success_rate)
        results_dict["finished_subgoal_ids"] = [x.tolist() if isinstance(x, np.ndarray) else x for x in self.finished_subgoals]
        results_dict["SPL"] = float(self.spl)
        results_dict["instruction"] = self.instruction
        results_dict["actions"] = [x.tolist() if isinstance(x, np.ndarray) else x for x in self.actions]
        
        with open(save_path, "w") as f:
            json.dump(results_dict, f, indent=4)
