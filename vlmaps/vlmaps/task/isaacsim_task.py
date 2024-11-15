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
