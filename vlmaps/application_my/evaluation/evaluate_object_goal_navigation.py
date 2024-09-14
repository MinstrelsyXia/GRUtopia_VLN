import os
from pathlib import Path
import numpy as np
from omegaconf import DictConfig
import hydra

from vlmaps.vlmaps.task.habitat_object_nav_task import HabitatObjectNavigationTask

from vlmaps.vlmaps.application_my.isaac_robot import IsaacSimLanguageRobot
from vlmaps.vlmaps.utils.llm_utils import parse_object_goal_instruction
from vlmaps.vlmaps.utils.matterport3d_categories import mp3dcat

from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv

import json

from build_static_map import IsaacMap

import os,sys
import gzip
import json
import math
import numpy as np
import argparse
import yaml
import time
import shutil
from collections import defaultdict
from PIL import Image
from copy import deepcopy

# enable multiple gpus
# import isaacsim
# import carb.settings
# settings = carb.settings.get_settings()
# settings.set("/renderer/multiGPU/enabled", True)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSAC_SIM_DIR = os.path.join(os.path.dirname(ROOT_DIR), "isaac-sim-4.0.0")
sys.path.append(ISSAC_SIM_DIR)

from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container
from grutopia.core.util.log import log

# from grutopia_extension.utils import get_stage_prim_paths

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

from vln.src.dataset.data_utils import VLNDataLoader
from vln.src.utils.utils import dict_to_namespace
# from vln.src.local_nav.global_topdown_map import GlobalTopdownMap

from vln.main import build_dataset


def load_data(args, split):
    ''' Load data based on VLN-CE
    '''
    dataset_root_dir = args.datasets.base_data_dir
    total_scans = []
    load_data = []
    with gzip.open(os.path.join(dataset_root_dir, f"{split}", f"{split}.json.gz"), 'rt', encoding='utf-8') as f:
        data = json.load(f)
        for item in data["episodes"]:
            item["original_start_position"] = copy.copy(item["start_position"])
            item["original_start_rotation"] = copy.copy(item["start_rotation"])
            item["start_position"] = [item["original_start_position"][0], -item["original_start_position"][2], item["original_start_position"][1]]
            item["start_rotation"] = [-item["original_start_rotation"][3], item["original_start_rotation"][0], item["original_start_rotation"][2], -item["original_start_rotation"][1]] # [x,y,z,-w] => [w,x,y,z]
            item["scan"] = item["scene_id"].split("/")[1]
            item["c_reference_path"] = []
            if "reference_path" in item.keys():
                for path in item["reference_path"]:
                    item["c_reference_path"].append([path[0], -path[2], path[1]])
                item["reference_path"] = item["c_reference_path"]
                del item["c_reference_path"]
            load_data.append(item)
            total_scans.append(item["scan"])

    log.info(f"Loaded data with a total of {len(load_data)} items from {split}")
    return load_data, list(set(total_scans))

class IsaacSimTask(HabitatObjectNavigationTask):
    def load_task(self,data_dir):
        task_path = Path(data_dir) / "object_navigation_tasks.json"
        with open(task_path, "r") as f:
            self.task_dict = json.load(f)

    def empty_recorded_actions(self):
        self.recorded_actions_list = []
        self.recorded_robot_pos = []
        self.goal_tfs = None
        self.all_goal_tfs = None
        self.goal_id = None

from vln.parser import process_args

def build_dataset(cfg):
    ''' Build dataset for VLN
    '''
    vln_config, sim_config = process_args(cfg)
    vln_datasets = {}

    # load_data
    camera_list = [x.name for x in sim_config.config.tasks[0].robots[0].sensor_params if x.enable]
    if vln_config.settings.mode == 'sample_episodes':
        data_camera_list = vln_config.settings.sample_camera_list
    else:
        data_camera_list = None
    # camera_list = ["pano_camera_0"] # !!! for debugging
    vln_config.camera_list = camera_list
    
    return vln_datasets, vln_config, sim_config, data_camera_list

@hydra.main(
    version_base=None,
    config_path="../../config_my",
    config_name="object_goal_navigation_cfg",
)
def main(config: DictConfig) -> None:
    # load vln config
    vln_config, sim_config = process_args(config.vln_config)
    
    args = vln_envs
    if args.path_id == -1:
        log.error("Please specify the path id")
        return
    # get the specific path
    vln_envs = vln_envs[args.env]
    data_item = vln_envs.init_one_path(args.path_id)
    env = vln_envs.env
    
    robot = IsaacSimLanguageRobot(config,sim_config=sim_config, vln_config=vln_config)
    
    robot.setup_scene(data_item['scan'])
    
    reference_paths = data_item['reference_path']
    paths = []
    object_categories = parse_object_goal_instruction(data_item['instruction']['instruction_text'])
    # prev_pose = vln_envs.agents.get_world_pose()[0]
    # for cat_i, cat in enumerate(object_categories):
    #     if (cat_i==0):
    #         paths.append(prev_pose)
    #     print(f"Navigating to category {cat}")
    #     next_pose = map.get_nearest_pos(prev_pose, cat)
    #     paths.append(next_pose)
    #     prev_pose = next_pose

    # current_point = 0
    # move_interval = 500 # move along the path every 5 seconds
    # reset_robot = False
    
    # if config.vln_config.windows_head:
    #     vln_envs.cam_occupancy_map.open_windows_head(text_info=data_item['instruction']['instruction_text'])
    
    # '''start simulation'''
    # i = 0
    # warm_step = 50 if args.headless else 500

    # # init the action
    # action_name = vln_config.settings.action
    # if action_name == 'move_along_path':
    #     actions = {'h1': {action_name: [paths]}}
    #     # vln_envs.init_BEVMap()
    #     current_point = 0
    # else:
    #     actions = {'h1': {action_name: []}}

    # while env.simulation_app.is_running():
    #     i += 1
    #     reset_flag = False
    #     env_actions = []
    #     # env_actions.append(actions)
        
    #     if i < warm_step:
    #         # give me some time to adjust the view position
    #         # let the robot stand still during the first warm steps.
    #         env_actions.append({'h1': {'stand_still': []}})
    #         obs = env.step(actions=env_actions)
    #         agent_action_state = {'finished': True}
    #         continue
        
    #     # TODO: wrap up check and reset robot
    #     if i % 10 == 0:
    #         # print(i)
    #         if vln_config.settings.check_and_reset_robot:
    #             topdown_map = vln_envs.GlobalTopdownMap(args, data_item['scan']) # !!!
    #             freemap, camera_pose = vln_envs.get_global_free_map(verbose=vln_config.test_verbose) # !!!
    #             topdown_map.update_map(freemap, camera_pose, verbose=vln_config.test_verbose) # !!!

    #             reset_robot = vln_envs.check_and_reset_robot(cur_iter=i, update_freemap=False, verbose=vln_config.test_verbose)
    #             reset_flag = reset_robot
    #             if reset_flag:
    #                 # actions = {'h1': {'stand_still': []}}
    #                 vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
    #                 robot_current_pos = vln_envs.agents.get_world_pose()[0]
    #                 exe_path, node_type = vln_envs.bev.navigate_p2p(robot_current_pos, paths[current_point], verbose=vln_config.test_verbose)

                    
    #         if vln_config.windows_head:
    #             # show the topdown camera
    #             vln_envs.cam_occupancy_map.update_windows_head(robot_pos=vln_envs.agents.get_world_pose()[0])

    #     # TODO: wrap up navigation
    #     # get_pose; save_observation; update_occupancy_map(-> ?); navigate_p2p; move_along_path
    #     # vln_envs.
    #     if i % 100 == 0:
    #         print(i)
    #         if not reset_flag:
    #             # if args.save_obs:
    #             #     vln_envs.save_observations(camera_list=vln_config.camera_list, data_types=["rgba", "depth"], step_time=i)

    #             # move to next waypoint
    #             if current_point == 0:
    #                 log.info(f"===The robot starts navigating===")
    #                 log.info(f"===The robot is navigating to the {current_point+1}-th target place.===")
    #                 # init BEVMapgru
    #                 agent_current_pose = vln_envs.agents.get_world_pose()[0]
    #                 vln_envs.init_BEVMap(robot_init_pose=agent_current_pose)
                    
    #                 if args.save_obs:
    #                     vln_envs.save_observations(camera_list=vln_config.camera_list, data_types=["rgba", "depth"], step_time=i)
    #                 vln_envs.bev.step_time = i
    #                 vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
    #                 exe_path, node_type = vln_envs.bev.navigate_p2p(paths[current_point], paths[current_point+1], verbose=vln_config.test_verbose)
    #                 if node_type == 2:
    #                     log.info("Path planning fails to find the path from the current point to the next point.")
    #                 current_point += 1
    #                 actions = {'h1': {'move_along_path': [exe_path]}}
    #             else:
    #                 if agent_action_state['finished']:
    #                     log.info("***The robot has finished the action.***")
    #                     if current_point < len(paths)-1:
    #                         if args.save_obs:
    #                             vln_envs.save_observations(camera_list=vln_config.camera_list, data_types=["rgba", "depth"], step_time=i)
    #                         vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
    #                         robot_current_pos = vln_envs.agents.get_world_pose()[0]
    #                         exe_path, node_type = vln_envs.bev.navigate_p2p(robot_current_pos, paths[current_point+1], verbose=vln_config.test_verbose)
    #                         current_point += 1

    #                         actions = {'h1': {'move_along_path': [exe_path]}} 
    #                         log.info(f"===The robot is navigating to the {current_point+1}-th target place.===")
    #                     else:
    #                         actions = {'h1': {'stand_still': []}}
    #                         log.info("===The robot has achieved the target place.===")
    #                 else:
    #                     log.info("===The robot has not finished the action yet.===")
        
        
    #     if i % 500 == 0 and not agent_action_state['finished']:
    #         if args.save_obs:
    #             vln_envs.save_observations(camera_list=vln_config.camera_list, data_types=["rgba", "depth"], step_time=i)
        
    #         # update BEVMap every specific intervals
    #         vln_envs.bev.step_time = i
    #         vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
            
    #         # after BEVMap's update, determine whether the robot's path is blocked
    #         if current_point > 0:
    #             agent_current_pose = vln_envs.agents.get_world_pose()[0]
    #             next_action_pose = exe_path[agent_action_state['current_index']+1] if len(exe_path)>agent_action_state['current_index']+1 else agent_action_state['current_point']
    #             if vln_envs.bev.is_collision(agent_current_pose, next_action_pose):
    #                 log.info("===The robot's path is blocked. Replanning now.===")
    #                 exe_path, _ = vln_envs.bev.navigate_p2p(agent_current_pose, paths[current_point], verbose=vln_config.test_verbose)
    #                 actions = {'h1': {'move_along_path': [exe_path]}}
                            
    #     env_actions.append(actions)
    #     obs = env.step(actions=env_actions)

    #     # get the action state
    #     if len(obs[vln_envs.task_name]) > 0:
    #         agent_action_state = obs[vln_envs.task_name][vln_envs.robot_name][action_name]
    #     else:
    #         agent_action_state['finished'] = False

    # env.simulation_app.close()
    # if vln_config.windows_head:
    #     # close the topdown camera
    #     vln_envs.cam_occupancy_map.close_windows_head()



if __name__ == "__main__":
    vln_envs, vln_config, sim_config, data_camera_list = build_dataset()
    main()



