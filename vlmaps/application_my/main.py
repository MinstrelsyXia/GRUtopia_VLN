import os
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from omegaconf import DictConfig
import hydra

# function to display the topdown map

import cv2
import open3d as o3d

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "vlmaps"))
print(sys.path)
from vlmaps.vlmaps.robot.lang_robot import LangRobot
from vlmaps.vlmaps.dataloader.isaacsim_dataloader import VLMapsDataloaderHabitat
from vlmaps.vlmaps.navigator.navigator import Navigator
from vlmaps.vlmaps.controller.discrete_nav_controller import DiscreteNavController

from vlmaps.vlmaps.utils.mapping_utils import (
    grid_id2base_pos_3d,
    grid_id2base_pos_3d_batch,
    base_pos2grid_id_3d,
    cvt_pose_vec2tf,
)
from vlmaps.vlmaps.utils.index_utils import find_similar_category_id
from vlmaps.vlmaps.utils.isaacsim_utils import  display_sample
from vlmaps.vlmaps.utils.matterport3d_categories import mp3dcat

from typing import List, Tuple, Dict, Any, Union

from grutopia.core.env import BaseEnv

import os
import yaml
from pathlib import Path
from types import SimpleNamespace
import shutil
from grutopia.core.config import SimulatorConfig
import argparse

ROOT_DIR = "/ssd/xiaxinyuan/code/w61-grutopia"


from vln.src.dataset.data_utils import VLNDataLoader
from vlmaps.robot.isaac_robot import IsaacSimLanguageRobot
from vlmaps.application_my.build_static_map import TMP
from vlmaps.vlmaps.utils.llm_utils import parse_object_goal_instruction

def extract_instruction(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    instruction = ""
    for line in lines:
        if line.startswith("Instruction:"):
            instruction = line.split("Instruction:")[1].strip()
            break

    return instruction

class VLNTaskManager:
    def __init__(self,config, sim_config):
        self.root_dir = config.data_paths.vlmaps_data_dir
        self.robot = IsaacSimLanguageRobot(config,sim_config)
        self.map = TMP(data_dir = config.data_paths.vlmaps_data_dir)


    def parse_instructions(self):
        instruction = extract_instruction(self.status_info_path)
        self.object_categories = parse_object_goal_instruction(instruction)


    def find_all_objects(self):
        for object in self.object_categories:
            choice = self.robot.try_to_move_to_object(object)

def dict_to_namespace(d):
    ns = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            value = dict_to_namespace(value)
        setattr(ns, key, value)
    return ns
def process_args(total_config):
    '''Load configuration from YAML file'''

    # Load the YAML configuration file
    config = total_config
    '''Init simulation config'''
    sim_config = SimulatorConfig(config.sim_cfg_file)

    '''Update VLN config'''
    vln_config = dict_to_namespace(config)

    '''Init save directory'''
    for key, value in vars(config).items():
        setattr(vln_config, key, value)
    vln_config.root_dir = ROOT_DIR
    vln_config.log_dir = os.path.join(ROOT_DIR, "logs")
    vln_config.log_image_dir = os.path.join(vln_config.log_dir, "images", str(vln_config.env), str(vln_config.path_id))
    if not os.path.exists(vln_config.log_image_dir):
        os.makedirs(vln_config.log_image_dir)
    
    if vln_config.settings.mode == "sample_episodes":
        vln_config.sample_episode_dir = os.path.join(ROOT_DIR, "logs", "sample_episodes")
        if os.path.exists(vln_config.sample_episode_dir) and vln_config.settings.force_sample:
            shutil.rmtree(vln_config.sample_episode_dir)
        os.makedirs(vln_config.sample_episode_dir)

    return vln_config, sim_config
def build_dataset(cfg):
    ''' Build dataset for VLN
    '''
    vln_config, sim_config = process_args(cfg)
    vln_datasets = {}

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
    config_path="../config_my",
    config_name="test_config.yaml",
)
def main(config: DictConfig) -> None:
    vln_envs, vln_config, sim_config, data_camera_list = build_dataset(config.vln_config)
    vln_task = VLNTaskManager(config, sim_config)
    vln_task.parse_instructions()





if __name__ == "__main__":
    main()