import os
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from omegaconf import DictConfig
import hydra
import importlib
# function to display the topdown map

import cv2
import open3d as o3d

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "vlmaps"))
# print(sys.path)

from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
import pickle
import time
from vlmaps.vlmaps.robot.lang_robot import LangRobot
from vlmaps.vlmaps.dataloader.isaacsim_dataloader import VLMapsDataloaderHabitat
from vlmaps.vlmaps.navigator.navigator import Navigator
from vlmaps.vlmaps.controller.discrete_nav_controller import DiscreteNavController
from vlmaps.vlmaps.task.isaacsim_task import IsaacSimSpatialGoalNavigationTask
from vlmaps.vlmaps.utils.mapping_utils import (
    grid_id2base_pos_3d,
    grid_id2base_pos_3d_batch,
    base_pos2grid_id_3d,
    cvt_pose_vec2tf,
)
from vlmaps.vlmaps.utils.index_utils import find_similar_category_id, get_segment_islands_pos, get_dynamic_obstacles_map_3d
from vlmaps.vlmaps.utils.isaacsim_utils import  display_sample
from vlmaps.vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.vlmaps.utils.visualize_utils import pool_3d_label_to_2d
from vlmaps.vlmaps.utils.llm_utils import parse_object_goal_instruction, parse_spatial_instruction

from typing import List, Tuple, Dict, Any, Union

from grutopia.core.env import BaseEnv
from grutopia.core.util.log import log
from grutopia.core.util.container import is_in_container
from vln.src.dataset.data_utils import load_data,load_scene_usd

from vln.src.utils.utils import  compute_rel_orientations,visualize_pc,get_diff_beween_two_quat




from vln.src.local_nav.BEVmap import BEVMap

from vlmaps.vlmaps.map.map import Map

from vlmaps.application_my.utils import NotFound, EarlyFound, TooManySteps,extract_parameters, extract_self_methods, visualize_subgoal_images

import logging
import argparse

logging.getLogger('PIL').setLevel(logging.WARNING) # used to delete the log file from PIL
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
def is_equal(a, b, threshold=0.1):
    # 确保输入是 numpy 数组
    a = np.array(a)
    b = np.array(b)
    
    # 计算欧氏距离
    distance = np.linalg.norm(a - b)
    
    # 如果距离小于指定阈值，则认为相等
    return distance < threshold


class IsaacSimLanguageRobot(LangRobot):

    # if vln_config.windows_head:
    #     vln_envs.cam_occupancy_map.open_windows_head(text_info=data_item['instruction']['instruction_text'])
    
    # while env.simulation_app.is_running():
    # ...
    # env.simulation_app.close()
    # if vln_config.windows_head:
    #     # close the topdown camera
    #     vln_envs.cam_occupancy_map.close_windows_head()
    def __init__(self, config: DictConfig, sim_config,vln_config,split):
        '''
        config: from VLMaps
        sim_config, vln_config: from VLN's two configs

        '''
        super().__init__(config)

        self.test_scene_dir = self.config["data_paths"]["habitat_scene_dir"]
        # data_dir = Path(self.config["data_paths"]["vlmaps_data_dir"]) / "vlmaps_dataset" 
        self.vlmaps_data_dir = self.config["data_paths"]["vlmaps_data_dir"]
        self.test_dir = self.config["data_paths"]["test_file_save_dir"]
        # self.vlmaps_data_save_dirs = [
        #     data_dir / x for x in sorted(os.listdir(data_dir)) if x != ".DS_Store"
        # ]  # ignore artifact generated in MacOS
        self.map_type = self.config["params"]["map_type"] # from params
        self.map_config = self.config["map_config"]
        self.camera_height = self.config["params"]["camera_height"]
        self.gs = self.config["params"]["gs"]
        self.cs = self.config["params"]["cs"]
        self.forward_dist = self.config["params"]["forward_dist"]
        self.turn_angle = self.config["params"]["turn_angle"]


        self.last_scene_name = ""
        # self.agent_model_dir = self.config["data_paths"]["agent_mesh_dir"]

        self.vis = False

        self.nav = Navigator()
        self.controller = DiscreteNavController(self.config["params"]["controller_config"])

        # from data_utils: init agents
        self.sim_config = sim_config
        self.vln_config = vln_config
        
        
        self.data, self._scans = load_data(vln_config, split)
        self.robot_type = sim_config.config.tasks[0].robots[0].type
        self.online = self.config['online']
        self.from_scratch = self.config['from_scratch']
        self.frontier_type = self.config['find_frontier_type']
        for cand_robot in vln_config.robots:
            if cand_robot["name"] == self.robot_type:
                self.robot_offset = np.array([0,0,cand_robot["z_offset"]])
                break
        if self.robot_offset is None:
            log.error("Robot offset not found for robot type %s", self.robot_type)
            raise ValueError("Robot offset not found for robot type")
        
        # process paths offset
        if vln_config.settings.mode == "sample_episodes":
            for scan, data in self.data.items():
                for item in data:
                    item["start_position"] += self.robot_offset
                    for i, path in enumerate(item["reference_path"]):
                        item["reference_path"][i] += self.robot_offset
        else:
            for item in self.data:
                item["start_position"] += self.robot_offset
                for i, path in enumerate(item["reference_path"]):
                    item["reference_path"][i] += self.robot_offset

        
        self.task_name = sim_config.config.tasks[0].name # only one task
        self.robot_name = sim_config.config.tasks[0].robots[0].name # only one robot type
        self.camera_list = vln_config.camera_list

        self.bev = None
        self.surrounding_freemap_connected = None
        self.step = 0
        ## obstacle map to get the dynamic exploration map
        
        self.min_height = 100 #! arbitray
        ## occupancy map to get the top-down oracle map:
        # in self._setup_sim
        self.eval_helper = IsaacSimSpatialGoalNavigationTask(config) 
    
    ############################### init env ############################################

    #! not in use !!!
    def scene_id2scene_name(self,scene_id):
        '''
        scene id: 0; scene name: 5J......
        '''
        self.scene_id = scene_id
        vlmaps_data_dir = self.vlmaps_data_dir[scene_id]
        print(vlmaps_data_dir)
        self.scene_name = vlmaps_data_dir.name.split("_")[0]


    def setup_camera(self):
        self.camera = self.env._runner.current_tasks[self.task_name].robots[self.robot_name].sensors[self.camera_list[0]]._camera
        camera_in = self.camera.get_intrinsics_matrix()
        self.fov = 2 * np.arctan(camera_in[0, 2] / camera_in[0, 0]) # 60 degrees; counted in rad

    def setup_scene(self,  episode_id: int ,trajectory_id: int,reset_scene=False,init_omni_scene=False):
        """
        Setup the simulator, load scene data and prepare
        the LangRobot interface for navigation
        from LangRobot
        """
        # self.scene_id = scene_id
        # vlmaps_data_dir = self.vlmaps_data_save_dirs[scene_id]
        # print(vlmaps_data_dir)
        # self.scene_name = vlmaps_data_dir.name.split("_")[0]

        # trajectory_id:37; scene_id:s8pcm....glb; scan_id: s8...
        item = self._setup_sim(self.sim_config, episode_id,  trajectory_id, vlmap_dataset=self.online,reset_scene=reset_scene,init_omni_scene=init_omni_scene) # from VLNdataloader init_one_path

        self.item = item

        self.setup_map(self.vlmaps_data_dir)
        self.setup_camera()
        cropped_obst_map = self.map.get_obstacle_cropped()
        #! need modification; args: judge by height or by index!!!
        if self.config.map_config.potential_obstacle_names and self.config.map_config.obstacle_names and self.config.map_config.customize_obstacle_map:
            print("come here")
            self.map.customize_obstacle_map(
                self.config.map_config.potential_obstacle_names,
                self.config.map_config.obstacle_names,
                vis=self.config.nav.vis,
            )
            cropped_obst_map = self.map.get_customized_obstacle_cropped()
        self.eval_helper.setup_task(item)
        # self.nav.build_visgraph(
        #     cropped_obst_map,
        #     0,
        #     0,
        #     vis=self.config["nav"]["vis"],
        # )

        # self._setup_localizer(vlmaps_data_dir)

    def load_scene_map(self, data_dir: str, map_config: DictConfig):
        self.map = Map.create(map_config) #! should include isaacmap!!!
        self.map.init_map(data_dir,self.test_file_save_dir)
        # self.map.generate_obstacle_map()

    def setup_map(self, vlmaps_data_dir: str):
        if self.from_scratch == False:
            self.load_scene_map(vlmaps_data_dir, self.config["map_config"])

            # TODO: check if needed
            if "3d" in self.config.map_config.map_type:
                self.map.init_categories(mp3dcat.copy())
                self.global_pc = grid_id2base_pos_3d_batch(self.map.grid_pos, self.cs, self.gs)

            # self.vlmaps_dataloader = VLMapsDataloaderHabitat(vlmaps_data_dir, self.config.map_config, map=self.map)
        else:
            from vlmaps.application_my.build_dynamic_map import TMP
            self.map = TMP(self.map_config, data_dir=self.vlmaps_data_dir,test_file_save_dir=self.test_file_save_dir,robot_init_pose = self.agent_init_pose)
            self.map.init_map(vlmaps_data_dir,self.test_file_save_dir)

    def init_omni_env(self):
        rotations_utils = importlib.import_module("omni.isaac.core.utils.rotations")
        self.quat_to_euler_angles = rotations_utils.quat_to_euler_angles
        self.euler_angles_to_quat = rotations_utils.euler_angles_to_quat
        # from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat

    def init_env(self, sim_config, headless=True):
        '''init env''' 
        self.env = BaseEnv(sim_config, headless=headless, webrtc=False)
    
    
    def init_agents(self):
        '''call after self.init_env'''
        self.agents = self.env._runner.current_tasks[self.task_name].robots[self.robot_name].isaac_robot
        self.agent_last_pose = None
        self.agent_init_pose = self.sim_config.config.tasks[0].robots[0].position
        self.agent_init_rotation = self.sim_config.config.tasks[0].robots[0].orientation

        self.set_agent_pose(self.agent_init_pose, self.agent_init_rotation)
    
    def set_agent_pose(self, position, rotation):
        self.agents.set_world_pose(position, rotation)
    
    def reset_sim_scene(self,scan):
        # reset scene without restart app
        start_time = time.time()
        self.env._runner._world.clear()
        self.env._runner.add_tasks(self.sim_config.config.tasks)
        log.info(f"Reset scene {scan} without restarting app for using {((time.time()-start_time)/60):.2f} minutes.")

    def _setup_sim(self, sim_config, episode_id,path_id,  headless=False, vlmap_dataset=False,reset_scene=False,init_omni_scene=False):
        """
        Setup IsaacSim simulator, load IsaacSim scene and relevant mesh data
        """
        # if not dynamic
        # if vlmap_dataset==False:
        #     for item in self.data:
        #         if item['trajectory_id'] == path_id:
        #             scene_usd_path = load_scene_usd(self.vln_config, item['scan'])
        #             instruction = item['instruction']['instruction_text']
        #             if 'stair' in instruction:
        #                 continue
        #             self.sim_config.config.tasks[0].scene_asset_path = scene_usd_path
        #             self.sim_config.config.tasks[0].robots[0].position = item["start_position"]
        #             self.sim_config.config.tasks[0].robots[0].orientation = item["start_rotation"] 
        #             self.init_env(self.sim_config, headless=self.vln_config.headless)
        #             self.init_omni_env()
        #             self.init_agents()
        #             self.init_cam_occunpancy_map(robot_prim=self.agents.prim_path,start_point=item["start_position"]) 
        #             log.info("Initialized path id %d", path_id)
        #             log.info("Scan: %s", item['scan'])
        #             log.info("Instruction: %s", item['instruction']['instruction_text'])
        #             self.instruction = item['instruction']['instruction_text']
        #             log.info(f"Start Position: {self.sim_config.config.tasks[0].robots[0].position}, Start Rotation: {self.sim_config.config.tasks[0].robots[0].orientation}")
        #             return item
        #     log.error("Path id %d not found in the dataset", path_id)
        #     return None
        # else:
        #     for item in self.data:
        #         if item['scan'] in self.vlmaps_data_save_dirs.split("/")[-1]:
        #             # item['scan]: s8pc...
        #             scene_usd_path = load_scene_usd(self.vln_config, item['scan'])
        #             self.sim_config.config.tasks[0].scene_asset_path = scene_usd_path
        #             self.sim_config.config.tasks[0].robots[0].position = item["start_position"]
        #             self.sim_config.config.tasks[0].robots[0].orientation = item["start_rotation"] 
        #             self.init_env(self.sim_config, headless=self.vln_config.headless)
        #             self.init_omni_env()
        #             self.init_agents()
        #             self.init_cam_occunpancy_map(robot_prim=self.agents.prim_path,start_point=item["start_position"]) 
        #             self.init_occupancy_map()
        #             log.info("Initialized path id %d", item['trajectory_id'])
        #             log.info("Scan: %s", item['scan'])
        #             log.info("Instruction: %s", item['instruction']['instruction_text'])
        #             self.instruction = item['instruction']['instruction_text']
        #             log.info(f"Start Position: {self.sim_config.config.tasks[0].robots[0].position}, Start Rotation: {self.sim_config.config.tasks[0].robots[0].orientation}")
        #             return item
        #     log.error("Path id %d not found in the dataset", path_id)
        #     return None

        for item in self.data:
            if item['episode_id'] == episode_id:
                scene_usd_path = load_scene_usd(self.vln_config, item['scan'])
                instruction = item['instruction']['instruction_text']
                # if 'stair' in instruction:
                #     print('erroe!!! stair occurrs')
                #     continue
                self.sim_config.config.tasks[0].scene_asset_path = scene_usd_path
                self.sim_config.config.tasks[0].robots[0].position = item["start_position"]
                self.sim_config.config.tasks[0].robots[0].orientation = item["start_rotation"] 
                self.vlmaps_data_dir = self.vlmaps_data_dir + f"/{item['scan']}/id_{item['episode_id']}"
                self.test_file_save_dir = self.test_dir + f"/{item['scan']}/id_{item['episode_id']}"
                self.vln_config.log_image_dir = self.test_file_save_dir
                if not os.path.exists(self.test_file_save_dir):
                    os.makedirs(self.test_file_save_dir, exist_ok=True)
                self.nav_save_dir = self.test_file_save_dir + "/nav"
                if not os.path.exists(self.nav_save_dir):
                    os.makedirs(self.nav_save_dir, exist_ok=True)
                if init_omni_scene:
                    self.init_env(self.sim_config, headless=self.vln_config.headless)
                    self.init_omni_env()
                if reset_scene:
                    self.env.simulation_app.close()
                    self.env = None
                    self.init_env(self.sim_config, headless=self.vln_config.headless)
                    self.init_omni_env()
                    # self.reset_sim_scene(item['scan'])
                self.init_agents()
                self.init_cam_occunpancy_map(robot_prim=self.agents.prim_path,start_point=item["start_position"]) 
                self.init_occupancy_map()
                log.info("Initialized path id %d", episode_id)
                log.info("Scan: %s", item['scan'])
                log.info("Instruction: %s", item['instruction']['instruction_text'])
                self.instruction = item['instruction']['instruction_text']
                log.info(f"Start Position: {self.sim_config.config.tasks[0].robots[0].position}, Start Rotation: {self.sim_config.config.tasks[0].robots[0].orientation}")
                self.reference_path = item["reference_path"]
                return item
        log.error("Path id %d not found in the dataset", episode_id)
        return None

    def set_agent_state(self, position, rotation):
        self.agents.set_world_pose(position,rotation)
        self._set_nav_curr_pose()

#################### update from VLNDataLoader ###############################
    def init_cam_occunpancy_map(self, robot_prim="/World/env_0/robots/World/h1",start_point=[0,0,0]):
        # some pacakages can only be imported after app.run()
        from vln.src.local_nav.camera_occupancy_map import CamOccupancyMap
        from vln.src.local_nav.global_topdown_map import GlobalTopdownMap

        self.GlobalTopdownMap = GlobalTopdownMap
        # self.cam_occupancy_map_local = CamOccupancyMap(self.vln_config, robot_prim, start_point, local=True)
        # self.cam_occupancy_map_global = CamOccupancyMap(self.vln_config, robot_prim, start_point, local=False)
        self.robots = self.env._runner.current_tasks[self.task_name].robots[self.robot_name]
        self.cam_occupancy_map_local = CamOccupancyMap(self.vln_config, self.robots.sensors['topdown_camera_50'])
        self.cam_occupancy_map_global = CamOccupancyMap(self.vln_config, self.robots.sensors['topdown_camera_500'])



#################### update from VLFM ###########################
    def init_occupancy_map(self):
        from vlmaps.vlfm.obstacle_map import ObstacleMap
        min_height, max_height = self.map_config["robot_z"][0],self.map_config["robot_z"][1]
        agent_radius = self.vln_config.maps.agent_radius
        area_thresh = self.map_config["area_thresh"]
        hole_area_thresh = self.map_config["hole_area_thresh"]
        size = self.map_config["map_size"]
        pixels_per_meter = 1.0/self.cs
        self.ObstacleMap = ObstacleMap(min_height=min_height, max_height=max_height,agent_radius=agent_radius, area_thresh=area_thresh,hole_area_thresh= hole_area_thresh, size = size, pixels_per_meter = pixels_per_meter,log_image_dir=self.test_file_save_dir)



    def update_semantic_map(self):
        # get observation
        obs = self.get_observations(["rgba","depth"])
        rgb = obs[self.task_name][self.robot_name][self.camera_list[0]]["rgba"][...,:3]
        depth_map = obs[self.task_name][self.robot_name][self.camera_list[0]]["depth"]
        max_depth = 10
        depth_map[depth_map > max_depth] = 0
        # update semantic map
        pc, max_depth= self.map._update_semantic_map(self.camera, rgb, depth_map, labels = mp3dcat[1:-1],step=self.step)

        return pc,max_depth
    
    def update_obstacle_map(self,pc,max_depth):
        '''
        called after update_semantic_map
        '''
        camera_pose = self.agents.get_world_pose()
        camera_position = camera_pose[0]
        camera_orientation = camera_pose[1]
        camera_orientation_angle = self.quat_to_euler_angles(camera_orientation)
        # update obstacle map
        # pc_filtered = pc[np.abs(pc[:,2] - self.agent_init_pose) < 0.6]
        # TODO: find a better way to filter small points
        # sorted_pc = np.sort(pc[:,2])
        # tenth_largest_value = sorted_pc[-10]
        # tenth_smallest_value = sorted_pc[10]
        # low_bound = - tenth_largest_value+ camera_position[2]
        # upper_bound = - tenth_smallest_value + camera_position[2]
        # pc_filtered = pc[(-0.9 < (camera_position[2]-pc[:,2])) & ((camera_position[2]-pc[:,2]) < 0.8)]
        pc_filtered = pc[(0 < (camera_position[2]-pc[:,2])) & ((camera_position[2]-pc[:,2]) < 0.6)]
        # print("check floor and ceiling", np.min(camera_position[2]-pc_filtered[2,:]),np.max(camera_position[2]-pc_filtered[2,:]))
        self.ObstacleMap.update_map_with_pc(
            pc_filtered,
            camera_position=camera_position,
            camera_orientation=camera_orientation_angle+np.pi/2,
            max_depth=max_depth, 
            topdown_fov=self.fov ,
            verbose=self.vln_config.test_verbose,
            step = self.step
            )
        
    def too_many_steps(self,max_step = 15000):
        if self.step > max_step:
            log.info(f"Too many steps: {self.step}, exiting.")
            raise TooManySteps("Too many steps")
        
    def get_frontier(self,action=None,verbose=True):
        '''
        return pos in obstacle map coord.
        '''

        frontiers = self.ObstacleMap.frontiers # array of waypoints
        if len(frontiers) == 0:
            log.info("Frontier not found. Moving to point at random")
            frontiers = np.array([self.ObstacleMap.get_random_free_point()])[0]
            return frontiers
            # frontiers.append(self.ObstacleMap.get_random_free_point())
            # raise NotFound("Frontier not found")
        else:
            # randomly pick a frontier:
            # frontier in world coord
            if self.frontier_type == "default":
                num = frontiers.shape[0]
                idx = np.random.randint(0,num)
                pos = frontiers[idx]
                
                # return self.ObstacleMap._xy_to_px(np.array([[pos[0],pos[1]]]))[0]
                return self.ObstacleMap._xy_to_px(np.array([pos]))[0]
            if self.frontier_type == 'closest':
                min_dist = 0
                for index, frontier in enumerate(frontiers):
                    current_position = self.agents.get_world_pose()[0][:2]
                    dist = np.linalg.norm(frontier - current_position)
                    if dist > min_dist:
                        min_dist = dist
                        pos = frontier
                return self.ObstacleMap._xy_to_px(np.array([pos]))[0]
            if self.frontier_type == 'vlmap':
                frontiers = self.ObstacleMap.frontiers.copy()
                print(frontiers)
                log.info(f"found {len(frontiers)} frontiers")
                if len(frontiers) == 1:
                    pos = self.ObstacleMap._xy_to_px(np.array([frontiers[0]]))[0]
                    return pos
                else:
                    subgoal = extract_parameters(action)
                    similarity_rate = []
                    rgb_list = []
                    for frontier in frontiers:
                        rgb, similarity = self.get_frontier_viewpoint(frontier,subgoal)
                        rgb_list.append(rgb)
                        similarity_rate.append(similarity[0])
                    idx = np.argmax(similarity_rate)
                    pos = frontiers[idx]
                    if verbose == True:
                        # save rgbs in one image, with text f"subgoal is {subgoal}, chosen {idx} pos}", for all images, the caption is "image {idx}, with similarity {similarity[idx]}", for the chosen image, the caption should be in red
                        path_save_path = self.nav_save_dir + f"/frontier_viewpoints_{self.step}.png"
                        visualize_subgoal_images(rgb_list,similarity_rate,idx,subgoal,path_save_path)
                    return self.ObstacleMap._xy_to_px(np.array([pos]))[0]


    def get_frontier_viewpoint(self,pos,subgoal):
        '''
        pos: isaac coord
        '''
        turn_angle = self.get_angle(pos)
        self.turn(turn_angle,threshold = 0.05)
        obs = self.get_observations(["rgba","depth"])
        rgb = obs[self.task_name][self.robot_name][self.camera_list[0]]["rgba"][...,:3]

        similarity = self.map.get_score_mat_clip(rgb,subgoal)
        return rgb, similarity
        
    def from_map_to_xyz(self, camera_pose, pcd_min):
        current_position, current_quaternion = camera_pose[0],camera_pose[1]
        theta_deg = self.quat_to_euler_angles(current_quaternion)[2]/np.pi*180
        row = int(current_position[0] - pcd_min[0]) / self.cs
        col = int(current_position[1] - pcd_min[1]) / self.cs
        self.full_map_pose = [row, col, theta_deg]

    def to_full_map_pose(self):
        '''
        from self.vlmaps_dataloader.to_full_map_pose()
        '''
        assert self.full_map_pose is not None, "Please call from_xx() first."
        return self.full_map_pose
    
    def get_frontier_xyz(self):
        '''
        input: frontier in semantic map
        output: frontier in xyz
        '''
        frontier = self.get_frontier()
        x,y = self.map.from_map_to_xyz(frontier[0],frontier[1])
        return np.array([x,y,self.agent_init_pose[2]])
    
    def look_around(self):
        self.turn(180)
        self.turn(180)

    def explore(self,action_name):
        """
        Explore the environment by moving the robot around
        """
        # self.look_around()
        log.info(f"enter explore with action {action_name}")
        try:
            eval(action_name)
            return True
        except NotFound as e:
            log.warning(f"{e}. Object not found after looking around, moving to a frontier.")
            frontier_point = self.get_frontier(action_name)         
            turn_angle = self.get_angle(frontier_point,coord = 'uv') # in obstacle map coord
            turn_flag = self.turn(turn_angle,threshold = 0.05) # turn left turn_angle
            move_flag =  self.move_to(frontier_point,type = 'obs',threshold = 0.3)   
            if move_flag == False and turn_flag == False:
                log.warning(f"Failed to move to the frontier point {frontier_point}")
                return False
            return False
        

    def get_angle(self, frontier_point, coord = 'xy'):
        """
        Get the angle between the robot and the frontier point in degree
        frontier_point: in Obs coord
        """
        # Get the robot's current position on vlmap
        # self._set_nav_curr_pose()
        # curr_pose_on_full_map = self.get_agent_pose_on_map()
        # roboo_pos = self.
        # robot_angle = curr_pose_on_full_map[2]
        # get robot's position on obsmap
        pose = self.agents.get_world_pose()
        position, orientation = pose[0], pose[1]

        orientation_yaw = self.quat_to_euler_angles(orientation)[2]
        xy = position[:2]
        if coord == 'uv':
            current_pos = self.ObstacleMap._xy_to_px(np.array([[xy[0],xy[1]]]))[0]
            current_angle = self.ObstacleMap._get_current_angle_on_map(orientation_yaw + np.pi / 2) 
            target_rotation = np.arctan2(frontier_point[1] - current_pos[1], frontier_point[0] - current_pos[0]) / np.pi * 180 # already in [-pi,pi]
            return ((current_angle -target_rotation+180) % 360-180)
            # Calculate the angle between the robot and the frontier point
        else:
            current_angle = orientation_yaw/np.pi*180
            target_rotation = np.arctan2(frontier_point[1] - xy[1], frontier_point[0] - xy[0]) / np.pi * 180 # already in [-pi,pi]
            return ((-orientation_yaw +target_rotation+180) % 360-180)

        

    def test_movement(self, action_name: str):
        """
        Tries to execute a movement action and explore if the action fails.
        If NotFound exception is raised, the robot starts exploring.
        """
        # prev_pos = self.curr_pos_on_map #! None
        prev_pos = np.array([0,0])
        # prev_ang = self.curr_ang_deg_on_map
        prev_ang = 0
        log.info(f"enter test_movement with action {action_name}")
        while True:
            try:
                self.map.load_3d_map()
                # 尝试执行传入的动作
                log.info(f"testing action {action_name}")
                eval(action_name)
                self._set_nav_curr_pose()
                # 如果位置发生变化，说明动作成功，退出循环
                if not (is_equal(self.curr_pos_on_map, prev_pos) and is_equal(self.curr_ang_deg_on_map, prev_ang)):
                    log.info(f"Successfully executed {action_name}")

                    break
                else:
                    # 如果位置没有发生变化，记录日志并进行探索
                    log.info(f"Robot didn't move after executing {action_name}, start exploring")
                    self.explore(action_name)

            except NotFound as e:
                # 捕获 NotFound 异常，记录日志并进行探索
                log.warning(f"{e}. Object not found , starting exploration.")
                found = self.explore(action_name) # update occupancy map, semantic map
                if found == True:
                    break
            except NameError as e:
                print(f"Instruction GPT parsed {action_name} doesn't exist: {e}, following next instruction")
                break
            except TooManySteps as e:
                log.warning(f"{e}. Too many steps, stopping exploration.")
                break


            # except Exception as e:
            #     # 捕获其他异常并记录日志（如评估 action_name 失败）
            #     log.error(f"An error occurred during {action_name}: {e}. Starting exploration.")

            # 更新 prev_pos，以便在下一次迭代中继续比较位置变化
            prev_pos = self.curr_pos_on_map
            prev_ang = self.curr_ang_deg_on_map

    def get_robot_bottom_z(self):
        '''get robot bottom z'''
        return self.env._runner.current_tasks[self.task_name].robots[self.robot_name].get_ankle_base_z()-self.sim_config.config_dict['tasks'][0]['robots'][0]['ankle_height']
    
    def move_to_object(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_nearest_pos(self.curr_pos_on_map, name)
        self.move_to(pos)

    def get_global_free_map(self, verbose=False):
        ''' Use top-down orthogonal camera to get the ground-truth surrounding free map
            This is useful to reset the robot's location when it get stuck or falling down.
        '''
        self.global_freemap_camera_pose = self.cam_occupancy_map_global.topdown_camera.get_world_pose()[0]
        self.global_freemap, _ = self.cam_occupancy_map_global.get_global_free_map(robot_pos=self.agents.get_world_pose()[0],robot_height=1.7, update_camera_pose=False, verbose=verbose)
        return self.global_freemap, self.global_freemap_camera_pose

    def update_all_maps(self):
        # print("to avoid out of memory, doesn't update for debug")
        # return
        topdown_map = self.GlobalTopdownMap(self.vln_config, self.item['scan']) 
        freemap, camera_pose = self.get_global_free_map(verbose=self.vln_config.test_verbose) 
        topdown_map.update_map(freemap, camera_pose, verbose=self.vln_config.test_verbose) 
        self.get_surrounding_free_map(verbose = True)  
        pc, max_depth = self.update_semantic_map()
        self.update_obstacle_map(pc,max_depth)
        self.eval_helper.add_pos(self.agents.get_world_pose()[0])
    
    def get_pos_on_obstacle_map(self):
        pos = self.agents.get_world_pose()[0]
        row, col = self.ObstacleMap._xy_to_px(pos[0],pos[1])

    def warm_up(self,warm_step =50):
        self.step = 0
        env_actions = [{'h1': {'stand_still': []}}]
        while self.step < warm_step:
            self.env.step(actions=env_actions)
            self.step += 1
            if (self.step % 50 == 0):
                self.check_and_reset_robot(self.step, update_freemap=True, verbose=True)
        self.update_all_maps()
        log.info("Warm up finished, updated all maps")

    def from_obsmap_to_vlmap(self,pos):
        '''
        Input: pos = (row, col) in obsmap
        Output: pos_new in vlmap
        '''
        xy = self.ObstacleMap._px_to_xy(np.array(((pos[0],pos[1]))))[0]
        x, y = xy[0],xy[1]
        row,col = self.map.from_xyz_to_map(x,y)
        pos_new = [row, col]
        return pos_new
    
    def from_vlmap_to_obsmap(self, pos):
        '''
        Input: pos = (row, col) in vlmap
        Output: pos_new in obsmap
        '''
        xy = self.map.from_map_to_xyz(pos[0],pos[1])
        pos_new = self.ObstacleMap._xy_to_px(np.array([[xy[0],xy[1]]]))[0]
        return pos_new


    def check_environment(self, name: str) -> bool:
        """
        Check if an object exists in the existing map with one rotation
        """
        for i in range(0,360,60):
            exist_flag = self.map.check_object(name)
            if exist_flag:
                return True
            self.turn(60)
            

    def move_to(self, pos: Tuple[float, float], type = 'sem',threshold = 1.0,subgoal = None) -> List[str]:
        """Move the robot to the position on the obstacle map
            based on accurate localization in the environment
            with falls and movements

        Args:
            pos (Tuple[float, float]): (row, col) in semantic map

        Returns:
            List[str]: list of actions
        """
        # obs = self.get_observations(["rgba","depth"])
        # action_name = {}
        # self.agent_action_state = obs[self.task_name][self.robot_name][action_name]

        # set a certain pose on semantic map
        self._set_nav_curr_pose()
        # get a certain pose on semantic map
        curr_pose_on_full_map = self.get_agent_pose_on_map() # pos0, pos1,deg

        if type == 'sem':
            print('calls from object indexed from semantic map')
            print("transfering to obstacle map coord")
            goal = self.from_vlmap_to_obsmap(pos)
            # start = self.from_vlmap_to_obsmap(curr_pose_on_full_map[:2])
            position = self.agents.get_world_pose()[0][:2]
            start = self.ObstacleMap._xy_to_px(np.array([[position[0],position[1]]]))[0]
        else:
            print("calls from Frontier, pos already in Obs coord")
            goal = pos
            # start = self.from_vlmap_to_obsmap(curr_pose_on_full_map[:2])
            position = self.agents.get_world_pose()[0][:2]
            start = self.ObstacleMap._xy_to_px(np.array([[position[0],position[1]]]))[0]
        
        # nav should be built on obstacle map; 
        # !
        # self.nav.build_visgraph(self.ObstacleMap._navigable_map,
        #                   rowmin = 0,
        #                   colmin = 0,
        #                   vis = True)
        start_modified = [start[0],start[1]]
        goal_modified = [goal[0],goal[1]]
        goal_xy = self.ObstacleMap._px_to_xy(np.array([[goal[0],goal[1]]]))[0]


        if (np.linalg.norm(goal_xy - self.agents.get_world_pose()[0][:2]) <= threshold):
            log.warning("no need to move, already very close")
            return False
        # if goal is near past path, see as seen, neglect moving
        # only impliment when type is sem
        if (type == 'sem'):
            past_path = self.eval_helper.get_past_path()
            if (np.linalg.norm(goal_xy - np.array(past_path[-1][:2])) <= threshold):
                log.warning("goal is near past path, see as seen, neglect moving")
                return False

        paths, paths_3d = self.planning_path(start_modified,goal_modified)
        init_step = self.step
        actions = {'h1': {'move_along_path': [paths_3d]}} # paths should be [N ,3]
        log.info(f"actions: {actions}")
        log.info(f"moving from {start} to {goal} on {paths},moving from {self.agents.get_world_pose()[0][:2]} to {goal_xy} on {paths_3d}'")
        self.eval_helper.add_action(actions)

        while np.linalg.norm(goal_xy - self.agents.get_world_pose()[0][:2]) > threshold:
            # np.linalg.norm(self.from_vlmap_to_obsmap(curr_pose_on_full_map[:2]) - np.array(goal_modified) )
            # np.linalg.norm(goal_xy - self.agents.get_world_pose()[0][:2])>1:
            self.step = self.step + 1
            env_actions = []
            env_actions.append(actions)
            self.env.step(actions=env_actions)
            # log.info(f'action now {actions}')

            #! check whether robot falls first, then update map
            if (self.step % 200 == 0):
                while True:
                    reset_robot = self.check_and_reset_robot(cur_iter=self.step, update_freemap=False, verbose=self.vln_config.test_verbose)
                    reset_flag = reset_robot
                    if reset_flag:
                        # self.map.update_occupancy_map(verbose = self.vln_config.test_verbose) #! find dilate->vlmap occupancy map
                        # self._set_nav_curr_pose()
                        # # plan the path
                        # curr_pose_on_full_map = self.get_agent_pose_on_map()
                        # start = self.from_vlmap_to_obsmap(curr_pose_on_full_map[:2])

                        current_pos = self.agents.get_world_pose()[0][:2]
                        start = self.ObstacleMap._xy_to_px(np.array([[current_pos[0],current_pos[1]]]))[0]
                        start_modified = [start[0],start[1]]
                        log.info(f"stuck or fall down, reset the robot to {start_modified}")
                        paths, paths_3d = self.planning_path(start_modified,goal_modified)
                        goal_xy = paths_3d[-1][:2]
                        actions = {'h1': {'move_along_path': [paths_3d]}} # paths should be [N ,3]
                        log.info(f"moving from {start} to {goal_modified} on {paths}")
                        log.info(f'moving from {self.agents.get_world_pose()[0][:2]} to {goal_xy} on {paths_3d}')
                        self.eval_helper.add_action(actions)
                        self._retrive_robot_stuck_check()
                        for _ in range(50):
                            self.step += 1
                            self.env.step(actions=env_actions)
                    else:
                        break


            if (self.step % 200 == 0):
                ### check and reset robot
                self.update_all_maps()
                # self._set_nav_curr_pose()
                # curr_pose_on_full_map = self.get_agent_pose_on_map()
                # start = self.from_vlmap_to_obsmap(curr_pose_on_full_map[:2])
                current_pos = self.agents.get_world_pose()[0][:2]
                start = self.ObstacleMap._xy_to_px(np.array([[current_pos[0],current_pos[1]]]))[0]
                start_modified = [start[0],start[1]]
                log.info(f"Step {self.step}: In obstacle map coord, present at {start_modified}, need to navigate to {goal_modified}")


            if (self.step % 1000 == 0):
                # check whether path is blocked

                if self.nav.check_path_blocked(start_modified, goal_modified):
                    current_pos = self.agents.get_world_pose()[0][:2]
                    start = self.ObstacleMap._xy_to_px(np.array([[current_pos[0],current_pos[1]]]))[0]
                    start_modified = [start[0],start[1]]
                    goal_xy = env_actions[0]['h1']['move_along_path'][0][-1]
                    goal_modified = self.ObstacleMap._xy_to_px(np.array([[goal_xy[0],goal_xy[1]]]))[0]
                    log.warning("Path is blocked, replanning")
                    paths, paths_3d = self.planning_path(start_modified,goal_modified)
                    actions = {'h1': {'move_along_path': [paths_3d]}} # paths should be [N ,3]
                    goal_xy = paths_3d[-1][:2]
                    log.info(f"moving from {start} to {goal_modified} on {paths}")
                    log.info(f'moving from {self.agents.get_world_pose()[0][:2]} to {goal_xy} on {paths_3d}')
                    env_actions = []
                    env_actions.append(actions)
                    self.eval_helper.add_action(actions)
                    continue
            if ((self.step-init_step) % 1000 == 0):
                # end this loop
                if subgoal is not None:
                    if self.map.check_object(subgoal):
                        log.info(f"Subgoal {subgoal} is reached at step {self.step}")
                        return True

            if ((self.step-init_step) % 3000 == 0):
                log.warning("Failed to reach the subgoal after 1000 steps")
                goal = self.ultimate_goal
                if(self.map.check_object(goal)):
                    eval()
                    raise EarlyFound(f"Goal {goal} is reached early")
            if ((self.step-init_step) % 10000 == 0):
                log.warning("Failed to reach the subgoal after 10000 steps")
                raise TooManySteps("Too many steps")


        return True


    def planning_path(self,start_modified,goal_modified):
        '''
        will change goal_modified if goal is not reachable
        '''
        rows, cols = np.where(self.ObstacleMap._navigable_map == 0)
        min_row = np.max(np.min(rows)-1,0)
        min_col = np.max(np.min(cols)-1,0)
        self.nav.build_visgraph(self.ObstacleMap._navigable_map,
            rowmin = min_row,
            colmin = min_col,
            vis = True)

        path_save_path = self.nav_save_dir + f"/path_{self.step}.png"
        paths = self.nav.plan_to( [start_modified[1],start_modified[0]],[goal_modified[1],goal_modified[0]] , vis=self.config["nav"]["vis"],navigable_map_visual = self.ObstacleMap.nav_map_visual,save_path=path_save_path)
        paths = np.array(paths)
        paths = np.array([paths[:,1], paths[:,0]]).T # paths in normal order
        paths_3d = []
        for path in paths:
            xy = self.ObstacleMap._px_to_xy(np.array([[path[0], path[1]]]))[0]
            paths_3d.append([xy[0],xy[1],self.agent_init_pose[2]]) # fix height
        paths_3d = np.array(paths_3d)
        goal_modified[:] = paths[-1]
        return paths, paths_3d

    def turn(self, angle_deg: float,threshold = 0.1,vis = True):
        """
        Turn right a relative angle in degrees
        """
        angle_deg = -angle_deg
        if np.abs(angle_deg) < 5:
            log.warning(f'no need to turn for degree {angle_deg}')
            return False
        current_orientation = self.agents.get_world_pose()[1]
        current_orientation_in_degree = self.quat_to_euler_angles(current_orientation)
        current_yaw = current_orientation_in_degree[2] # ！ indeed in rot
        base_yaw = current_yaw 

        rotation_goals = [(current_yaw + degree)%(2*np.pi) - (2*np.pi) if (current_yaw + degree)%(2*np.pi) > np.pi else (self.quat_to_euler_angles(current_orientation)[2] + degree)%(2*np.pi) for degree in np.linspace( angle_deg / 180.0 * np.pi, 0, 10, endpoint=False)]
        # rotation_goals = [(current_yaw + degree) % 360 - 360 if (current_yaw + degree) % 360 > 180 else (current_yaw + degree) % 360 for degree in np.linspace(2 * angle_deg, 0, 360, endpoint=False)]
        # rotation_goals = [(current_yaw + degree) % 360 - 360 if (current_yaw + degree) % 360 > 180 else (current_yaw + degree) % 360 for degree in np.arange( angle_deg, 0, -5)] # [-180,180]

        log.info(f"turning from {current_yaw} to {base_yaw+angle_deg / 180.0 * np.pi}")
        while len(rotation_goals)>0:

            env_actions = []
            rotation_goal= rotation_goals.pop()
            actions = {
                    "h1": {
                        'rotate': [self.euler_angles_to_quat(np.array([0, 0, rotation_goal]))],
                    },
                }
            env_actions.append(actions)
            # log.info(f'action now {actions}')
            while abs(self.quat_to_euler_angles(current_orientation)[2] - rotation_goal) > threshold:
                self.step += 1
                # if step_time%100==0 or step_time <= 3:
                #     agent.bev_map.step_time = step_time
                #     obs = self.env.step(actions=actions, render = True)
                #     rgb, depth = agent.update_memory(dialogue_result=None, update_candidates= True, verbose=task_config['verbose']) 
                # else:
                #     obs = runner.step(actions=actions, render = False)
                self.env.step(actions= env_actions)
                current_orientation = self.agents.get_world_pose()[1]



                if (self.step % 200 == 0): # change from 50 to 200
                    if vis: 
                        self.update_all_maps()
                    log.info(f"Step {self.step}: Present at {self.quat_to_euler_angles(current_orientation)[2]}, need to navigate to {rotation_goal}")
                    #! fall down check
                    if (self.step % 1000 == 0):
                        reset_robot = self.check_and_reset_robot(cur_iter=self.step, update_freemap=False, verbose=self.vln_config.test_verbose)
                        reset_flag = reset_robot
                        if reset_flag:
                            # self.map.update_occupancy_map(verbose = self.vln_config.test_verbose) #! find dilate->vlmap occupancy map
                            self._set_nav_curr_pose()
                            # plan the path
                            curr_pose_on_full_map = self.get_agent_pose_on_map()  # TODO: (row, col, angle_deg) on full map
                            current_yaw = curr_pose_on_full_map[2]
                            # rotation_goals = [(current_yaw + degree) % 360 - 360 if (current_yaw + degree) % 360 > 180 else (current_yaw + degree) % 360 for degree in np.arange(angle_deg+base_yaw-current_yaw, 0, -2)]
      
                            rotation_goals = [(current_yaw + degree)%(2*np.pi) - (2*np.pi) if (current_yaw + degree)%(2*np.pi) > np.pi else (self.quat_to_euler_angles(current_orientation)[2] + degree)%(2*np.pi) for degree in np.linspace( (angle_deg / 180.0 * np.pi + base_yaw - current_yaw+4*np.pi)%(2*np.pi), 0, 10, endpoint=False)]
                            break    
        self._retrive_robot_stuck_check()
        return True

    
    def execute_actions(
        self,
        actions_list: List[str],
        poses_list: List[List[float]] = None,
        vis: bool = False,
    ) -> Tuple[bool, List[str]]:
        """
        Execute actions and check
        """
        raise NotImplementedError
        if poses_list is not None:
            assert len(actions_list) == len(poses_list)
        if vis:
            map = self.map.get_customized_obstacle_cropped()
            map = (map[:, :, None] * 255).astype(np.uint8)
            map = np.tile(map, [1, 1, 3])
            self.display_goals_on_map(map, 3, (0, 255, 0))
            if hasattr(self, "recorded_robot_pos") and len(self.recorded_robot_pos) > 0:
                map = self.display_full_map_pos_list_on_map(map, self.recorded_robot_pos)
            else:
                self.recorded_robot_pos = []

        real_actions_list = []
        for action_i, action in enumerate(actions_list):
            self._execute_action(action)

            real_actions_list.append(action)
            if vis:
                self.display_obs(waitkey=True,camera='pano_camera_180')
                self.display_curr_pos_on_map(map)
            if poses_list is None:
                continue
            row, col, angle = self._get_full_map_pose()
            if vis:
                self.recorded_robot_pos.append((row, col))
            # x, z = grid_id2base_pos_3d(self.gs, self.cs, col, row)
            # pred_x, pred_z, pred_angle = poses_list[action_i]
            # success = self._check_if_pose_match_prediction(x, z, pred_x, pred_z)
            # if not success:
            #     return success, real_actions_list
        return True, real_actions_list

    def pass_goal_bboxes(self, goal_bboxes: Dict[str, Any]):
        self.goal_bboxes = goal_bboxes

    def pass_goal_tf(self, goal_tfs: List[np.ndarray]):
        self.goal_tfs = goal_tfs

    def pass_goal_tf_list(self, goal_tfs: List[List[np.ndarray]]):
        self.all_goal_tfs = goal_tfs
        self.goal_id = 0

    def _execute_action(self, action: str):
        self.env.step(actions=action)

    def _check_if_pose_match_prediction(self, real_x: float, real_z: float, pred_x: float, pred_z: float):
        dist_thres = self.forward_dist
        dx = pred_x - real_x
        dz = pred_z - real_z
        dist = np.sqrt(dx * dx + dz * dz)
        return dist < dist_thres

    
    def _set_nav_curr_pose(self, type = 'sem'):
        """
        Set self.curr_pos_on_map and self.curr_ang_deg_on_map
        based on the simulator agent ground truth pose
        agent: 
        """
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        agent_state = self.agents.get_world_pose()
        pc_min = self.map.get_min_pcd()
        self.from_map_to_xyz(agent_state, pc_min) # modified from_habitat_tf
        row, col, angle_deg = self.to_full_map_pose() # dummy
        self.curr_pos_on_map = (row, col)
        self.curr_ang_deg_on_map = angle_deg
        print("set curr pose: ", row, col, angle_deg)
    


    def _get_full_map_pose(self) -> Tuple[float, float, float]:
        agent_state = self.agents.get_world_pose()
        # hab_tf = agent_state2tf(agent_state)
        pc_min = self.map.get_min_pcd()
        self.from_map_to_xyz(agent_state, pc_min)
        row, col, angle_deg = self.to_full_map_pose()
        return row, col, angle_deg

    def get_observations(self, data_types):
        ''' GEt observations from the sensors
        '''
        return self.env.get_observations()
    
    def get_robot_bottom_z(self):
        '''get robot bottom z'''
        return  self.env._runner.current_tasks[self.task_name].robots[self.robot_name].get_ankle_height() - self.sim_config.config_dict['tasks'][0]['robots'][0]['ankle_height']
        # return self.env._runner.current_tasks[self.task_name].robots[self.robot_name].get_ankle_base_z()-self.sim_config.config_dict['tasks'][0]['robots'][0]['ankle_height']
    

    def check_robot_fall(self, agent, pitch_threshold=35, roll_threshold=15, height_threshold=0.5, adjust=False, initial_pose=None, initial_rotation=None):
        '''
        Determine if the robot is falling based on its rotation quaternion.
        '''
        # current_quaternion = obs['orientation']
        current_position, current_quaternion = agent.get_world_pose() # orientation
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        roll, pitch, yaw = self.quat_to_euler_angles(current_quaternion)

        # Check if the pitch or roll exceeds the thresholds
        if abs(pitch) > pitch_threshold/180.0*np.pi or abs(roll) > roll_threshold/180.0*np.pi :
            is_fall = True
            log.info(f"Robot falls down!!!")
            log.info(f"Current Position: {current_position}, Orientation: {self.quat_to_euler_angles(current_quaternion)}")
        else:
            is_fall = False
        
        # Check if the height between the robot base and the robot ankle is smaller than a threshold
        robot_ankle_z = self.get_robot_bottom_z()
        robot_base_z = self.get_agent_pose()[0][2]
        if robot_base_z - robot_ankle_z < height_threshold:
            is_fall = True
            log.info(f"Robot falls down!!!")
            log.info(f"Current Position: {current_position}, Orientation: {self.quat_to_euler_angles(current_quaternion)}")

        return is_fall
    
    def get_agent_pose(self):
        return self.agents.get_world_pose()
    
    def _retrive_robot_stuck_check(self):
        self.agent_last_pose = None
    def check_robot_stuck(self, cur_iter, max_iter=300, threshold=0.3):
        ''' Check if the robot is stuck
        '''
        is_stuck = False
        if self.agent_last_pose is None:
            self.agent_last_valid_pose = self.get_agent_pose()[0]
            self.agent_last_pose, self.agent_last_rotation = self.get_agent_pose()
            self.agent_last_valid_pose = self.agent_last_pose
            self.agent_last_valid_rotation = self.agent_last_rotation
            self.stuck_threshold = 0
            self.stuck_last_iter = cur_iter
            return is_stuck

        current_pose, current_rotation = self.get_agent_pose()
        diff = np.linalg.norm(current_pose - self.agent_last_pose) + get_diff_beween_two_quat(current_rotation, self.agent_last_rotation)
        self.stuck_threshold += diff

        if (cur_iter - self.stuck_last_iter) >= max_iter:
            if self.stuck_threshold < threshold:
                is_stuck = True
            else:
                self.stuck_threshold = 0
                self.stuck_last_iter = cur_iter
                self.agent_last_valid_pose = current_pose
                self.agent_last_valid_rotation = current_rotation

        self.agent_last_pose = current_pose
        self.agent_last_rotation = current_rotation

        return is_stuck
    
    def randomly_pick_position_from_freemap(self, verbose=False):
        ''' Randomly pick a position from the free map
        '''
        # if self.surrounding_freemap_connected is None:
        #     self.get_surrounding_free_map(verbose=verbose)
        #     # return None
        self.get_surrounding_free_map(verbose=verbose)
        free_map = self.surrounding_freemap_connected
        camera_pose = self.surrounding_freemap_camera_pose

        freemap_center = np.array(free_map.shape) // 2
        free_indices = np.argwhere(free_map == 1)

        # Calculate distances from the center
        distances = np.linalg.norm(free_indices - freemap_center, axis=1)

        # Compute weights using a Gaussian function
        # sigma = 10 * np.std(distances)  # Standard deviation as a parameter for the Gaussian function
        # distance_weights = np.exp(-distances**2 / (2 * sigma**2))
        distance_weights = 1 / (1 + distances)

        # Compute density of free space around each free position
        # kernel = np.ones((7, 7))  # Example kernel size, adjust as needed
        # density_map = convolve(free_map.astype(float), kernel, mode='constant', cval=0.0)
        # kernel_size = 200  # Increase kernel size for smoother density map
        # density_map = gaussian_filter(free_map.astype(float), sigma=kernel_size/2)
        # density_weights = density_map[free_indices[:, 0], free_indices[:, 1]]

        # Combine distance-based weights with density-based weights
        # combined_weights = distance_weights * density_weights
        combined_weights = distance_weights

        # Normalize weights to sum to 1
        combined_weights /= combined_weights.sum()

        random_index = np.random.choice(free_indices.shape[0], p=combined_weights)
        random_position = free_indices[random_index]

        random_position = self.cam_occupancy_map_local.pixel_to_world(random_position, camera_pose)
        # random_position = [random_position[0], random_position[1], self.agent_last_valid_pose[2]]
        random_position = [random_position[0], random_position[1], self.agent_init_pose[2]]

        return random_position

    def reset_robot(self, position, rotation):
        ''' Reset the robot's pose
        '''
        self.agents.set_world_pose(position, rotation)
        self.agent_last_pose = position
        self.agents.set_joint_velocities(np.zeros(len(self.agents.dof_names)))
        self.agents.set_joint_positions(np.zeros(len(self.agents.dof_names)))
    

    
    def check_and_reset_robot(self, cur_iter, update_freemap=False, verbose=False):
        is_fall = self.check_robot_fall(self.agents, adjust=False)
        is_stuck = self.check_robot_stuck(cur_iter=cur_iter, max_iter=300, threshold=0.2)
        if (not is_fall) and (not is_stuck):
            if update_freemap:
                self.get_surrounding_free_map(verbose=verbose) # update the surrounding_free_map
                # ! using gt because in real life, a robot knows when it falls
            return False
        else:
            if is_fall:
                log.info("Robot falls down. Reset robot pose.")
            if is_stuck:
                log.info("Robot is stuck. Reset robot pose.")
            random_position = self.randomly_pick_position_from_freemap()
            # self.reset_robot(random_position, self.agent_last_valid_rotation)
            self.reset_robot(random_position, self.agent_init_rotation)
            log.info(f"Reset robot pose to {random_position}.")
            return True
        
    def get_surrounding_free_map(self, verbose=False):
        ''' Use top-down orthogonal camera to get the ground-truth surrounding free map
            This is useful to reset the robot's location when it get stuck or falling down.
        '''
        # agent_current_pose = self.get_agent_pose()[0]
        # agent_bottom_z = self.get_robot_bottom_z()
        robot_ankle = self.env._runner.current_tasks[self.task_name].robots[self.robot_name]._robot_right_ankle.get_world_pose()[0][2]
        self.surrounding_freemap, self.surrounding_freemap_connected = self.cam_occupancy_map_local.get_surrounding_free_map(robot_pos=self.get_agent_pose()[0],robot_height=robot_ankle, verbose=verbose)
        self.surrounding_freemap_camera_pose = self.cam_occupancy_map_local.topdown_camera.get_world_pose()[0]
    
    def set_ultimate_goal(self, goal:str):
        self.ultimate_goal = goal
    
    def save_metric(self):
        self.eval_helper.calculate_metric()
        self.eval_helper.save_single_task_metric(save_path = self.test_file_save_dir + '/metric.json')

    def clear_maps(self):
        self.map = None
        self.cam_occupancy_map_local = None
        self.cam_occupancy_map_global = None
        self.ObstacleMap = None




import os
import yaml
from pathlib import Path
from types import SimpleNamespace
import shutil
from grutopia.core.config import SimulatorConfig
import argparse
from vln.src.utils.utils import dict_to_namespace
#! ROOT_DIR is absolute
ROOT_DIR = "/ssd/xiaxinyuan/code/w61-grutopia"

def process_args(vln_total_config):
    '''Load configuration from YAML file
    vln_total_config: includes all necessary configs from w61-grutopia'''

    # Load the YAML configuration file
    '''Init simulation config'''
    sim_config = SimulatorConfig(vln_total_config.sim_cfg_file)

    '''Update VLN config'''
    with open(vln_total_config.vln_cfg_file, 'r') as f:
        vln_config = dict_to_namespace(yaml.load(f.read(), yaml.FullLoader))

    '''Init save directory'''
    for key, value in vln_total_config.items():
        setattr(vln_config, key, value)
    vln_config.root_dir = ROOT_DIR
    # vln_config.log_dir = os.path.join(ROOT_DIR, "logs")
    # vln_config.log_image_dir = os.path.join(vln_config.log_dir, "images", str(vln_config.env), str(vln_config.path_id))
    # if not os.path.exists(vln_config.log_image_dir):
    #     os.makedirs(vln_config.log_image_dir)
    
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


from vlmaps.vlmaps.utils.llm_utils import parse_object_goal_instruction, parse_spatial_instruction
import pickle


@hydra.main(
    version_base=None,
    config_path="../config_my",
    config_name="vlmap_dataset_cfg_docker.yaml",
)
def main(config: DictConfig) -> None:

    # Read episode IDs from the file
    # with open(config.episode_file, 'r') as f:
    #     episode_trajectory_pairs = [line.strip().split(',') for line in f.readlines()]
    # change the episode_trajectory_pairs to be a list of tuples
    try:
        with open(config.episode_file, 'r') as f:
            scan_trajectory_episode_pairs = [line.strip().split(',') for line in f.readlines()]
    except PermissionError:
        log.error(f"Permission denied when trying to read file: {config.episode_file}")
        log.error("Please check file permissions and ownership")
        sys.exit(1)
    except FileNotFoundError:
        log.error(f"Episode file not found: {config.episode_file}")
        log.error("Please check if the file path is correct")
        sys.exit(1)
    except IOError as e:
        log.error(f"IO Error when reading episode file: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Unexpected error when reading episode file: {e}")
        sys.exit(1)

    
    scan_trajectory_episode_pairs = [((scene_id), int(trajectory_id), int(episode_id)) for scene_id, trajectory_id, episode_id in scan_trajectory_episode_pairs]

    # record last trajectory id
    if config.resume_scan is not None:
        # 找到resume_scan对应的行号
        start_idx = None
        for idx, (scene_id,trajectory_id, episode_id) in enumerate(scan_trajectory_episode_pairs):
            if episode_id == config.resume_scan:
                start_idx = idx
                break
        if start_idx is None:
            log.error(f"Could not find scan {config.resume_scan} in episode_trajectory_pairs")
            return
        log.info(f"Resuming from index {start_idx} (scan {config.resume_scan})")
    else:
        start_idx = 0
    print(f"start_idx: {start_idx}")
    vln_envs, vln_config, sim_config, data_camera_list = build_dataset(config.vln_config)
    log.info(f'Is in container: {is_in_container()}')
    init_omni_scene = True
    reset_scene = False
    for split in vln_config.datasets.splits:
        robot = IsaacSimLanguageRobot(config, sim_config, vln_config=vln_config, split=split)
        last_scene_name = scan_trajectory_episode_pairs[start_idx][0]
        for scene_name, trajectory_id, episode_id in scan_trajectory_episode_pairs[start_idx:]: 
            ''' if the first episode: open the scene '''
            reset_scene =  (scene_name!= last_scene_name)
            print("scene_name", scene_name, "last_scene_name", last_scene_name)
            if reset_scene:
                # with open(config.last_scan_file, 'w') as f:
                #     f.write(str(episode_id))
                try:
                    # 确保父目录存在
                    os.makedirs(os.path.dirname(config.last_scan_file), exist_ok=True)
                    
                    # 写入文件
                    with open(config.last_scan_file, 'w') as f:
                        f.write(str(episode_id))
                    log.info(f"Successfully wrote scan {episode_id} to {config.last_scan_file}")
                except Exception as e:
                    log.error(f"Unexpected error while writing file: {e}")
                # sys.exit(1)
                if robot.env is not None and robot.env.simulation_app.is_running():
                    robot.env.simulation_app.close()
            robot.setup_scene(episode_id, trajectory_id,reset_scene=reset_scene,init_omni_scene=init_omni_scene)
            init_omni_scene = False
            # for the following episodes: if the new trajectory id is different from the last one, then use reset_scene()

            print('not moving for debug')
            # ! remember to align with isaac_robot.py
            # else: only set the task and robot position
            robot.map.init_categories(mp3dcat.copy())
            gpt_ans = parse_spatial_instruction(robot.instruction)
            parsed_instructions = extract_self_methods(gpt_ans)
            log.info(f"instruction: {robot.instruction}")
            log.info(f"parsed instructions: {parsed_instructions}")

            for idx in range(len(parsed_instructions) - 1, -1, -1):
                subgoal = parsed_instructions[idx]
                if not (('move_forward' in subgoal) or ('turn' in subgoal)):
                    fin_obj = extract_parameters(subgoal)
                    robot.set_ultimate_goal(fin_obj)
                    skipped_i = idx
                    break
            skip_flag = 0
            while robot.env.simulation_app.is_running():
                robot.warm_up(200)
                # robot.look_around()
                for cat_i, subgoal in enumerate(parsed_instructions):
                    if cat_i >= skip_flag:
                        log.info(f"Executing {subgoal}")
                        try:
                            robot.test_movement(subgoal)
                        except EarlyFound as e:
                            log.info(f"{e}. Found object early, stopping exploration.")
                            skip_flag = skipped_i
                        break

                break
            ''' execute the last spatial instruction'''
            for subgoal in parsed_instructions[skip_flag:]:
                log.info(f"Executing {subgoal}")
                robot.test_movement(subgoal)
            robot.save_metric()
            # robot.env.simulation_app.close()
            last_scene_name = scene_name
            robot.clear_maps()
            # if (episode_id > 100):
            #     break

if __name__ == "__main__":
    main()
