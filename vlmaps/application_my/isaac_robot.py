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
print(sys.path)

from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter


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
from vlmaps.vlmaps.utils.index_utils import find_similar_category_id, get_segment_islands_pos, get_dynamic_obstacles_map_3d
from vlmaps.vlmaps.utils.isaacsim_utils import  display_sample
from vlmaps.vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.vlmaps.utils.visualize_utils import pool_3d_label_to_2d

from typing import List, Tuple, Dict, Any, Union

from grutopia.core.env import BaseEnv
from grutopia.core.util.log import log
from vln.src.dataset.data_utils import load_data,load_scene_usd

from vln.src.utils.utils import  compute_rel_orientations,visualize_pc,get_diff_beween_two_quat




from vln.src.local_nav.BEVmap import BEVMap

from vlmaps.vlmaps.map.map import Map

from vlmaps.application_my.utils import NotFound

import logging
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
        self.test_file_save_dir = self.config["data_paths"]["test_file_save_dir"]
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

    def setup_scene(self,  episode_id: int ,trajectory_id: int):
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
        item = self._setup_sim(self.sim_config, episode_id,  trajectory_id, vlmap_dataset=self.online) # from VLNdataloader init_one_path

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
        
    def _setup_sim(self, sim_config, episode_id,path_id,  headless=False, vlmap_dataset=False):
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
                if 'stair' in instruction:
                    print('erroe!!! stair occurrs')
                    continue
                self.sim_config.config.tasks[0].scene_asset_path = scene_usd_path
                self.sim_config.config.tasks[0].robots[0].position = item["start_position"]
                self.sim_config.config.tasks[0].robots[0].orientation = item["start_rotation"] 
                self.vlmaps_data_dir = self.vlmaps_data_dir + f"/{item['scan']}/id_{item['episode_id']}"
                self.test_file_save_dir = self.test_file_save_dir + f"/{item['scan']}/id_{item['episode_id']}"
                if not os.path.exists(self.test_file_save_dir):
                    os.makedirs(self.test_file_save_dir, exist_ok=True)
                self.nav_save_dir = self.test_file_save_dir + "/nav"
                if not os.path.exists(self.nav_save_dir):
                    os.makedirs(self.nav_save_dir, exist_ok=True)
                self.init_env(self.sim_config, headless=self.vln_config.headless)
                self.init_omni_env()
                self.init_agents()
                self.init_cam_occunpancy_map(robot_prim=self.agents.prim_path,start_point=item["start_position"]) 
                self.init_occupancy_map()
                log.info("Initialized path id %d", episode_id)
                log.info("Scan: %s", item['scan'])
                log.info("Instruction: %s", item['instruction']['instruction_text'])
                self.instruction = item['instruction']['instruction_text']
                log.info(f"Start Position: {self.sim_config.config.tasks[0].robots[0].position}, Start Rotation: {self.sim_config.config.tasks[0].robots[0].orientation}")
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
        self.cam_occupancy_map_local = CamOccupancyMap(self.vln_config, robot_prim, start_point, local=True)
        self.cam_occupancy_map_global = CamOccupancyMap(self.vln_config, robot_prim, start_point, local=False)



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
        pc_filtered = pc[(0 < (camera_position[2]-pc[:,2])) & ((camera_position[2]-pc[:,2]) < 0.8)]
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
    
    def get_frontier(self):
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
        try:
            eval(action_name)
            return True
        except NotFound as e:
            log.warning(f"{e}. Object not found after looking around, moving to a frontier.")
            frontier_point = self.get_frontier()
            move_flag =  self.move_to(frontier_point,type = 'obs')            
            turn_angle = self.get_angle(frontier_point) # in obstacle map coord
            turn_flag = self.turn(-turn_angle) # turn left turn_angle
            if move_flag == False and turn_flag == False:
                log.warning(f"Failed to move to the frontier point {frontier_point}")
                return False
            return False

    def get_angle(self, frontier_point):
        """
        Get the angle between the robot and the frontier point
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
        current_pos = self.ObstacleMap._xy_to_px(np.array([[xy[0],xy[1]]]))[0]
        current_angle = self.ObstacleMap._get_current_angle_on_map(orientation_yaw + np.pi / 2) 
        target_rotation = np.arctan2(frontier_point[1] - current_pos[1], frontier_point[0] - current_pos[0]) / np.pi * 180 # already in [-pi,pi]

        return ((current_angle -target_rotation+180) % 360-180)
        # Calculate the angle between the robot and the frontier point
        

    def test_movement(self, action_name: str):
        """
        Tries to execute a movement action and explore if the action fails.
        If NotFound exception is raised, the robot starts exploring.
        """
        # prev_pos = self.curr_pos_on_map #! None
        prev_pos = np.array([0,0])
        # prev_ang = self.curr_ang_deg_on_map
        prev_ang = 0
        while True:
            try:
                self.map.load_3d_map()
                # 尝试执行传入的动作
                eval(action_name)
                
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
                log.warning(f"{e}. Object not found, starting exploration.")
                found = self.explore(action_name) # update occupancy map, semantic map
                if found == True:
                    break

            # except Exception as e:
            #     # 捕获其他异常并记录日志（如评估 action_name 失败）
            #     log.error(f"An error occurred during {action_name}: {e}. Starting exploration.")

            # 更新 prev_pos，以便在下一次迭代中继续比较位置变化
            prev_pos = self.curr_pos_on_map
            prev_ang = self.curr_ang_deg_on_map

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
        topdown_map = self.GlobalTopdownMap(self.vln_config, self.item['scan']) 
        freemap, camera_pose = self.get_global_free_map(verbose=self.vln_config.test_verbose) 
        topdown_map.update_map(freemap, camera_pose, verbose=self.vln_config.test_verbose) 

        pc, max_depth = self.update_semantic_map()
        self.update_obstacle_map(pc,max_depth)
    
    def get_pos_on_obstacle_map(self):
        pos = self.agents.get_world_pose()[0]
        row, col = self.ObstacleMap._xy_to_px(pos[0],pos[1])

    def warm_up(self,warm_step =50):
        self.step = 0
        env_actions = [{'h1': {'stand_still': []}}]
        while self.step < warm_step:
            self.env.step(actions=env_actions)
            self.step += 1
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


    def move_to(self, pos: Tuple[float, float], type = 'sem') -> List[str]:
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
            start = self.from_vlmap_to_obsmap(curr_pose_on_full_map[:2])
        else:
            print("calls from Frontier, pos already in Obs coord")
            goal = pos
            start = self.from_vlmap_to_obsmap(curr_pose_on_full_map[:2])
        
        # nav should be built on obstacle map; 
        # !
        # self.nav.build_visgraph(self.ObstacleMap._navigable_map,
        #                   rowmin = 0,
        #                   colmin = 0,
        #                   vis = True)
        start_modified = [start[0],start[1]]
        goal_modified = [goal[0],goal[1]]
        goal_xy = self.ObstacleMap._px_to_xy(np.array([[goal[0],goal[1]]]))[0]


        if (np.linalg.norm(goal_xy - self.agents.get_world_pose()[0][:2]) <= 1.0):
            log.warning("no need to move, already very close")
            return False

        actions = self.planning_path(start_modified,goal_modified)



        while np.linalg.norm(goal_xy - self.agents.get_world_pose()[0][:2]) > 1.0:
            # np.linalg.norm(self.from_vlmap_to_obsmap(curr_pose_on_full_map[:2]) - np.array(goal_modified) )
            # np.linalg.norm(goal_xy - self.agents.get_world_pose()[0][:2])>1:
            self.step = self.step + 1
            env_actions = []
            env_actions.append(actions)
            self.env.step(actions=env_actions)
            # log.info(f'action now {actions}')

            #! check whether robot falls first, then update map
            if (self.step % 200 == 0):
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
                    actions = {'h1': {'move_along_path': [paths_3d]}} # paths should be [N ,3]
                    log.info(f"moving from {start} to {goal} on {paths}")
                    log.info(f'moving from {self.agents.get_world_pose()[0][:2]} to {goal_xy} on {paths_3d}')
                    self._retrive_robot_stuck_check()


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
                current_pos = self.agents.get_world_pose()[0][:2]
                start = self.ObstacleMap._xy_to_px(np.array([[current_pos[0],current_pos[1]]]))[0]
                start_modified = [start[0],start[1]]
                goal = env_actions[0]['h1']['move_along_path'][0][-1]
                goal_xy = self.ObstacleMap._px_to_xy(np.array([[goal[0],goal[1]]]))[0]
                if self.nav.check_path_blocked(start_modified, goal_modified):
                    log.warning("Path is blocked, replanning")
                    paths, paths_3d = self.planning_path(start_modified,goal_modified)
                    actions = {'h1': {'move_along_path': [paths_3d]}} # paths should be [N ,3]
                    log.info(f"moving from {start} to {goal} on {paths}")
                    log.info(f'moving from {self.agents.get_world_pose()[0][:2]} to {goal_xy} on {paths_3d}')
                    env_actions = []
                    env_actions.append(actions)
                    continue


        return True


    def planning_path(self,start_modified,goal_modified):
        rows, cols = np.where(self.ObstacleMap._navigable_map == 0)
        min_row = np.max(np.min(rows)-1,0)
        min_col = np.max(np.min(cols)-1,0)
        self.nav.build_visgraph(self.ObstacleMap._navigable_map,
            rowmin = min_row,
            colmin = min_col,
            vis = True)
        
        path_save_path = self.nav_save_dir + f"/path_{self.step}.png"
        paths = self.nav.plan_to( start_modified, goal_modified , vis=self.config["nav"]["vis"],navigable_map_visual = self.ObstacleMap.nav_map_visual,save_dir=path_save_path)
        paths = np.array(paths)
        paths = np.array([paths[:,1], paths[:,0]]).T
        paths_3d = []
        for path in paths:
            xy = self.ObstacleMap._px_to_xy(np.array([[path[0], path[1]]]))[0]
            paths_3d.append([xy[0],xy[1],self.agent_init_pose[2]]) # fix height
        paths_3d = np.array(paths_3d)
        
        return paths, paths_3d

    def turn(self, angle_deg: float):
        """
        Turn right a relative angle in degrees
        """
        if np.abs(angle_deg) < 5:
            log.warning(f'no need to turn for degree {angle_deg}')
            return False
        current_orientation = self.agents.get_world_pose()[1]
        current_orientation_in_degree = self.quat_to_euler_angles(current_orientation)
        current_yaw = current_orientation_in_degree[2] # indeed in rot
        base_yaw = current_yaw 

        rotation_goals = [(current_yaw + degree)%(2*np.pi) - (2*np.pi) if (current_yaw + degree)%(2*np.pi) > np.pi else (self.quat_to_euler_angles(current_orientation)[2] + degree)%(2*np.pi) for degree in np.linspace( angle_deg / 180.0 * np.pi, 0, 10, endpoint=False)]
        # rotation_goals = [(current_yaw + degree) % 360 - 360 if (current_yaw + degree) % 360 > 180 else (current_yaw + degree) % 360 for degree in np.linspace(2 * angle_deg, 0, 360, endpoint=False)]
        # rotation_goals = [(current_yaw + degree) % 360 - 360 if (current_yaw + degree) % 360 > 180 else (current_yaw + degree) % 360 for degree in np.arange( angle_deg, 0, -5)] # [-180,180]

        log.info(f"turning from {current_yaw} to {base_yaw+angle_deg}")
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
            while abs(self.quat_to_euler_angles(current_orientation)[2] - rotation_goal) > 0.1:
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
        return self.env.get_observations(data_type=data_types)
















    ########################## from habitat language robot ###################################
    def get_agent_tf(self) -> np.ndarray:
        agent_state = self.sim.get_agent(0).get_state()
        return agent_state2tf(agent_state)

    def load_gt_region_map(self, region_gt: List[Dict[str, np.ndarray]]):
        obst_cropped = self.vlmaps_dataloader.get_obstacles_cropped()
        self.region_categories = sorted(list(region_gt.keys()))
        self.gt_region_map = np.zeros(
            (len(self.region_categories), obst_cropped.shape[0], obst_cropped.shape[1]), dtype=np.uint8
        )

        for cat_i, cat in enumerate(self.region_categories):
            for box_i, box in enumerate(region_gt[cat]):
                center = np.array(box["region_center"])
                size = np.array(box["region_size"])
                top_left = center - size / 2
                bottom_right = center + size / 2
                top_left_2d = self.vlmaps_dataloader.convert_habitat_pos_list_to_cropped_map_pos_list([top_left])[0]
                bottom_right_2d = self.vlmaps_dataloader.convert_habitat_pos_list_to_cropped_map_pos_list(
                    [bottom_right]
                )[0]

                self.gt_region_map[cat_i] = cv2.rectangle(
                    self.gt_region_map[cat_i],
                    (int(top_left_2d[1]), int(top_left_2d[0])),
                    (int(bottom_right_2d[1]), int(bottom_right_2d[0])),
                    1,
                    -1,
                )

    def get_distribution_map(
        self, name: str, scores: np.ndarray, pos_list_cropped: List[List[float]], decay_rate: float = 0.1
    ):
        if scores.shape[0] > 1:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        obst_map_cropped = self.map.get_customized_obstacle_cropped()
        dist_map = np.zeros_like(obst_map_cropped, dtype=np.float32)
        for pos_i, pos in enumerate(pos_list_cropped):
            # dist_map[int(pos[0]), int(pos[1])] = scores[pos_i]
            tmp_dist_map = np.zeros_like(dist_map, dtype=np.float32)
            pos = np.round(pos[0]), np.round(pos[1])
            tmp_dist_map[int(pos[0]), int(pos[1])] = scores[pos_i]

            con = scores[pos_i]
            dists = distance_transform_edt(tmp_dist_map == 0)
            reduct = con * dists * decay_rate
            tmp = np.ones_like(tmp_dist_map) * con - reduct
            tmp_dist_map = np.clip(tmp, 0, 1)
            dist_map += tmp_dist_map
        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(dist_map, name=name)
        return dist_map

    def get_distribution_map_3d(
        self, name: str, scores: np.ndarray, pos_list_3d: List[List[float]], decay_rate: float = 0.1
    ):
        """
        pos_list_3d: list of 3d positions in 3d map coordinate
        """
        if scores.shape[0] > 1:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        sim_mat = np.zeros((self.global_pc.shape[0], len(scores)))
        for pro_i, (con, pos) in enumerate(zip(scores, pos_list_3d)):
            print("confidence: ", con)

            dists = np.linalg.norm(self.global_pc[:, [0, 2]] - pos[[0, 2]], axis=1) / self.vlmaps_dataloader.cs
            sim = np.clip(con - decay_rate * dists, 0, 1)
            sim_mat[:, pro_i] = sim

        sim = np.max(sim_mat, axis=1)
        if self.config["nav"]["vis"]:
            self._vis_dist_map_3d(sim, name=name)

        return sim.flatten()

    def get_vl_distribution_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        predict_mask = self.map.get_predict_mask(name)
        predict_mask = predict_mask.astype(np.float32)
        predict_mask = (gaussian_filter(predict_mask, sigma=1) > 0.5).astype(np.float32)
        dists = distance_transform_edt(predict_mask == 0)
        tmp = np.ones_like(dists) - (dists * decay_rate)
        dist_map = np.where(tmp < 0, np.zeros_like(tmp), tmp)
        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(predict_mask, name=name + "_predict_mask")
            self._vis_dist_map(dist_map, name=name + f"_{decay_rate}")
        return dist_map

    def get_vl_distribution_map_3d(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        predict = np.argmax(self.map.scores_mat, axis=1)
        i = find_similar_category_id(name, self.map.categories)
        sim = predict == i

        target_pc = self.global_pc[sim, :]
        other_ids = np.where(sim == 0)[0]
        other_pc = self.global_pc[other_ids, :]
        target_sim = np.ones((target_pc.shape[0], 1))
        other_sim = np.zeros((other_pc.shape[0], 1))
        for other_p_i, p in enumerate(other_pc):
            dist = np.linalg.norm(target_pc - p, axis=1) / self.cs
            min_dist_i = np.argmin(dist)
            min_dist = dist[min_dist_i]
            other_sim[other_p_i] = np.clip(1 - min_dist * decay_rate, 0, 1)

        new_pc_global = self.global_pc.copy()
        new_sim = np.ones((new_pc_global.shape[0], 1), dtype=np.float32)
        for s_i, s in enumerate(other_sim):
            new_sim[other_ids[s_i]] = s

        if self.config["nav"]["vis"]:
            self._vis_dist_map_3d(new_sim, name=name)
        return new_sim.flatten()

    def get_region_distribution_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        if self.area_map_type == "clip_sparse":
            return self.get_clip_sparse_region_distribution_map(name, decay_rate)
        elif self.area_map_type == "concept_fusion":
            return self.get_concept_fusion_region_distribution_map(name, decay_rate)
        elif self.area_map_type == "lseg":
            return self.get_lseg_region_map(name, decay_rate)
        elif self.area_map_type == "gt":
            return self.get_gt_region_map(name, decay_rate)

    def get_gt_region_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        assert self.area_map_type == "gt"
        id = find_similar_category_id(name, self.region_categories)
        predict_mask = self.gt_region_map[id]

        obst_map_cropped = self.map.get_customized_obstacle_cropped()
        dist_map = np.zeros_like(obst_map_cropped, dtype=np.float32)
        dists = distance_transform_edt(predict_mask == 0)
        tmp = np.ones_like(dists) - (dists * decay_rate)
        dist_map = np.clip(tmp, 0, 1)
        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(dist_map, name=name)
        return dist_map

    def get_lseg_region_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        assert self.map_type == "lseg"
        assert self.area_map_type == "lseg"
        obst_map_cropped = self.map.get_customized_obstacle_cropped()
        dist_map = np.zeros_like(obst_map_cropped, dtype=np.float32)
        predict_mask = self.map.get_region_predict_mask(name)
        dists = distance_transform_edt(predict_mask == 0)
        tmp = np.ones_like(dists) - (dists * decay_rate)
        dist_map = np.clip(tmp, 0, 1)
        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(dist_map, name=name)
        return dist_map

    def get_concept_fusion_region_distribution_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        assert self.area_map is not None, "Area map is not initialized."
        obst_map_cropped = self.map.get_customized_obstacle_cropped()
        dist_map = np.zeros_like(obst_map_cropped, dtype=np.float32)
        predict_mask = self.area_map.get_predict_mask(name)
        # print("predict_mask: ", predict_mask.shape)
        # mask_vis = cv2.cvtColor((predict_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # cv2.imshow("mask_vis", mask_vis)
        # cv2.waitKey(1)
        dists = distance_transform_edt(predict_mask == 0)
        tmp = np.ones_like(dists) - (dists * decay_rate)
        dist_map = np.where(tmp < 0, np.zeros_like(tmp), tmp)
        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(dist_map, name=name)
        return dist_map

    def get_clip_sparse_region_distribution_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        assert self.area_map is not None, "Area map is not initialized."
        obst_map_cropped = self.map.get_customized_obstacle_cropped()
        dist_map = np.zeros_like(obst_map_cropped, dtype=np.float32)
        scores = self.area_map.get_scores(name)
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        robot_pose_list = self.area_map.get_robot_pose_list()
        ids = np.argsort(-scores.flatten())
        # max_id = np.argmax(scores)
        # obst_map = self.vlmaps_dataloader.get_obstacles_cropped_no_floor()
        # obst_map = np.tile(obst_map[:, :, None] * 255, [1, 1, 3]).astype(np.uint8)

        for i, tf_hab in enumerate(robot_pose_list):
            tmp_dist_map = np.zeros_like(dist_map, dtype=np.float32)
            row, col, deg = self.vlmaps_dataloader.convert_habitat_tf_to_cropped_map_pose(tf_hab)
            if row < 0 or row >= dist_map.shape[0] or col < 0 or col >= dist_map.shape[1]:
                continue
            # print(i, tf_hab[:3, 3].flatten(), row, col)
            # obst_map[int(row), int(col)] = (0, 255, 0)
            # cv2.circle(obst_map, (int(col), int(row)), 3, (0, 255, 0), -1)
            # cv2.imshow("obst_map", obst_map)
            # cv2.waitKey(1)
            s = scores[i]
            tmp_dist_map[row, col] = s
            dists = distance_transform_edt(tmp_dist_map == 0)
            tmp = np.ones_like(dists) * s - (dists * decay_rate)
            tmp_dist_map = np.clip(tmp, 0, 1)
            dist_map = np.where(dist_map > tmp_dist_map, dist_map, tmp_dist_map)

        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(dist_map, name=name)
        return dist_map

    def get_map(self, obj: str = None, sound: str = None):
        """
        Return the distribution map of a certain object or sound category with decay 0.01
        """
        assert obj is not None or sound is not None, "Object and sound names are both None."
        if obj is not None:
            return self.get_vl_distribution_map(obj, decay_rate=0.01)
        elif sound is not None:
            return self.get_sound_distribution_map(sound, decay_rate=0.01)

    def get_major_map(self, obj: str = None, sound: str = None):
        """
        Return the distribution map of a certain object or sound category with decay 0.1
        """
        assert obj is not None or sound is not None, "Object and sound names are both None."
        if obj is not None:
            return self.get_vl_distribution_map(obj, decay_rate=0.1)
        elif sound is not None:
            return self.get_sound_distribution_map(sound, decay_rate=0.1)

    def get_map_3d(self, obj: str = None, sound: str = None, img: np.ndarray = None, intr_mat: np.ndarray = None):
        """
        Return the distribution map of a certain object or sound category with decay 0.01
        """
        assert obj is not None or sound is not None or img is not None, "Object, sound names, and image are all None."
        if obj is not None:
            return self.get_vl_distribution_map_3d(obj, decay_rate=0.03)
        elif sound is not None:
            return self.get_sound_distribution_map_3d(sound, decay_rate=0.05)
        elif img is not None:
            return self.get_image_distribution_map_3d(img, query_intr_mat=intr_mat, decay_rate=0.05)

    def get_major_map_3d(self, obj: str = None, sound: str = None, img: np.ndarray = None, intr_mat: np.ndarray = None):
        """
        Return the distribution map of a certain object or sound category with decay 0.1
        """
        assert obj is not None or sound is not None or img is not None, "Object, sound names, and image are all None."
        if obj is not None:
            return self.get_vl_distribution_map_3d(obj, decay_rate=0.1)
        elif sound is not None:
            return self.get_sound_distribution_map_3d(sound, decay_rate=0.05)
        elif img is not None and intr_mat is not None:
            return self.get_image_distribution_map_3d(img, query_intr_mat=intr_mat, decay_rate=0.01)

    def _vis_dist_map(self, dist_map: np.ndarray, name: str = ""):
        obst_map = self.vlmaps_dataloader.get_obstacles_cropped_no_floor()
        obst_map = np.tile(obst_map[:, :, None] * 255, [1, 1, 3]).astype(np.uint8)
        target_heatmap = show_cam_on_image(obst_map.astype(float) / 255.0, dist_map)
        cv2.imshow(f"heatmap_{name}", target_heatmap)

    def _vis_dist_map_3d(self, heatmap: np.ndarray, transparency: float = 0.3, name: str = ""):
        print(f"heatmap of {name}")
        sim_new = (heatmap * 255).astype(np.uint8)
        rgb_pc = cv2.applyColorMap(sim_new, cv2.COLORMAP_JET)
        rgb_pc = rgb_pc.reshape(-1, 3)[:, ::-1].astype(np.float32) / 255.0
        rgb_pc = rgb_pc * transparency + self.map.grid_rgb / 255.0 * (1 - transparency)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.global_pc)
        pcd.colors = o3d.utility.Vector3dVector(rgb_pc)
        visualize_pc(pcd,headless=False) #! change different name
        # o3d.visualization.draw_geometries([pcd])

    def get_max_pos(self, map: np.ndarray) -> Tuple[float, float]:
        id = np.argmax(map)
        row, col = np.unravel_index(id, map.shape)

        if self.config["nav"]["vis"]:
            self._vis_dist_map(map, name="fuse")
        return row + self.vlmaps_dataloader.rmin, col + self.vlmaps_dataloader.cmin

    def get_max_pos_3d(self, heat: np.ndarray) -> Tuple[float, float, float]:
        id = np.argmax(heat)
        grid_map_pos_3d = self.map.grid_pos[id]
        return grid_map_pos_3d


    
    def display_obs(self, waitkey: bool = False,camera='pano_camera_180'):
        obs = self.get_observations("rgba")[self.task_name][self.robot_name][camera]
        display_sample(self.sim_setting, obs, waitkey=waitkey)

    def display_curr_pos_on_map(self, map: np.ndarray):
        row, col, angle = self._get_full_map_pose()
        self.vlmaps_dataloader.from_full_map_pose(row, col, angle)
        row, col, angle = self.vlmaps_dataloader.to_cropped_map_pose()
        map = cv2.circle(map, (int(col), int(row)), 3, (255, 0, 0), -1)
        cv2.imshow("real path", map)
        cv2.waitKey(1)

    def display_full_map_pos_list_on_map(self, map: np.ndarray, pos_list: List[List[float]]) -> np.ndarray:
        for pos_i, pos in enumerate(pos_list):
            self.vlmaps_dataloader.from_full_map_pose(pos[0], pos[1], 0)
            row, col, _ = self.vlmaps_dataloader.to_cropped_map_pose()
            map = cv2.circle(map, (int(col), int(row)), 3, (255, 0, 0), -1)
        return map

    def display_goals_on_map(
        self,
        map: np.ndarray,
        radius_pix: int = 3,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        if not hasattr(self, "goal_tfs") and not hasattr(self, "goal_bboxes"):
            return map
        if hasattr(self, "goal_tfs"):
            map = self.display_goal_tfs_on_map(map, radius_pix, color)
        elif hasattr(self, "goal_bboxes"):
            map = self.display_goal_bboxes_on_map(map, color)
        else:
            print("no goal tfs or bboxes passed to the robot.")
        return map

    def display_goal_bboxes_on_map(
        self,
        map: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        centers = self.goal_bboxes["centers"]
        sizes = self.goal_bboxes["sizes"]
        cs = self.vlmaps_dataloader.cs
        centers_cropped = self.vlmaps_dataloader.convert_habitat_pos_list_to_cropped_map_pos_list(centers)
        for center_i, center in enumerate(centers_cropped):
            size = [float(x) / cs for x in sizes[center_i]]  # in habitat robot coord
            size = size[[2, 0]]
            min_corner, max_corner = get_bbox(center, size)
            rmin, rmax = int(min_corner[0]), int(max_corner[0])
            cmin, cmax = int(min_corner[1]), int(max_corner[1])
            cv2.rectangle(map, (cmin, rmin), (cmax, rmax), color, 2)

        return map

    def display_goal_tfs_on_map(
        self,
        map: np.ndarray,
        radius_pix: int = 3,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        if self.goal_tfs is None:
            if self.all_goal_tfs is None:
                print("no goal tfs passed to the robot")
                return map
            self.goal_tfs = self.all_goal_tfs[self.goal_id]
            self.goal_id += 1

        for tf_i, tf in enumerate(self.goal_tfs):
            self.vlmaps_dataloader.from_habitat_tf(tf)
            (row, col, angle_deg) = self.vlmaps_dataloader.to_cropped_map_pose()
            cv2.circle(map, (col, row), radius_pix, color, 2)
        self.goal_tfs = None
        return map
    
    def get_robot_bottom_z(self):
        '''get robot bottom z'''
        return self.env._runner.current_tasks[self.task_name].robots[self.robot_name].get_ankle_base_z()-self.sim_config.config_dict['tasks'][0]['robots'][0]['ankle_height']
    

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
        self.surrounding_freemap, self.surrounding_freemap_connected = self.cam_occupancy_map_local.get_surrounding_free_map(robot_pos=self.get_agent_pose()[0],robot_height=1.65, verbose=verbose)
        self.surrounding_freemap_camera_pose = self.cam_occupancy_map_local.topdown_camera.get_world_pose()[0]




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


from vlmaps.vlmaps.utils.llm_utils import parse_object_goal_instruction
import pickle


@hydra.main(
    version_base=None,
    config_path="../config_my",
    config_name="vlmap_dataset_cfg.yaml",
)
def main(config: DictConfig) -> None:
    vln_envs, vln_config, sim_config, data_camera_list = build_dataset(config.vln_config)
    # with open('sim_config.pkl','rb') as f:
    #     sim_config=pickle.load(f)
    # with open('vln_config.pkl','rb') as f:
    #     vln_config=pickle.load(f)

    # with open('api_key/openai_api_key.txt','r') as f:
    #     api_key = f.read().strip()
    # os.environ["OPENAI_KEY"] = api_key

    for split in vln_config.datasets.splits:
        # data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
        robot = IsaacSimLanguageRobot(config,sim_config,vln_config = vln_config, split= split)
        robot.setup_scene(config.episode_id,config.trajectory_id)
        robot.map.init_categories(mp3dcat.copy())
        object_categories = parse_object_goal_instruction(robot.instruction)
        # "Enter the bedroom and go around the bed. Go to the closet on the far wall on the left. Stop in the doorway"
        # object_categories = ["bedroom", "bed", "closet", "doorway"]
        # object_categories = ["bed", "closet", "doorway"]
        print("object categories", object_categories)
        print(f"instruction: {robot.instruction}")

        while robot.env.simulation_app.is_running():
            robot.warm_up(200)
            # robot.turn(-90)
            
            for cat_i, cat in enumerate(object_categories):
                print(f"Navigating to category {cat}")
                # robot.move_to_object(cat)
                robot.test_movement(f'self.move_to_object("{cat}")')
                #! already moved, missing goal achieved parameter
            break
        robot.env.simulation_app.close()
        # hab_tf = cvt_pose_vec2tf(robot.vlmaps_dataloader.base_poses[0])
        # robot.set_agent_state(hab_tf)
        # obs = robot.get_observations(data_types=["rgba","depth"])
        # rgb = obs["color_sensor"]
        # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # cv2.imshow("scr rgb", bgr)
        # cv2.waitKey(1)

        # tar_hab_tf = cvt_pose_vec2tf(robot.vlmaps_dataloader.base_poses[800])
        # robot.set_agent_state(tar_hab_tf)
        # obs = robot.sim.get_sensor_observations(0)
        # rgb = obs["color_sensor"]
        # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # cv2.imshow("tar rgb", bgr)
        # cv2.waitKey(1)

        # robot.set_agent_state(hab_tf)
        # robot.vlmaps_dataloader.from_habitat_tf(tar_hab_tf)
        # tar_row, tar_col, tar_angle_deg = robot.vlmaps_dataloader.to_full_map_pose()
        # robot.empty_recorded_actions()
        # robot.pass_goal_tf([tar_hab_tf])
        # robot.move_to((tar_row, tar_col))


if __name__ == "__main__":
    main()
