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

from vln.src.utils.utils import euler_angles_to_quat, quat_to_euler_angles, compute_rel_orientations,visualize_pc

from vlmaps.vlmaps.map.map import Map

from vlmaps.application_my.utils import NotFound

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
        self.vlmaps_data_save_dirs = self.config["data_paths"]["vlmaps_data_dir"]
        self.test_file_save_dir = self.config["data_paths"]["test_file_save_dir"]
        # self.vlmaps_data_save_dirs = [
        #     data_dir / x for x in sorted(os.listdir(data_dir)) if x != ".DS_Store"
        # ]  # ignore artifact generated in MacOS
        self.map_type = self.config["params"]["map_type"] # from params
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
        self.vlmap_dataset = self.config["vlmap_dataset"]
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
        
        self.bev = None
        self.surrounding_freemap_connected = None
        self.step = 0
    
    #! not in use !!!
    def scene_id2scene_name(self,scene_id):
        '''
        scene id: 0; scene name: 5J......
        '''
        self.scene_id = scene_id
        vlmaps_data_dir = self.vlmaps_data_dirs[scene_id]
        print(vlmaps_data_dir)
        self.scene_name = vlmaps_data_dir.name.split("_")[0]

    def setup_scene(self, scene_id: int):
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
        item = self._setup_sim(self.sim_config, scene_id,vlmap_dataset=self.vlmap_dataset) # from VLNdataloader init_one_path

        self.item = item
        # vlmaps_data_dir = self.vlmaps_data_save_dirs + f"/{item['scan']}/id_{item['trajectory_id']}"
        self.setup_map(self.vlmaps_data_save_dirs)

        cropped_obst_map = self.map.get_obstacle_cropped()
        if self.config.map_config.potential_obstacle_names and self.config.map_config.obstacle_names:
            print("come here")
            self.map.customize_obstacle_map(
                self.config.map_config.potential_obstacle_names,
                self.config.map_config.obstacle_names,
                vis=self.config.nav.vis,
            )
            cropped_obst_map = self.map.get_customized_obstacle_cropped()

        self.nav.build_visgraph(
            cropped_obst_map,
            self.vlmaps_dataloader.rmin,
            self.vlmaps_dataloader.cmin,
            vis=self.config["nav"]["vis"],
        )

        # self._setup_localizer(vlmaps_data_dir)

    def load_scene_map(self, data_dir: str, map_config: DictConfig):
        self.map = Map.create(map_config) #! should include isaacmap!!!
        self.map.load_map(data_dir,self.test_file_save_dir)
        self.map.generate_obstacle_map()


    def setup_map(self, vlmaps_data_dir: str):
        self.load_scene_map(vlmaps_data_dir, self.config["map_config"])

        # TODO: check if needed
        if "3d" in self.config.map_config.map_type:
            self.map.init_categories(mp3dcat.copy())
            self.global_pc = grid_id2base_pos_3d_batch(self.map.grid_pos, self.cs, self.gs)

        self.vlmaps_dataloader = VLMapsDataloaderHabitat(vlmaps_data_dir, self.config.map_config, map=self.map)

    def init_omni_env(self):
        rotations_utils = importlib.import_module("omni.isaac.core.utils.rotations")
        self.quat_to_euler_angles = rotations_utils.quat_to_euler_angles
        self.euler_angles_to_quat = rotations_utils.euler_angles_to_quat
        # from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat

    def init_env(self, sim_config, headless=True):
        '''init env''' 
        self.env = BaseEnv(sim_config, headless=headless, webrtc=False)
    
    def set_agent_pose(self, position, rotation):
        self.agents.set_world_pose(position, rotation)
    
    def init_agents(self):
        '''call after self.init_env'''
        self.agents = self.env._runner.current_tasks[self.task_name].robots[self.robot_name].isaac_robot
        self.agent_last_pose = None
        self.agent_init_pose = self.sim_config.config.tasks[0].robots[0].position
        self.agent_init_rotation = self.sim_config.config.tasks[0].robots[0].orientation

        self.set_agent_pose(self.agent_init_pose, self.agent_init_rotation)
        
    def _setup_sim(self, sim_config, path_id,headless=False, vlmap_dataset=False):
        """
        Setup IsaacSim simulator, load IsaacSim scene and relevant mesh data
        """
        # Demo for visualizing simply one path
        if vlmap_dataset==False:
            for item in self.data:
                if item['trajectory_id'] == path_id:
                    scene_usd_path = load_scene_usd(self.vln_config, item['scan'])
                    instruction = item['instruction']['instruction_text']
                    if 'stair' in instruction:
                        continue
                    self.sim_config.config.tasks[0].scene_asset_path = scene_usd_path
                    self.sim_config.config.tasks[0].robots[0].position = item["start_position"]
                    self.sim_config.config.tasks[0].robots[0].orientation = item["start_rotation"] 
                    self.init_env(self.sim_config, headless=self.vln_config.headless)
                    self.init_omni_env()
                    self.init_agents()
                    self.init_cam_occunpancy_map(robot_prim=self.agents.prim_path,start_point=item["start_position"]) 
                    log.info("Initialized path id %d", path_id)
                    log.info("Scan: %s", item['scan'])
                    log.info("Instruction: %s", item['instruction']['instruction_text'])
                    log.info(f"Start Position: {self.sim_config.config.tasks[0].robots[0].position}, Start Rotation: {self.sim_config.config.tasks[0].robots[0].orientation}")
                    return item
            log.error("Path id %d not found in the dataset", path_id)
            return None
        else:
            for item in self.data:
                if item['scan'] in self.vlmaps_data_save_dirs.split("/")[-1]:
                    # item['scan]: s8pc...
                    scene_usd_path = load_scene_usd(self.vln_config, item['scan'])
                    self.sim_config.config.tasks[0].scene_asset_path = scene_usd_path
                    self.sim_config.config.tasks[0].robots[0].position = item["start_position"]
                    self.sim_config.config.tasks[0].robots[0].orientation = item["start_rotation"] 
                    self.init_env(self.sim_config, headless=self.vln_config.headless)
                    self.init_omni_env()
                    self.init_agents()
                    self.init_cam_occunpancy_map(robot_prim=self.agents.prim_path,start_point=item["start_position"]) 
                    log.info("Initialized path id %d", item['trajectory_id'])
                    log.info("Scan: %s", item['scan'])
                    log.info("Instruction: %s", item['instruction']['instruction_text'])
                    log.info(f"Start Position: {self.sim_config.config.tasks[0].robots[0].position}, Start Rotation: {self.sim_config.config.tasks[0].robots[0].orientation}")
                    return item
            log.error("Path id %d not found in the dataset", path_id)
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

    def test_movement(self, action_name: str):
        """
        Tries to execute a movement action and explore if the action fails.
        If NotFound exception is raised, the robot starts exploring.
        """
        prev_pos = self.curr_pos_on_map
        prev_ang = self.curr_ang_deg_on_map
        while True:
            try:
                # 尝试执行传入的动作
                eval(action_name)
                
                # 如果位置发生变化，说明动作成功，退出循环
                if not (is_equal(self.curr_pos_on_map, prev_pos) and is_equal(self.curr_ang_deg_on_map, prev_ang)):
                    log.info(f"Successfully executed {action_name}")
                    break
                else:
                    # 如果位置没有发生变化，记录日志并进行探索
                    log.info(f"Robot didn't move after executing {action_name}, start exploring")
                    self.explore()

            except NotFound as e:
                # 捕获 NotFound 异常，记录日志并进行探索
                log.warning(f"{e}. Object not found, starting exploration.")
                self.explore()

            except Exception as e:
                # 捕获其他异常并记录日志（如评估 action_name 失败）
                log.error(f"An error occurred during {action_name}: {e}. Starting exploration.")
                self.explore()

            # 更新 prev_pos，以便在下一次迭代中继续比较位置变化
            prev_pos = self.curr_pos_on_map
            prev_ang = self.curr_ang_deg_on_map

    def move_to_object(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_nearest_pos(self.curr_pos_on_map, name)


        self.move_to(pos)

    def move_to(self, pos: Tuple[float, float]) -> List[str]:
        """Move the robot to the position on the full map
            based on accurate localization in the environment
            with falls and movements

        Args:
            pos (Tuple[float, float]): (row, col) on full map

        Returns:
            List[str]: list of actions
        """
        env_actions = []
        #! init: if i< warm_step: agent_action_state{'finished': True}
        obs = self.get_observations(["rgba","depth"])
        action_name = {}
        self.agent_action_state = obs[self.task_name][self.robot_name][action_name]

                    # set a certain pose
        self._set_nav_curr_pose()
            # plan the path
        curr_pose_on_full_map = self.get_agent_pose_on_map()  # TODO: (row, col, angle_deg) on full map
        paths = self.nav.plan_to(curr_pose_on_full_map[:2], pos, vis=self.config["nav"]["vis"])   
        actions = {'h1': {'move_along_path': [paths]}} # paths should be [N ,3]
        env_actions.append(actions)

        while True:
            self.env.step(actions=env_actions)
            self.step = self.step+1

            # check and reset roboet every 10 steps:
            if (self.step % 10 == 0):
                ### check and reset robot
                topdown_map = self.GlobalTopdownMap(self.vln_config, self.item['scan']) 
                freemap, camera_pose = self.get_global_free_map(verbose=self.vln_config.test_verbose) 
                topdown_map.update_map(freemap, camera_pose, verbose=self.vln_config.test_verbose) 

                reset_robot = self.check_and_reset_robot(cur_iter=self.step, update_freemap=False, verbose=self.vln_config.test_verbose)
                reset_flag = reset_robot
                if reset_flag:
                    self.map.update_occupancy_map(verbose = self.vln_config.test_verbose) #! find dilate->vlmap occupancy map
                    robot_current_pos = self.agents.get_world_pose()[0]
                    self._set_nav_curr_pose()
                    # plan the path
                    curr_pose_on_full_map = self.get_agent_pose_on_map()  # TODO: (row, col, angle_deg) on full map
                    paths = self.nav.plan_to(
                    curr_pose_on_full_map[:2], pos, vis=self.config["nav"]["vis"]
                )   
                    actions = {'h1': {'move_along_path': [paths]}} # paths should be [N ,3]
                    env_actions.append(actions)

            if (self.step % 100 == 0):
                if not reset_flag:
                    if self.agent_action_state['finished']:
                        break

    def turn(self, angle_deg: float):
        """
        Turn right a relative angle in degrees
        """
        current_orientation = self.agents.get_world_pose()[1]
        step_time = 0
        rotation_goals = [(quat_to_euler_angles(current_orientation)[2] + degree)%(2*np.pi) - (2*np.pi) if (quat_to_euler_angles(current_orientation)[2] + degree)%(2*np.pi) > np.pi else (quat_to_euler_angles(current_orientation)[2] + degree)%(2*np.pi) for degree in np.linspace(2 * angle_deg / 180.0 * np.pi, 0, 10, endpoint=False)]
        while len(rotation_goals)>0:
            rotation_goal= rotation_goals.pop()
            while abs(quat_to_euler_angles(current_orientation)[2] - rotation_goal) > 0.1:
                step_time += 1
                actions = {
                    "h1": {
                        'rotate': [euler_angles_to_quat(np.array([0, 0, rotation_goal]))],
                    },
                }
                # if step_time%100==0 or step_time <= 3:
                #     agent.bev_map.step_time = step_time
                #     obs = self.env.step(actions=actions, render = True)
                #     rgb, depth = agent.update_memory(dialogue_result=None, update_candidates= True, verbose=task_config['verbose']) 
                # else:
                #     obs = runner.step(actions=actions, render = False)
                self.step(actions= actions)
                current_orientation = self.agent.get_world_pose()[1]
                #! need fall down check
                # if (obs['h1']['position'][2] < task_config['fall_threshold']) or step_time>1000: # robot falls down
                #     break

    
    def execute_actions(
        self,
        actions_list: List[str],
        poses_list: List[List[float]] = None,
        vis: bool = False,
    ) -> Tuple[bool, List[str]]:
        """
        Execute actions and check
        """
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

    def _set_nav_curr_pose(self):
        """
        Set self.curr_pos_on_map and self.curr_ang_deg_on_map
        based on the simulator agent ground truth pose
        agent: 
        """
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        agent_state = self.agents.get_world_pose()

        self.vlmaps_dataloader.from_habitat_tf(agent_state)
        row, col, angle_deg = self.vlmaps_dataloader.to_full_map_pose() # dummy
        self.curr_pos_on_map = (row, col)
        self.curr_ang_deg_on_map = angle_deg
        print("set curr pose: ", row, col, angle_deg)

    def _get_full_map_pose(self) -> Tuple[float, float, float]:
        agent_state = self.agents.get_world_pose()
        # hab_tf = agent_state2tf(agent_state)
        self.vlmaps_dataloader.from_habitat_tf(agent_state)
        row, col, angle_deg = self.vlmaps_dataloader.to_full_map_pose()
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
    
    def check_robot_fall(self, agent, pitch_threshold=35, roll_threshold=15, height_threshold=0.5, adjust=False, initial_pose=None, initial_rotation=None):
        '''
        Determine if the robot is falling based on its rotation quaternion.
        '''
        # current_quaternion = obs['orientation']
        current_position, current_quaternion = agent.get_world_pose() # orientation
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        roll, pitch, yaw = self.quat_to_euler_angles(current_quaternion,degrees=True)

        # Check if the pitch or roll exceeds the thresholds
        if abs(pitch) > pitch_threshold or abs(roll) > roll_threshold:
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
    
    def check_robot_stuck(self, cur_iter, max_iter=300, threshold=0.2):
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
        diff = np.linalg.norm(current_pose - self.agent_last_pose)
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

        return is_stuck
    
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
        robot.setup_scene(config.scene_id)
        robot.map.init_categories(mp3dcat.copy())
        # object_categories = parse_object_goal_instruction(robot.instruction)
        object_categories = ["bedroom", "bed", "closet", "doorway"]
        print("object categories", object_categories)
        print(f"instruction: {robot.instruction}")

        for cat_i, cat in enumerate(object_categories):
            print(f"Navigating to category {cat}")
            robot.move_to_object(cat)
            #! already moved, missing goal achieved parameter
            #! for object navigation, doesn't have gt value for any object
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
