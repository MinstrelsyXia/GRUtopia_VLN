from vln.src.models.init_policy import initialize_policy
from vln.src.utils.logger import MyLogger
from vln.src.utils.utils import Config
import logging
import os
import torch
from vln.src.dataset.data_utils_multi_env import load_gather_data
import lmdb
import msgpack_numpy
import sys
from grutopia.core.config import SimulatorConfig
from vln.src.dataset.data_utils_multi_env import load_scene_usd
import numpy as np
from grutopia.core.env import BaseEnv
from vln.src.local_nav.path_planner import AStarPlanner
import time
import importlib
from vln.src.models.utils.feature_extract import extract_instruction_tokens
from vln.src.utils.utils import batch_obs
import tqdm
import math

def world_to_pixel(world_pose, camera_pose,aperture,width,height):
    cx, cy = camera_pose[0]*10/aperture*width, -camera_pose[1]*10/aperture*height

    X, Y = world_pose[0]*10/aperture*width, -world_pose[1]*10/aperture*height
    pixel_x = width - (X - cx + width/2)
    pixel_y = Y - cy + height/2

    return [pixel_x, pixel_y]
def pixel_to_world(pixel_pose,camera_pose,aperture,width,height):
    cx, cy = camera_pose[0]*10/aperture*width, -camera_pose[1]*10/aperture*height
    px = height - pixel_pose[0] + cx - height/2
    py = pixel_pose[1] + cy - width/2

    world_x = px/10/height*aperture
    world_y = -py/10/width*aperture

    return [world_x, world_y]
def get_shortest_path(
    camera_pose,
    robot_position,
    reference_path,
    path_planner,
):
    freemap, _ = occupancy_map.get_global_free_map(robot_pos=robot_position, robot_height=1.55, update_camera_pose=False, verbose=False)
    topdown_map.update_map(freemap, camera_pose, verbose=False, env_idx=0, update_map=True)
    occupancy_map_, _ = topdown_map.get_map(robot_position, return_camera_pose=True)
    start = time.time()
    start_pixel = world_to_pixel(reference_path[0],camera_pose,200,500,500)
    goal_pixel = world_to_pixel(reference_path[-1],camera_pose,200,500,500)
    paths, find_flag = path_planner.planning(
        start_pixel[0], 
        start_pixel[1],
        goal_pixel[0], 
        goal_pixel[1],
        obs_map=occupancy_map_,
    )
    end = time.time()
    print(f"old planning 耗时{round(end - start,2)}秒")
    exe_path = []
    if find_flag:
        for node in paths:
            world_coords = pixel_to_world([node[0],node[1]], camera_pose,200,500,500)
            exe_path.append([world_coords[0], world_coords[1], reference_path[0][2]])
    if exe_path is not None and len(exe_path)>1:
        exe_path.pop(0)
    if exe_path is not None:
        shortest_path_length = 0
        for i in range(len(exe_path)-1):
            # 计算相邻两点之间的欧氏距离
            shortest_path_length += np.linalg.norm(np.array(exe_path[i+1]) - np.array(exe_path[i]))
    else:
        shortest_path_length = 0
    return exe_path, shortest_path_length
def norm_depth(depth_info, min_depth=0, max_depth=10):
    depth_info[depth_info > max_depth] = max_depth
    depth_info = (depth_info - min_depth) / (max_depth - min_depth)
    return depth_info
def get_obs(env, instruction,robot_position,robot_rotation):
    obs = env.get_observations(add_rgb_subframes=True)
    obs_data = {}
    obs_data['globalgps'] = None
    obs_data['global_rotation'] = None
    obs_data['globalyaw'] = None
    obs_data['rgb'] = None
    obs_data['depth'] = None
    obs_data['instruction'] = instruction['instruction_text']
    if "instruction_tokens" in instruction:
        obs_data['instruction_tokens'] = instruction['instruction_tokens']
    obs_data['step'] = 0
    cur_obs = obs['vln_0']["h1_0"]['pano_camera_0']
    rgb_info = cur_obs['rgba'][..., :3]
    depth_info = norm_depth(cur_obs['depth'])
    obs_data['rgb'] = rgb_info
    obs_data['depth'] = depth_info[..., np.newaxis]
    rotations_utils = importlib.import_module("omni.isaac.core.utils.rotations")
    quat_to_euler_angles = rotations_utils.quat_to_euler_angles
    _,_, yaw = quat_to_euler_angles(robot_rotation)
    obs_data['globalgps'] = np.array(robot_position)
    obs_data['global_rotation'] = np.array(robot_rotation)
    obs_data['globalyaw'] = yaw
    return [obs_data]
def check_robot_fall(robot_position, robot_rotation, robots_bottom_z, pitch_threshold=35, roll_threshold=15, height_threshold=0.5):
    from grutopia.core.util.log import log
    from omni.isaac.core.utils.rotations import quat_to_euler_angles
    roll, pitch, yaw = quat_to_euler_angles(robot_rotation, degrees=True)
    # Check if the pitch or roll exceeds the thresholds
    if abs(pitch) > pitch_threshold or abs(roll) > roll_threshold:
        is_fall = True
        log.info(f"Robot falls down!!!")
        log.info(f"Current Position: {robot_position}, Orientation: {roll, pitch, yaw}")
    else:
        is_fall = False
    
    # Check if the height between the robot base and the robot ankle is smaller than a threshold
    robot_ankle_z = robots_bottom_z
    robot_base_z = robot_position[2]
    if robot_base_z - robot_ankle_z < height_threshold:
        is_fall = True
        log.info(f"Robot falls down!!!")
        log.info(f"Current Position: {robot_position}, Orientation: {roll, pitch, yaw}")
    return is_fall
class StuckChecker:
    def __init__(self, offset, isaac_robot):
        self.offset = offset
        self.last_iter = 0
        position, rotation = isaac_robot.get_world_pose()
        self.agent_last_position = position
        self.agent_last_rotation = rotation

    def check_robot_stuck(self, robot_position, robot_rotation, cur_iter, max_iter=300, threshold=0.2, rotation_threshold = math.pi / 12):
        ''' Check if the robot is stuck
        '''
        robot_position = robot_position - self.offset
        if (cur_iter - self.last_iter) <= max_iter:
            return False
        from omni.isaac.core.utils.rotations import quat_to_euler_angles
        position_diff = np.linalg.norm(robot_position[:2] - self.agent_last_position[:2])
        rotation_diff = abs(quat_to_euler_angles(robot_rotation)[2] - quat_to_euler_angles(self.agent_last_rotation)[2])
        if position_diff < threshold and rotation_diff < rotation_threshold:
            return True
        else:
            self.position_diff = 0
            self.rotation_diff = 0
            self.last_iter = cur_iter
            self.agent_last_position = robot_position
            self.agent_last_rotation = robot_rotation
            return False

class ActionExecutor:
    def __init__(
        self, 
        env:BaseEnv, 
        task, 
        robot_position,
        current_path_length,
        eval_logger,
        max_step,
        step_interval,
        stuck_checker:StuckChecker,
        robot_bottom_z,
        isaac_robot,
        path_data,
        success_distance,
        shortest_to_goal_distance,
        shortest_path_length,
        current_step_list,
    ):
        self.env=env
        self.task=task
        self.prev_position = robot_position
        self.current_path_length = current_path_length
        self.eval_logger = eval_logger
        self.max_step = max_step
        self.step_interval = step_interval
        self.instruction = path_data['instruction']
        self.pred_traj_list = [[]]
        self.stuck_checker = stuck_checker
        self.robot_bottom_z = robot_bottom_z
        self.isaac_robot = isaac_robot
        self.path_data = path_data
        self.success_distance = success_distance
        self.shortest_to_goal_distance = shortest_to_goal_distance
        self.shortest_path_length = shortest_path_length
        self.current_step_list = current_step_list

    def _get_action_state(self, obs, action_name):
        for env_idx, (task_name, task) in enumerate(obs.items()):
            for robot_name, robot in task.items():
                action_state = robot[action_name]
                return action_state['finished']
        return False

    def _check_max_steps(self, start_step):
        return False

    def _update_states(
        self, 
        robot_position, 
        robot_rotation,
        stack_rgb, 
        stack_depth, 
        prev_globalgps, 
        prev_globalyaw, 
        total_rgb_list, 
        total_topdown_rgb_list, 
    ):
        outputs_dict = get_obs(self.env,self.instruction,robot_position,robot_rotation)
        for idx in range(len(outputs_dict)):
            if stack_rgb is not None:
                stack_rgb[idx].push(outputs_dict[idx]["rgb"])
                stack_depth[idx].push(outputs_dict[idx]["depth"])
                prev_globalgps[idx].push(outputs_dict[idx]["globalgps"])
                prev_globalyaw[idx].push(outputs_dict[idx]["global_rotation"][-1])
            self.pred_traj_list[idx].append(robot_position)
        
        return stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list   

    def _check_fall_and_stuck(self,robot_position,robot_rotation,step):
        is_stuck = self.stuck_checker.check_robot_stuck(robot_position, robot_rotation, cur_iter=step, max_iter=2500, threshold=0.2)
        is_fall = check_robot_fall(robot_position, robot_rotation, self.robot_bottom_z)

        if is_stuck or is_fall:
            reason = 'fall' if is_fall else 'stuck'
            self.eval_logger.warning(f"Current action has been interrupted by {reason}.")
            return [True], reason
        return [False], ''

    def _execute_action(
        self,
        action, 
        action_name, 
        stack_rgb, 
        stack_depth, 
        prev_globalgps, 
        prev_globalyaw, 
        total_rgb_list, 
        total_topdown_rgb_list, 
        verbose, 
        check_fall_and_stuck,
        ):

        finish_state = False
        step = 0
        dones = [False]
        reason = ''
        
        while not finish_state:
            obs = self.env.step(actions=action, add_rgb_subframes=False, render=False)
            robot_position, robot_rotation = the_task.get_robot_poses_without_offset()
            self.current_path_length += np.linalg.norm(robot_position - self.prev_position)
            self.prev_position = robot_position

            finish_state = self._get_action_state(obs, action_name)
            step += 1
            self.current_step_list[0] += 1

            if self._check_max_steps(step):
                done = True
                reason = 'exceed_max_step'
                break

            if step % self.step_interval == 0:
                pack_ret = self._update_states(
                    robot_position,
                    robot_rotation,
                    stack_rgb, 
                    stack_depth, 
                    prev_globalgps, 
                    prev_globalyaw, 
                    total_rgb_list, 
                    total_topdown_rgb_list
                )
                stack_rgb = pack_ret[0]
                stack_depth = pack_ret[1]
                prev_globalgps = pack_ret[2]
                prev_globalyaw = pack_ret[3] 
                total_rgb_list = pack_ret[4] 
                total_topdown_rgb_list = pack_ret[5]
            
            if check_fall_and_stuck and step % 20 == 0:
                fall_or_stuck, reason = self._check_fall_and_stuck(verbose)
                if fall_or_stuck[self.env_idx]:
                    dones[0] = True
                    self.eval_logger.warning(f"Current action has been interrupted by {reason}.")
                    break

        return dones, reason, stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list

    def compute_metrics(self, fail_reason=''):
        """计算VLN任务的评估指标
        
        Args:
            infos: 当前episode的信息,包含当前位置、目标位置等
            
        Returns:
            metrics: 包含各项指标的字典
        """
        metrics = {}
        
        # 获取当前位置和目标位置
        robot_position, robot_rotation = the_isaac_robot.get_world_pose()
        current_position = robot_position
        goal_position = self.path_data['reference_path'][-1]

        # 计算Navigation Error (NE) - 当前位置到目标的欧氏距离
        ne = np.linalg.norm(current_position[:2] - goal_position[:2])
        metrics['NE'] = ne 
        
        # 计算Success Rate (SR) - 是否到达目标点
          # 成功阈值通常设为3米
        success = ne < self.success_distance
        metrics['success'] = float(success)
        
        # 计算Oracle Success Rate (OSR) - 轨迹中是否有点达到目标
        min_distance = ne if ne < self.shortest_to_goal_distance else self.shortest_to_goal_distance # 如果需要轨迹中最小距离,需要在step中记录
        metrics['osr'] = float(min_distance < self.success_distance)
        
        # 计算Trajectory Length (TL) - 轨迹总长度
        metrics['TL'] = self.current_path_length

        # 计算SPL (Success weighted by Path Length)
        if metrics['TL'] > 0:
            spl = metrics['success'] * self.shortest_path_length / max(
                metrics['TL'], self.shortest_path_length
            )
        else:
            spl = 0
        metrics['spl'] = spl
        
        # 计算NDTW (Normalized Dynamic Time Warping)
        # 计算当前轨迹与参考轨迹之间的DTW距离
        # DTW参数
        dtw_threshold = self.success_distance  # 通常设置为success_distance
        # 计算路径间的累积DTW距离
        dtw_distance = 0.0
        trajectory = []
        if len(self.pred_traj_list[0]) > 0:
            trajectory = np.array(self.pred_traj_list[0])[:,:2] # 只取x,y坐标
            reference_path = np.array(self.path_data['reference_path'])[:,:2]

            for point in trajectory:
                # 找到参考路径上最近的点
                min_dist = float('inf')
                for ref_point in reference_path:
                    dist = np.linalg.norm(point - ref_point)
                    min_dist = min(min_dist, dist)
                # 累加DTW距离,使用高斯函数进行归一化
                dtw_distance += np.exp(-min_dist**2 / (2 * dtw_threshold**2))
            
        # 归一化DTW得分
        ndtw = dtw_distance / len(trajectory) if len(trajectory) > 0 else 0.0
        metrics['ndtw'] = ndtw
        
        # 计算SDTW (Success weighted Dynamic Time Warping)
        # metrics['sdtw'] = metrics['success'] * metrics['ndtw']
        
        # 其他可能的指标
        metrics['steps'] = self.current_step_list[0]  # 步数
        metrics['episode_id'] = self.path_data['episode_id']  # episode ID
        metrics['trajectory_id'] = self.path_data['trajectory_id']  # 轨迹 ID

        metrics['fail_reason'] = fail_reason
        
        return [metrics] # batch size = 1

    def env_step(
        self,
        actions, 
        stack_rgb=None, 
        stack_depth=None, 
        prev_globalgps=None, 
        prev_globalyaw=None, 
        total_rgb_list=None, 
        total_topdown_rgb_list=None, 
        rot_action=None, 
    ):
        '''step in isaac-sim until the action has finished'''
        dones = [False]
        reason = ''
        action_name = list(actions[0]['h1'].keys())[0]
        current_position = robot_position
        if action_name == 'stop':
            dones = [True]
        else:
            if len(actions) > 0:
                packed_ret = self._execute_action(
                    action = actions, 
                    action_name=action_name, 
                    stack_rgb=stack_rgb, 
                    stack_depth=stack_depth, 
                    prev_globalgps=prev_globalgps, 
                    prev_globalyaw=prev_globalyaw, 
                    total_rgb_list=total_rgb_list, 
                    total_topdown_rgb_list=total_topdown_rgb_list, 
                    verbose=False, 
                    check_fall_and_stuck=True,
                )
                dones = packed_ret[0]
                reason = packed_ret[1]
                stack_rgb = packed_ret[2]
                stack_depth = packed_ret[3] 
                prev_globalgps = packed_ret[4] 
                prev_globalyaw = packed_ret[5] 
                total_rgb_list = packed_ret[6] 
                total_topdown_rgb_list = packed_ret[7]
            
            if rot_action is not None:
                packed_ret = self._execute_action(
                    action = rot_action, 
                    action_name=action_name, 
                    stack_rgb=stack_rgb, 
                    stack_depth=stack_depth, 
                    prev_globalgps=prev_globalgps, 
                    prev_globalyaw=prev_globalyaw, 
                    total_rgb_list=total_rgb_list, 
                    total_topdown_rgb_list=total_topdown_rgb_list, 
                    verbose=False, 
                    check_fall_and_stuck=True,
                )
                dones = packed_ret[0]
                reason = packed_ret[1]
                stack_rgb = packed_ret[2]
                stack_depth = packed_ret[3] 
                prev_globalgps = packed_ret[4] 
                prev_globalyaw = packed_ret[5] 
                total_rgb_list = packed_ret[6] 
                total_topdown_rgb_list = packed_ret[7]
        robot_position, robot_rotation = the_isaac_robot.get_world_pose()
        outputs_dict = get_obs(self.env,self.instruction,robot_position,robot_rotation)
        if action_name == 'move_to_point':
            self.pred_traj_list[0].extend(actions[0]['h1']['move_to_point'])
        infos = self.compute_metrics(fail_reason=reason)


        packed_ret = self._update_states(
            robot_position=robot_position, 
            robot_rotation=robot_rotation,
            stack_rgb=stack_rgb, 
            stack_depth=stack_depth, 
            prev_globalgps=prev_globalgps, 
            prev_globalyaw=prev_globalyaw, 
            total_rgb_list=total_rgb_list, 
            total_topdown_rgb_list=total_topdown_rgb_list, 
        )
        stack_rgb=packed_ret[0]
        stack_depth=packed_ret[1] 
        prev_globalgps=packed_ret[2] 
        prev_globalyaw=packed_ret[3] 
        total_rgb_list=packed_ret[4] 
        total_topdown_rgb_list=packed_ret[5]
        
        return {
            "outputs_dict": outputs_dict,
            "dones": dones,
            "infos": infos,
            "current_step_list": self.current_step_list,
            "stack_rgb": stack_rgb,
            "stack_depth": stack_depth,
            "prev_globalgps": prev_globalgps,
            "prev_globalyaw": prev_globalyaw,
            "total_rgb_list": total_rgb_list,
            "total_topdown_rgb_list": total_topdown_rgb_list
        }

headless = False
project_path = '/ssd/zhaohui/workspace/w61_grutopia_1220'
config_dict={
    "local_rank":0,
    "DDP":{
        "use":False,
    },
    "TORCH_GPU_IDS": [0],
    "fp16":False,
    "seed":0,
    "MODEL":{
        "policy_name":"CMA_Policy",
        "ablate_instruction":False,
        "ablate_depth":False,
        "ablate_rgb":False,
        "normalize_rgb":False,
        "INSTRUCTION_ENCODER":{
            "sensor_uuid": "instruction",
            "vocab_size": 2504,
            "use_pretrained_embeddings": True,
            "embedding_file": f"{project_path}/data/datasets/R2R_VLNCE_v1-3_preprocessed/embeddings.json.gz",
            "dataset_vocab": f"{project_path}/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz",
            "fine_tune_embeddings": False,
            "embedding_size": 50,
            "hidden_size": 128,
            "rnn_type": "LSTM",
            "final_state_only": True,
            "bidirectional": True,
        },
        "RGB_ENCODER":{
            "cnn_type": "TorchVisionResNet50",
            "output_size": 256,
            "trainable": False,
        },
        "DEPTH_ENCODER":{
            "cnn_type": "VlnResnetDepthEncoder",
            "output_size": 128,
            "backbone": "resnet50",
            "ddppo_checkpoint": f"{project_path}/data/ddppo-models/gibson-4plus-mp3d-train-val-test-resnet50.pth",
            "trainable": False,
        },
        "STATE_ENCODER":{
            "hidden_size": 512,
            "rnn_type": "GRU"
        },
        "PROGRESS_MONITOR":{
            "use": False,
            "alpha": 1.0,
        }
    },
    "IL":{
        "ckpt_to_load": f"{project_path}/data/checkpoints/habitat_sotas/CMA_PM_DA_Aug_converted.pth",
        "lr_schedule":{
            "use":True,
            "type": "cosine",
            "min_lr": 1e-5,
        },
        "epochs": 60,
        "lr": 1e-4,
        "camera_name": 'pano_camera_0'
    },
    "EVAL":{
        "ACTION": 'descrete',
        "step_interval":50,
        "success_distance": 3.0,
    }
}
config = Config(config_dict)
local_rank=0
log_dir="/ssd/zhaohui/workspace/w61_grutopia_1220/test/zhaohui/"
base_data_dir = '/ssd/share/VLN/VLNCE/R2R_VLNCE_v1-3_corrected'
mp3d_data_dir = "/ssd/share/Matterport3D/data/v1/scans"
split_data_types = ['val_unseen','val_seen']
project_path = '/ssd/zhaohui/workspace/w61_grutopia_1220'
name = '20241216_sample_episodes'
lmdb_path = project_path + f'/data/sample_episodes/{name}'
the_scan = "zsNo4HB9uLZ"
checkpoint_index=0

eval_logger_filename = os.path.join(log_dir, f"eval.log")
eval_logger = MyLogger(
    name="eval", level=logging.INFO, format_str="%(asctime)-15s %(message)s",
    filename=eval_logger_filename
)
device = torch.device("cuda", local_rank)
policy, _, _, _ = initialize_policy(
    config,
    eval_logger,
    load_from_ckpt=True,
    device=device,
    load_from_pretrain=False,
    action_stats=None,
)
policy.eval()

args_dict = {
    "datasets":{
        "mp3d_data_dir":mp3d_data_dir,
        "base_data_dir":base_data_dir,
    }
}
data_map = {}
for split_data_type in split_data_types:
    load_data_map, _ = load_gather_data(Config(args_dict), split_data_type, filter_same_trajectory=False, filter_stairs=True)
    for scan,path_list in load_data_map.items():
        path_key_list = []
        for path in path_list:
            trajectory_id = path['trajectory_id']
            episode_id = path['episode_id']
            path_key = f"{trajectory_id}_{episode_id}"
            data_map[path_key] = path
database = lmdb.open(f"{lmdb_path}/sample_data.lmdb", readonly=True, lock=False)
rank = 0
key = f"eval_rank_{rank}".encode()
with database.begin() as txn:
    value = txn.get(key)
    value = msgpack_numpy.unpackb(value)
    if value is None:
        print(f"value is None")
        sys.exit()
for scan,path_key_list in value.items():
    if scan != the_scan:
        continue
    else:
        path_key = path_key_list[0]
data = data_map[path_key]
print(data)

# 加载环境和机器人
sim_cfg_file = f'{project_path}/vln/configs/sample_episodes_sim_cfg.yaml'
sim_config = SimulatorConfig(sim_cfg_file)
scene_asset_path = load_scene_usd(Config(args_dict), the_scan)
sim_config.config.tasks[0].scene_asset_path = scene_asset_path
path_zero=data
start_position = np.array(path_zero["start_position"])
start_rotation = np.array(path_zero["start_rotation"])
sim_config.config.tasks[0].robots[0].position = start_position
sim_config.config.tasks[0].robots[0].orientation = start_rotation
env = BaseEnv(sim_config, headless=headless, webrtc=False)
the_task = env._runner.current_tasks[list(env._runner.current_tasks.keys())[0]]
the_robot = the_task.robots[list(the_task.robots.keys())[0]]
the_isaac_robot = the_robot.isaac_robot

for _ in range(10):
    env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=False, render=False)
env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=True, render=True)

from vln.src.local_nav.camera_occupancy_map import CamOccupancyMap
from vln.src.local_nav.global_topdown_map import GlobalTopdownMap
args_dict = {
    "maps":{
        "dilation_iterations":4,
        "add_dilation":True,
        "global_topdown_config":{
            "width":500,
            "height":500,
            "aperture":200,
            "camera_transform_height":0.8,
            "voxel_size":0.1
        },
        "agent_radius":0.25,
    },
    "planners":{
        "stair_sample_step":2,
        "a_star_max_iter":5000,
    },
    "windows_head":False,
    "save_path_planning":False,
    "settings":{
        "use_llm":True,
        "max_step":25000,
        "sample_camera_list":['pano_camera_0']
    },
    "log_image_dir":"",
}
topdown_map = GlobalTopdownMap(Config(args_dict), scan, vis_verbose=False)
occupancy_map = CamOccupancyMap(Config(args_dict), the_robot.sensors['topdown_camera_500'])
path_planner = AStarPlanner(
    args=Config({}),
    map_width=500,
    map_height=500,
    max_step=50000,
    windows_head=False,
    for_llm=False,
    verbose=False
)
stuck_checker = StuckChecker(the_task._offset,the_isaac_robot)
robot_bottom_z = the_robot.get_ankle_height() - sim_config.config_dict['tasks'][0]['robots'][0]['ankle_height']
robot_position, robot_rotation = the_isaac_robot.get_world_pose()
camera_pose = occupancy_map.topdown_camera.get_world_pose()[0] - the_task._offset
exe_path, shortest_path_length = get_shortest_path(
    camera_pose = camera_pose,
    robot_position = robot_position,
    reference_path = path_zero['reference_path'],
    path_planner = path_planner
)
eval_logger.info(f"The shortest path length is {shortest_path_length:.2f}")
if shortest_path_length == 0:
    eval_logger.error(f"The shortest path planning for {the_scan} :{path_key} has failed. Please check the data.")
    sys.exit()

robot_position, robot_rotation = the_task.get_robot_poses_without_offset()
observations = get_obs(env, path_zero,robot_position,robot_rotation)
bert_tokenizer=None  #?
is_clip_long = False
observations = extract_instruction_tokens(
    observations, 
    bert_tokenizer=bert_tokenizer,
    is_clip_long=is_clip_long
)
batch = batch_obs(observations, device)
batch_size = batch['instruction'].shape[0]
net = policy
env_nums = 1
rnn_states = torch.zeros(
    env_nums,
    policy.num_recurrent_layers,
    config.MODEL.STATE_ENCODER.hidden_size,
    device=device,
)
prev_actions = torch.zeros(
    env_nums,
    1, device=device, dtype=torch.long
)
not_done_masks = torch.zeros(
    env_nums,
    1, dtype=torch.uint8, device=device
)
stats_episodes = {}
rgb_frames = [[] for _ in range(env_nums,)]
num_eps = 1
pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
pbar_iter = 0
log_str = (
    f"[Ckpt: {checkpoint_index}]"
    " [Episodes evaluated: {evaluated}/{total}]"
    " [Time elapsed (s): {time}]"
)
start_time = time.time()

steps = [0] * batch_size
sim_steps = [0] * batch_size
steps_batch = torch.from_numpy(np.array(steps)).to(device)
batch["steps"] = steps_batch

spl_dict = {}
total_actions = []
current_episode_start_time = time.time()
current_episodes = path_zero

while True:
    if len(spl_dict) > 0 and np.mean(list(spl_dict.values())) < 0.02 and len(stats_episodes) > 100:
        # this ckpt is too bad to continue
        eval_logger.info(f"Break. This ckpt is too bad to continue with average SPL {np.mean(list(spl_dict.values())):.3f}")
        sys.exit()
    with torch.no_grad():
        actions, rnn_states = policy(
            batch,
            rnn_states,
            prev_actions,
            not_done_masks,
            deterministic=not config.EVAL.SAMPLE,
        )
        prev_actions.copy_(actions)
        if config.EVAL.ACTION == 'descrete':
            for bs_i, a in enumerate(actions):
                if a == 0:
                    action = [
                        {'h1': {'stop': ['stop']}}
                    ]
                else:
                    action = [
                        {'h1': {'move_by_descrete': [a.item()]}}
                    ]
        current_path_length=0
        current_step_list=[0]
        total_rgb_list = []
        total_topdown_rgb_list = []
        executor = ActionExecutor(
            env=env, 
            task=the_task, 
            robot_position=robot_position,
            current_path_length=current_path_length,
            eval_logger=eval_logger,
            max_step=50000,
            step_interval=config.EVAL.step_interval,
            instruction=path_zero['instruction'],
            stuck_checker=stuck_checker,
            robot_bottom_z=robot_bottom_z,
            isaac_robot=the_isaac_robot,
            path_data=path_zero,
            success_distance=config.EVAL.success_distance,
            shortest_to_goal_distance=999,
            shortest_path_length=shortest_path_length,
            current_step_list=current_step_list,
        )
        outputs = executor.env_step(
            action,
            total_rgb_list=total_rgb_list, 
            total_topdown_rgb_list=total_topdown_rgb_list,
        )
        outputs_dict = outputs['outputs_dict']
        dones = outputs['dones']
        infos = outputs['infos']
        sim_steps = outputs['current_step_list']
        total_rgb_list = outputs['total_rgb_list']
        total_topdown_rgb_list = outputs['total_topdown_rgb_list']
        steps[0] += 1

        observations = outputs_dict

        not_done_masks = torch.tensor(
            [[0] if done else [1] for done in dones],
            dtype=torch.uint8,
            device=device,
        )
        if dones[0]:
            break