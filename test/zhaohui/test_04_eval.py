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
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from vln.src.utils import progress_log_util

aperture=500

def vis_nav_path(start_pixel, goal_pixel, points, occupancy_map, img_save_path='path_planning.jpg'):
    cmap = mcolors.ListedColormap(['white', 'green', 'gray', 'black'])
    bounds = [0, 1, 3, 254, 256]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=(10, 10))
    # plt.imshow(occupancy_map, cmap='binary', origin='lower')
    plt.imshow(occupancy_map, cmap=cmap, norm=norm, origin='upper')

    # Plot start and goal points
    plt.plot(start_pixel[1], start_pixel[0], 'ro', markersize=6, label='Start')
    plt.plot(goal_pixel[1], goal_pixel[0], 'bo', markersize=6, label='Goal')

    # Plot the path
    if len(points) > 0:
        path = np.array(points)
        plt.plot(path[:, 1], path[:, 0], 'xb-', linewidth=1, markersize=5, label='Path')

    # Customize the plot
    plt.title('Path planning')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.colorbar(label='Occupancy (0: Free, 1: Occupied)')

    # Save the plot
    plt.savefig(img_save_path, pad_inches=0, bbox_inches='tight', dpi=100)
    print(f"Saved path planning visualization to {img_save_path}")
    plt.close()
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
    path_planner:AStarPlanner,
):
    freemap, _ = occupancy_map.get_global_free_map(robot_pos=robot_position, robot_height=1.55, update_camera_pose=False, verbose=False)
    topdown_map.update_map(freemap, camera_pose, verbose=False, env_idx=0, update_map=True)
    occupancy_map_, _ = topdown_map.get_map(robot_position, return_camera_pose=True)
    start = time.time()
    start_pixel = world_to_pixel(reference_path[0],camera_pose,aperture,500,500)
    goal_pixel = world_to_pixel(reference_path[-1],camera_pose,aperture,500,500)
    paths, find_flag, fail_reason = path_planner.planning(
        start_pixel[0], 
        start_pixel[1],
        goal_pixel[0], 
        goal_pixel[1],
        obs_map=occupancy_map_,
    )
    end = time.time()
    print(f"old planning 耗时{round(end - start,2)}秒")
    # file_name = f"new_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg" 
    # vis_nav_path(
    #     start_pixel, 
    #     goal_pixel, 
    #     paths, 
    #     occupancy_map_, 
    #     img_save_path=os.path.join(f'{project_path}/test/zhaohui/', file_name)
    # )
    exe_path = []
    if find_flag:
        for node in paths:
            world_coords = pixel_to_world([node[0],node[1]], camera_pose,aperture,500,500)
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
    ext={
        "map_info":occupancy_map_,
        "camera_pose":camera_pose,
        "aperture":aperture,
        "width":500,
        "height":500,
        "path_planning_fail_reason":fail_reason,
    }
    return exe_path, shortest_path_length, ext
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
def describe_action(action):
    if action == 1:
        return "向前走0.25米"
    elif action == 2:
        return "左转15°"
    elif action == 3:
        return "右转15°"
    else:
        return "结束"
def generate_result_key(ckpt_name, path_key):
    return f"eval_{ckpt_name}_{path_key}"

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

class Statistic_Info:
    def __init__(
            self,
            env,
            path_data,
            shortest_path_length,
            shortest_to_goal_distance,
            step_interval,
            success_distance,
        ):
        self.env = env
        # policy 执行次数
        self.policy_step = 0
        # sim step 执行次数
        self.sim_step = 0
        # 目前已经移动的长度
        self.current_path_length = 0
        self.step_interval = step_interval
        # 摄像头采样
        self.stack_rgb = [[]]
        self.stack_depth = [[]]
        # 机器人位置采样
        self.prev_globalgps = [[]]
        self.prev_globalyaw = [[]]
        # 地图采样
        self.total_rgb_list = [[]]
        self.total_topdown_rgb_list = [[]]
        self.pred_traj_list = [[]]
        # 统计结果需要的信息
        self.path_data = path_data
        self.shortest_path_length = shortest_path_length
        self.shortest_to_goal_distance = shortest_to_goal_distance
        self.success_distance = success_distance
    
    def _update_states(
        self,
        step,
        robot_position, 
        robot_rotation,
        instruction,
    ):
        if step % self.step_interval != 0:
            return
        # outputs_dict = get_obs(self.env,instruction,robot_position,robot_rotation)
        # for idx in range(len(outputs_dict)):
            # self.stack_rgb[idx].push(outputs_dict[idx]["rgb"])
            # self.stack_depth[idx].push(outputs_dict[idx]["depth"])
            # self.prev_globalgps[idx].push(outputs_dict[idx]["globalgps"])
            # self.prev_globalyaw[idx].push(outputs_dict[idx]["global_rotation"][-1])
        self.pred_traj_list[0].append(robot_position)

    def compute_metrics(
        self, 
        robot_position, 
        fail_reason=''
    ):
        """计算VLN任务的评估指标
        
        Args:
            infos: 当前episode的信息,包含当前位置、目标位置等
            
        Returns:
            metrics: 包含各项指标的字典
        """
        metrics = {}
        current_position = robot_position
        goal_position = self.path_data['reference_path'][-1]

        # 计算Navigation Error (NE) - 当前位置到目标的欧氏距离
        ne = np.linalg.norm(current_position[:2] - goal_position[:2])
        metrics['reference_path'] = self.path_data['reference_path']
        metrics['shortest_path_length'] = self.shortest_path_length
        metrics['pred_traj_list'] = self.pred_traj_list[0]
        metrics['NE'] = ne 
        
        # 计算Success Rate (SR) - 是否到达目标点
          # 成功阈值通常设为3米
        success = ne < self.success_distance
        metrics['success'] = float(success)
        
        # 计算Oracle Success Rate (OSR) - 轨迹中是否有点达到目标
        if ne < self.shortest_to_goal_distance:
            self.shortest_to_goal_distance = ne
        metrics['osr'] = float(self.shortest_to_goal_distance < self.success_distance)
        
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
        metrics['steps'] = self.sim_step  # 步数
        metrics['episode_id'] = self.path_data['episode_id']  # episode ID
        metrics['trajectory_id'] = self.path_data['trajectory_id']  # 轨迹 ID
        metrics['fail_reason'] = fail_reason
        
        return [metrics] # batch size = 1

class ActionExecutor:
    def __init__(
        self, 
        env:BaseEnv, 
        task, 
        eval_logger,
        stuck_checker:StuckChecker,
        isaac_robot,

        per_action_max_step,
        total_max_step,
        robot_bottom_z,

        statistic_info:Statistic_Info,
    ):
        # 执行 step 用到的工具类
        self.env=env
        self.task=task
        self.eval_logger = eval_logger
        self.stuck_checker = stuck_checker
        self.isaac_robot = isaac_robot

        # 执行 step 需要的配置信息
        self.per_action_max_step=per_action_max_step
        self.total_max_step = total_max_step
        self.robot_bottom_z = robot_bottom_z
        # 统计信息
        self.statistic_info = statistic_info
        # 可以优化掉的变量
        self.instruction = self.statistic_info.path_data['instruction']

    def _get_action_state(self, obs, action_name):
        for env_idx, (task_name, task) in enumerate(obs.items()):
            for robot_name, robot in task.items():
                action_state = robot[action_name]
                return action_state['finished']
        return False

    def _check_max_steps(self, step):
        if step > self.per_action_max_step:
            return True, 'exceed_per_action_max_step'
        if self.statistic_info.sim_step > self.total_max_step:
            return True, 'exceed_total_max_step'
        return False, ''

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
        check_fall_and_stuck,
    ):
        finish_state = False
        step = 0
        dones = [False]
        reason = ''
        prev_position, _ = the_task.get_robot_poses_without_offset()
        while not finish_state:
            obs = self.env.step(actions=action, add_rgb_subframes=False, render=False)
            robot_position, robot_rotation = the_task.get_robot_poses_without_offset()
            
            self.statistic_info.current_path_length += np.linalg.norm(robot_position - prev_position)
            prev_position = robot_position

            finish_state = self._get_action_state(obs, action_name)
            step += 1
            self.statistic_info.sim_step += 1

            over_max_step, desc = self._check_max_steps(step)
            if over_max_step:
                dones = [True]
                reason = desc
                break

            if check_fall_and_stuck and step % 20 == 0:
                fall_or_stuck, desc = self._check_fall_and_stuck(robot_position, robot_rotation, self.statistic_info.sim_step)
                if fall_or_stuck[0]:
                    dones = [True]
                    reason = desc
                    self.eval_logger.warning(f"Current action has been interrupted by {reason}.")
                    break
            
            if not finish_state:
                self.statistic_info._update_states(
                    step=step,
                    robot_position=robot_position,
                    robot_rotation=robot_rotation,
                    instruction=self.instruction,
                )            
        return dones, reason

    def env_step(
        self,
        actions, 
    ):
        '''step in isaac-sim until the action has finished'''
        dones = [False]
        reason = ''
        action_name = list(actions[0]['h1'].keys())[0]
        if action_name == 'stop' or len(actions) == 0:
            dones = [True]
        else:
            dones,reason = self._execute_action(
                action = actions, 
                action_name=action_name,  
                check_fall_and_stuck=True,
            )
            if action_name == 'move_by_descrete':
                action_list = actions[0]['h1']['move_by_descrete']
                for one_action in action_list:
                    self.eval_logger.info(f"[descrete][step:{self.statistic_info.sim_step}] 完成动作:{describe_action(one_action)}")

        robot_position, robot_rotation = the_isaac_robot.get_world_pose()
        outputs_dict = get_obs(self.env,self.instruction,robot_position,robot_rotation)
        if action_name == 'move_to_point':
            self.statistic_info.pred_traj_list[0].extend(actions[0]['h1']['move_to_point'])
        
        self.statistic_info._update_states(
            step=self.statistic_info.step_interval,
            robot_position=robot_position, 
            robot_rotation=robot_rotation,
            instruction=self.instruction,
        )
        infos = self.statistic_info.compute_metrics(robot_position=robot_position,fail_reason=reason)
        
        return {
            "outputs_dict": outputs_dict,
            "dones": dones,
            "infos": infos,
            "reason": reason,
        }

headless = True
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
        "SAMPLE":False,
    },
    "use_pbar":False,
}
config = Config(config_dict)
local_rank=0
log_dir="/ssd/zhaohui/workspace/w61_grutopia_1220/test/zhaohui/"
base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
mp3d_data_dir = "/ssd/share/Matterport3D/data/v1/scans"
split_data_types = ['val_unseen','val_seen']
project_path = '/ssd/zhaohui/workspace/w61_grutopia_1220'
name = '20241216_sample_episodes'
lmdb_path = project_path + f'/data/sample_episodes/{name}'
the_scan = "zsNo4HB9uLZ"
checkpoint_index=0
per_action_max_step=1500
max_step=25000
is_clip_long = False
bert_tokenizer=None  #?
ckpt_name="test"

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
split_map={}
robot_offset = np.array([0.   , 0.   , 0.975])
for split_data_type in split_data_types:
    load_data_map, _ = load_gather_data(Config(args_dict), split_data_type, filter_same_trajectory=False, filter_stairs=True)
    for scan,path_list in load_data_map.items():
        path_key_list = []
        for path in path_list:
            trajectory_id = path['trajectory_id']
            episode_id = path['episode_id']
            path_key = f"{trajectory_id}_{episode_id}"
            path["start_position"] += robot_offset
            for i, _ in enumerate(path["reference_path"]):
                path["reference_path"][i] += robot_offset
            data_map[path_key] = path
            split_map[path_key] = split_data_type
database = lmdb.open(f"{lmdb_path}/sample_data.lmdb", readonly=True, lock=False)
rank = 0
key = f"eval_rank_{rank}".encode()
with database.begin() as txn:
    value = txn.get(key)
    value = msgpack_numpy.unpackb(value)
    if value is None:
        print(f"value is None")
        sys.exit()
target_path_key_list=[]
retry_list=['']
for scan,path_key_list in value.items():
    if scan != the_scan:
        continue
    else:
        target_path_key_list = path_key_list
tmp = []
retry_list=['']
for path_key in target_path_key_list:
    eval_key = generate_result_key(ckpt_name=ckpt_name,path_key=path_key)
    with database.begin() as txn:
        value = txn.get(eval_key.encode())
        if value is None:
            tmp.append(path_key)
        else:
            value = msgpack_numpy.unpackb(value)
            if value['success'] == 1.0:
                if 'success' in retry_list:
                    tmp.append(path_key)
                else:
                    continue
            else:
                fail_reason = value['fail_reason']
                if fail_reason in retry_list:
                    tmp.append(path_key)

target_path_key_list=tmp

if len(target_path_key_list) == 0:
    print(f"[scan:{{the_scan}}] has no data to eval")
    sys.exit(0)



# 加载环境和机器人
sim_cfg_file = f'{project_path}/vln/configs/sim_cfg_policy_eval.yaml'
sim_config = SimulatorConfig(sim_cfg_file)
scene_asset_path = load_scene_usd(Config(args_dict), the_scan)
sim_config.config.tasks[0].scene_asset_path = scene_asset_path
path_zero=data_map[target_path_key_list[0]]
start_position = np.array(path_zero["start_position"])
start_rotation = np.array(path_zero["start_rotation"])
sim_config.config.tasks[0].robots[0].position = start_position
sim_config.config.tasks[0].robots[0].orientation = start_rotation
env = BaseEnv(sim_config, headless=headless, webrtc=False)
the_task = env._runner.current_tasks[list(env._runner.current_tasks.keys())[0]]
the_robot = the_task.robots[list(the_task.robots.keys())[0]]
the_isaac_robot = the_robot.isaac_robot


from vln.src.local_nav.camera_occupancy_map import CamOccupancyMap
from vln.src.local_nav.global_topdown_map import GlobalTopdownMap
args_dict = {
    "maps":{
        "dilation_iterations":2,
        "add_dilation":True,
        "global_topdown_config":{
            "width":500,
            "height":500,
            "aperture":aperture,
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
        "use_llm":False,
        "max_step":25000,
        "sample_camera_list":['pano_camera_0']
    },
    "log_image_dir":"",
}
topdown_map = GlobalTopdownMap(Config(args_dict), the_scan, vis_verbose=False)
occupancy_map = CamOccupancyMap(Config(args_dict), the_robot.sensors['topdown_camera_500'])
path_planner = AStarPlanner(
    args=Config({}),
    map_width=500,
    map_height=500,
    max_step=per_action_max_step,
    windows_head=False,
    for_llm=False,
    verbose=False
)
stuck_checker = StuckChecker(the_task._offset,the_isaac_robot)
robot_bottom_z = the_robot.get_ankle_height() - sim_config.config_dict['tasks'][0]['robots'][0]['ankle_height']
progress_log_util.init(the_scan, len(target_path_key_list))
progress_log_util.progress_logger.info(f"start sampling scan: {the_scan}, total_path:{len(target_path_key_list)}")


for i in range(len(target_path_key_list)):
    path_key = target_path_key_list[i]
    data = data_map[path_key]
    episode_id=path_key.split("_")[1]
    trajectory_id=path_key.split("_")[0]
    print(f"split: {split_map[path_key]}")
    print(f"scan: {the_scan}")
    print(f"trajectory_id_episode_id: {path_key}")
    print(f"data: {data}")
    progress_log_util.trace_start(
        trajectory_id = path_key,
        step_count=0,
    )

    the_task.set_single_robot_poses_without_offset(start_position, start_rotation)
    the_isaac_robot.set_world_velocity(np.zeros(6))
    the_isaac_robot.set_joint_velocities(np.zeros(len(the_isaac_robot.dof_names)))
    the_isaac_robot.set_joint_positions(np.zeros(len(the_isaac_robot.dof_names)))
    the_isaac_robot.set_joint_efforts(np.zeros(len(the_isaac_robot.dof_names)))

    for _ in range(240):
        env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=False, render=False)
    env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=True, render=True)

    robot_position, robot_rotation = the_isaac_robot.get_world_pose()
    camera_pose = occupancy_map.topdown_camera.get_world_pose()[0] - the_task._offset
    exe_path, shortest_path_length, ext_info = get_shortest_path(
        camera_pose = camera_pose,
        robot_position = robot_position,
        reference_path = path_zero['reference_path'],
        path_planner = path_planner
    )
    eval_logger.info(f"The shortest path length is {shortest_path_length:.2f}")
    if shortest_path_length == 0:
        eval_logger.error(f"The shortest path planning for {the_scan} :{path_key} has failed. Please check the data.")
        result = ext_info['path_planning_fail_reason']
        progress_log_util.trace_end(
            trajectory_id = path_key,
            step_count=statistic_info.sim_step,
            result = result,
        )
        ext_info['exe_path'] = exe_path
        info={
            "reference_path":data['reference_path'],
            "shortest_path_length":shortest_path_length,
            "pred_traj_list":[],
            "NE":-1,
            "success":0.0,
            "osr":-1,
            "TL":0,
            "spl":0.0,
            "ndtw":-1,
            "steps":0,
            "episode_id":int(episode_id),
            "trajectory_id":int(trajectory_id),
            "fail_reason":result,
            'ext_info':ext_info,
        }
        database_write = lmdb.open(f"{lmdb_path}/sample_data.lmdb", map_size=1 * 1024 * 1024 * 1024 * 1024, max_dbs=0)
        with database_write.begin(write=True) as txn:
            key_write = generate_result_key(ckpt_name=ckpt_name,path_key=path_key).encode()
            value_write = msgpack_numpy.packb(info, use_bin_type=True)
            txn.put(key_write, value_write)
        continue

    robot_position, robot_rotation = the_task.get_robot_poses_without_offset()
    observations = get_obs(env, path_zero['instruction'],robot_position,robot_rotation)
    observations = extract_instruction_tokens(
        observations, 
        bert_tokenizer=bert_tokenizer,
        is_clip_long=is_clip_long
    )
    observations = batch_obs(observations, device)
    observations["steps"] = torch.from_numpy(np.array([0])).to(device)

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


    statistic_info = Statistic_Info(
        env=env,
        path_data=path_zero,
        shortest_path_length=shortest_path_length,
        shortest_to_goal_distance=999,
        step_interval=config.EVAL.step_interval,
        success_distance=config.EVAL.success_distance,
    )
    spl_dict = {}
    stats_episodes = {}

    while True:
        if len(spl_dict) > 0 and np.mean(list(spl_dict.values())) < 0.02 and len(stats_episodes) > 100:
            # this ckpt is too bad to continue
            eval_logger.info(f"Break. This ckpt is too bad to continue with average SPL {np.mean(list(spl_dict.values())):.3f}")
            sys.exit()
        if statistic_info.sim_step % 1000 == 0:
            eval_logger.info(f"[split:{split_map[path_key]}][scan:{the_scan}][trajectory_id_episode_id: {path_key}][step:{statistic_info.sim_step}]")

        batch = {
            'mode': 'inference',
            'observations': observations,
            'rnn_states': rnn_states,
            'prev_actions': prev_actions,
            'masks': not_done_masks
        }
        with torch.no_grad():
            actions, rnn_states = policy(batch)
        prev_actions.copy_(actions)
        if config.EVAL.ACTION == 'descrete':
            for bs_i, a in enumerate(actions):
                if a == 0:
                    eval_logger.info(f"[split:{split_map[path_key]}][scan:{the_scan}][trajectory_id_episode_id: {path_key}][stop!!!]")
                    action = [
                        {'h1': {'stop': ['stop']}}
                    ]
                else:
                    action = [
                        {'h1': {'move_by_descrete': [a.item()]}}
                    ]
        executor = ActionExecutor(
            env=env, 
            task=the_task, 
            eval_logger=eval_logger,
            stuck_checker=stuck_checker,
            isaac_robot=the_isaac_robot,

            per_action_max_step=per_action_max_step,
            total_max_step=max_step,
            robot_bottom_z=robot_bottom_z,

            statistic_info=statistic_info,
        )
        outputs = executor.env_step(actions = action)
        outputs_dict = outputs['outputs_dict']
        dones = outputs['dones']
        info = outputs['infos'][0]
        reason = outputs['reason']
        statistic_info = executor.statistic_info
        statistic_info.policy_step +=1

        outputs_dict = extract_instruction_tokens(
            outputs_dict, 
            bert_tokenizer=bert_tokenizer,
            is_clip_long=is_clip_long
        )
        observations = batch_obs(outputs_dict, device)
        observations["steps"] = torch.from_numpy(np.array([0])).to(device)
        not_done_masks = torch.tensor(
            [[0] if done else [1] for done in dones],
            dtype=torch.uint8,
            device=device,
        )
        if dones[0]:
            result = reason
            if result == '':
                if info['success'] > 0:
                    result='success'
                else:
                    info['fail_reason']='not_reach_goal'
                    result='not_reach_goal'
            progress_log_util.trace_end(
                trajectory_id = path_key,
                step_count=statistic_info.sim_step,
                result = result,
            )
            ext_info['exe_path']=exe_path
            info['ext_info']=ext_info
  
            database_write = lmdb.open(f"{lmdb_path}/sample_data.lmdb", map_size=1 * 1024 * 1024 * 1024 * 1024, max_dbs=0)
            with database_write.begin(write=True) as txn:
                key_write = generate_result_key(ckpt_name=ckpt_name,path_key=path_key).encode()
                value_write = msgpack_numpy.packb(info, use_bin_type=True)
                txn.put(key_write, value_write)
            stats_episodes[path_key] = info
            spl_dict[path_key] = float(stats_episodes[path_key]["spl"])
            mean_spl = np.mean(list(spl_dict.values()))
            eval_logger.info(f"Average SPL: {mean_spl}")
            break