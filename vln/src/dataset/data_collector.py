import os
import json
import time
import numpy as np
from PIL import Image
import multiprocessing as mp
import lmdb
import torch
import msgpack_numpy
from multiprocessing import Lock
import pickle
import math
import importlib
import zlib

# This uses multi-thread to save raw data in sub-dictionaries
class dataCollector:
    def __init__(self, args, parent_pipe, child_pipe, split, scan, path_id_list):
        self.args = args
        self.data_collection_interval = 1
        self.parent_pipe = parent_pipe
        self.child_pipe = child_pipe

        self.split = split
        self.scan = scan
        self.path_id_list = path_id_list
        self.save_dir_list = []
        self.pose_save_path_list = []
        self.cam_save_path_list = []
        self.robot_save_path_list = []

        self.generate_save_dir_list(path_id_list)
    
    def generate_save_dir_list(self, path_id_list):
        save_dir_list, pose_save_path_list, cam_save_path_list, robot_save_path_list = [], [], [], []
        for path_id in path_id_list:
            save_dir = os.path.join(self.args.sample_episode_dir, self.split, self.scan, f"id_{str(path_id)}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
            pose_save_path = os.path.join(save_dir, 'poses.txt')
            cam_save_path = os.path.join(save_dir, 'camera_param.jsonl')
            robot_save_path = os.path.join(save_dir, 'robot_param.jsonl')

            save_dir_list.append(save_dir)
            pose_save_path_list.append(pose_save_path)
            cam_save_path_list.append(cam_save_path)
            robot_save_path_list.append(robot_save_path)
        
        return save_dir_list, pose_save_path_list, cam_save_path_list, robot_save_path_list


    def collect_and_send_data(self, step_time, env, camera_list, 
                              camera_pose_dict, robot_pose_dict, end_list, 
                              path_id_list, start_step_list, 
                              add_rgb_subframes=True, finish_flag=False):
        # generate path id list
        save_dir_list, pose_save_path_list, cam_save_path_list, robot_save_path_list = self.generate_save_dir_list(path_id_list)

        obs = env.get_observations(add_rgb_subframes=add_rgb_subframes)

        episode_datas = []

        for env_idx, (task_name, task) in enumerate(obs.items()):
            episode_data = None
            if not end_list[env_idx]:
                for robot_name, robot in task.items():
                    episode_data = {
                        robot_name: {
                            'camera_data': {},
                            'robot_info': {},
                            'finish_flag': finish_flag,
                            'env_idx': env_idx,
                            'save_dir_list': save_dir_list,
                            'pose_save_path_list': pose_save_path_list,
                            'cam_save_path_list': cam_save_path_list,
                            'robot_save_path_list': robot_save_path_list
                        }
                    }
                    for camera in camera_list:
                        cur_obs = obs[task_name][robot_name][camera]
                        camera_pose = camera_pose_dict[task_name][camera]
                        pos, quat = camera_pose[0], camera_pose[1]

                        rgb_info = cur_obs['rgba'][..., :3]
                        depth_info = cur_obs['depth']
                        max_depth = 10
                        depth_info[depth_info > max_depth] = 0

                        episode_data[robot_name]['camera_data'][camera] = {
                            'step_time': step_time-start_step_list[env_idx],
                            'rgb': rgb_info,
                            'depth': depth_info,
                            'position': pos.tolist(),
                            'orientation': quat.tolist()
                        }

                    episode_data[robot_name]['robot_info'] = {
                        "step_time": step_time-start_step_list[env_idx],
                        "position": robot_pose_dict[env_idx][0].tolist(),
                        "orientation": robot_pose_dict[env_idx][1].tolist()
                    }

            episode_datas.append(episode_data)

        self.parent_pipe.send(episode_datas)

    def save_episode_data(self):
        while True:
            time.sleep(self.data_collection_interval)
            episode_datas = self.child_pipe.recv()

            if len(episode_datas) == 0:
                continue

            if 'finish_flag' in episode_datas:
                if episode_datas['finish_flag'] == True:
                    break

            # Save camera information
            all_finish_flag = True
            finish_flag = True
            for i, episode_data_item in enumerate(episode_datas):
                if episode_data_item is not None:
                    for robot_name, episode_data in episode_data_item.items():
                        save_dir_list = episode_data['save_dir_list']
                        pose_save_path_list = episode_data['pose_save_path_list']
                        cam_save_path_list = episode_data['cam_save_path_list']
                        robot_save_path_list = episode_data['robot_save_path_list']

                        idx = episode_data['env_idx']
                        if 'camera_data' in episode_data:
                            for camera, camera_data in episode_data['camera_data'].items():
                                # Save camera pose
                                pos = camera_data['position']
                                quat = camera_data['orientation']
                                with open(pose_save_path_list[idx], 'a') as f:
                                    f.write(f"{pos[0]}\t{pos[1]}\t{pos[2]}\t{quat[0]}\t{quat[1]}\t{quat[2]}\t{quat[3]}\n")

                                # Save images
                                step_time = camera_data['step_time']
                                rgb_image = Image.fromarray(camera_data['rgb'], "RGB")
                                depth_image = camera_data['depth']

                                rgb_filename = os.path.join(save_dir_list[idx], f"{camera}_image_step_{step_time}.png")
                                rgb_image.save(rgb_filename)

                                depth_filename = os.path.join(save_dir_list[idx], f"{camera}_depth_step_{step_time}.npy")
                                np.save(depth_filename, depth_image)

                            # Save robot information
                            with open(robot_save_path_list[idx], 'a') as f:
                                json.dump(episode_data['robot_info'], f)
                                f.write('\n')
                        
                    finish_flag = episode_data['finish_flag']
                all_finish_flag = finish_flag and all_finish_flag

                if not self.parent_pipe.poll():
                    self.child_pipe.send({'save_flag': True})
            if all_finish_flag:
                break
    
    def get_save_finish_flag(self):
        flag = self.parent_pipe.recv()
        if flag is not None and 'save_flag' in flag:
            return True
        return False

class LmdbDataCollector:
    def __init__(self, args, split, scan, path_id_list, lmdb_path, env_num):
        self.args = args
        self.data_collection_interval = 1
        self.split = split
        self.scan = scan
        self.path_id_list = path_id_list
        self.lmdb_path = lmdb_path  # LMDB database file path
        
        self.episode_total_data = [[] for _ in range(env_num)]
        
        # This can only be imported after env init
        rotations_utils = importlib.import_module("omni.isaac.core.utils.rotations")
        self.quat_to_euler_angles = rotations_utils.quat_to_euler_angles
    
    def quaternion_to_yaw(self, w, x, y, z):
        # 计算 yaw
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return yaw
    
    def norm_depth(self, depth_info, min_depth=0, max_depth=10):
        depth_info[depth_info > max_depth] = max_depth
        depth_info = (depth_info - min_depth) / (max_depth - min_depth)
        return depth_info

    def collect_data(self, step_time, env, camera_list, camera_pose_dict,
                     robot_pose_dict, end_list, path_id_list, start_step_list,
                     progress_list,  
                     add_rgb_subframes=True, success_list=0, fail_reasons=None):
        """Collect data from environment observations for each step."""
        obs = env.get_observations(add_rgb_subframes=add_rgb_subframes)

        for env_idx, (task_name, task) in enumerate(obs.items()):
            episode_data = None
            if not end_list[env_idx]:
                for robot_name, robot in task.items():
                    # Suppose there are no multiple robots in one env.
                    episode_data = {
                        'camera_info': {},
                        'robot_info': {},
                        # 'path_id': path_id_list[env_idx],
                        'step': step_time - start_step_list[env_idx],
                        'progress': progress_list[env_idx]
                    }
                    for camera in camera_list:
                        cur_obs = obs[task_name][robot_name][camera]
                        camera_pose = camera_pose_dict[task_name][camera]
                        pos, quat = camera_pose[0], camera_pose[1]
                        _,_, yaw = self.quat_to_euler_angles(quat) #TODO: Check!

                        rgb_info = cur_obs['rgba'][..., :3]
                        depth_info = self.norm_depth(cur_obs['depth'])

                        episode_data['camera_info'][camera] = {
                            'rgb': rgb_info,
                            'depth': depth_info,
                            'position': pos.tolist(),
                            'orientation': quat.tolist(),
                            'yaw': yaw
                        }

                    pos, quat = robot_pose_dict[env_idx][0], robot_pose_dict[env_idx][1]
                    _,_, yaw = self.quat_to_euler_angles(quat)
                    episode_data['robot_info'] = {
                        "position": pos.tolist(),
                        "orientation": quat.tolist(),
                        "yaw": yaw
                    }

                self.episode_total_data[env_idx].append(episode_data)

    def save_data(self, env_idx, path_id, success_flag, fail_reason, instruction):
        finish_flag = "success" if success_flag else "fail"
        self.save_episode_data(self.episode_total_data[env_idx], path_id, finish_flag, instruction, fail_reason)
        self.episode_total_data[env_idx] = []
    
    def collate_data(self, episode_datas):
        camera_info_dict = {}
        robot_info_list = {
            "position": [],
            "orientation": [],
            "yaw": [],
        }
        progress_list = []
        step_list = []

        for episode_data in episode_datas:
            for camera, info in episode_data['camera_info'].items():
                if camera not in camera_info_dict:
                    camera_info_dict[camera] = {
                        "rgb": [],
                        "depth": [],
                        "position": [],
                        "orientation": [],
                        "yaw": [],
                    }
                
                camera_info_dict[camera]["rgb"].append(info["rgb"])
                camera_info_dict[camera]["depth"].append(info["depth"])
                camera_info_dict[camera]["position"].append(info["position"])
                camera_info_dict[camera]["orientation"].append(info["orientation"])
                camera_info_dict[camera]["yaw"].append(info["yaw"])

            robot_info_list["position"].append(episode_data["robot_info"]["position"])
            robot_info_list["orientation"].append(episode_data["robot_info"]["orientation"])
            robot_info_list["yaw"].append(episode_data["robot_info"]["yaw"])
            
            step_list.append(episode_data["step"])
            progress_list.append(episode_data["progress"])

        for camera, info in camera_info_dict.items():
            for key, values in info.items():
                camera_info_dict[camera][key] = np.array(values)

        for key, values in robot_info_list.items():
            robot_info_list[key] = np.array(values)
        
        collate_data = {
            'camera_info': camera_info_dict,
            'robot_info': robot_info_list,
            'progress': np.array(progress_list),
            'step': np.array(step_list)
        }
        
        return collate_data
        
    def save_episode_data(self, episode_datas, path_id, finish_flag, instruction, fail_reason=None):
        """Save finished episode into the LMDB database.
        :finish_flag: ['success', 'fail']
        :fail_reason:
        """
        env = lmdb.open(self.lmdb_path, map_size=1 * 1024 * 1024 * 1024 * 1024, max_dbs=0)  # Adjust map_size as needed
        with env.begin(write=True) as txn:
            # Use the path_id as the key and store all episode data under it
            # path_id = episode_datas[0].get('path_id', 'unknown_path')
            key = f"{path_id}".encode()

            episode_datas = self.collate_data(episode_datas)
            # Package episode data with metadata
            data_to_store = {
                'episode_data': episode_datas,
                'finish_status': finish_flag,
                'fail_reason': fail_reason,
                'instruction': instruction
            }
            # serialized_data = pickle.dumps(data_to_store)
            # compressed_data = zlib.compress(serialized_data)
            
            serialized_data = msgpack_numpy.packb(data_to_store, use_bin_type=True)

            # Serialize data using pickle and write to LMDB
            txn.put(key, serialized_data)

        env.close()
        # print(f"Episode {path_id} saved with status {self.finish_status[finish_flag]}.")