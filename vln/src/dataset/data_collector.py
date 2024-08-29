import os
import json
import time
import numpy as np
from PIL import Image
import multiprocessing as mp

class dataCollector:
    def __init__(self, args, parent_pipe, child_pipe, split, scan, path_id):
        # TODO: this has not supported multiple robots
        self.args = args
        self.data_collection_interval = 1
        self.parent_pipe = parent_pipe
        self.child_pipe = child_pipe

        self.split = split
        self.scan = scan
        self.path_id = path_id
        self.save_dir = os.path.join(self.args.sample_episode_dir, split, scan, f"id_{str(path_id)}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.pose_save_path = os.path.join(self.save_dir, 'poses.txt')
        self.cam_save_path = os.path.join(self.save_dir, 'camera_param.jsonl')
        self.robot_save_path = os.path.join(self.save_dir, 'robot_param.jsonl')

    def collect_and_send_data(self, step_time, env, camera_list, camera_pose_dict, add_rgb_subframes=True, finish_flag=False):
        obs = env.get_observations(add_rgb_subframes=add_rgb_subframes)

        # episode_data = {
        #     'camera_data': {},
        #     'robot_info': {},
        #     'finish_flag': finish_flag
        # }
        episode_datas = []

        for task_name, task in obs.items():
            for robot_name, robot in task.items():
                episode_data = {
                    robot_name: {
                        'camera_data': {},
                        'robot_info': {},
                        'finish_flag': finish_flag
                    }
                }
                for camera in camera_list:
                    cur_obs = obs[task_name][robot_name][camera]
                    camera_pose = camera_pose_dict[camera]
                    pos, quat = camera_pose[0], camera_pose[1]

                    rgb_info = cur_obs['rgba'][..., :3]
                    depth_info = cur_obs['depth']
                    max_depth = 10
                    depth_info[depth_info > max_depth] = 0

                    episode_data[robot_name]['camera_data'][camera] = {
                        'step_time': step_time,
                        'rgb': rgb_info,
                        'depth': depth_info,
                        'position': pos.tolist(),
                        'orientation': quat.tolist()
                    }

                episode_data[robot_name]['robot_info'] = {
                    "step_time": step_time,
                    "position": obs[task_name][robot_name]['position'].tolist(),
                    "orientation": obs[task_name][robot_name]['orientation'].tolist()
                }

            episode_datas.append(episode_data)

        self.parent_pipe.send(episode_datas)

    def save_episode_data(self):
        while True:
            time.sleep(self.data_collection_interval)
            episode_datas = self.child_pipe.recv()
            if len(episode_datas) == 0:
                continue

            # Save camera information
            for episode_data_item in episode_datas:
                for robot_name, episode_data in episode_data_item.items():
                    if 'camera_data' in episode_data:
                        for camera, camera_data in episode_data['camera_data'].items():
                            # Save camera pose
                            pos = camera_data['position']
                            quat = camera_data['orientation']
                            with open(self.pose_save_path, 'a') as f:
                                f.write(f"{pos[0]}\t{pos[1]}\t{pos[2]}\t{quat[0]}\t{quat[1]}\t{quat[2]}\t{quat[3]}\n")

                            # Save images
                            step_time = camera_data['step_time']
                            rgb_image = Image.fromarray(camera_data['rgb'], "RGB")
                            depth_image = camera_data['depth']

                            rgb_filename = os.path.join(self.save_dir, f"{camera}_image_step_{step_time}.png")
                            rgb_image.save(rgb_filename)

                            depth_filename = os.path.join(self.save_dir, f"{camera}_depth_step_{step_time}.npy")
                            np.save(depth_filename, depth_image)

                        # Save robot information
                        with open(self.robot_save_path, 'a') as f:
                            json.dump(episode_data['robot_info'], f)
                            f.write('\n')
                        
                finish_flag = episode_data['finish_flag']
            if finish_flag:
                break