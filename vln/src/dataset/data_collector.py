import os
import json
import time
import numpy as np
from PIL import Image
import multiprocessing as mp

class dataCollector:
    def __init__(self, args, parent_pipe, child_pipe, split, scan, path_id_list):
        # TODO: this has not supported multiple robots
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

        for path_id in self.path_id_list:
            save_dir = os.path.join(args.sample_episode_dir, split, scan, f"id_{str(path_id)}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
            pose_save_path = os.path.join(save_dir, 'poses.txt')
            cam_save_path = os.path.join(save_dir, 'camera_param.jsonl')
            robot_save_path = os.path.join(save_dir, 'robot_param.jsonl')

            self.save_dir_list.append(save_dir)
            self.pose_save_path_list.append(pose_save_path)
            self.cam_save_path_list.append(cam_save_path)
            self.robot_save_path_list.append(robot_save_path)


    def collect_and_send_data(self, step_time, env, camera_list, camera_pose_dict, robot_pose_dict, end_list, add_rgb_subframes=True, finish_flag=False):
        obs = env.get_observations(add_rgb_subframes=add_rgb_subframes)

        # episode_data = {
        #     'camera_data': {},
        #     'robot_info': {},
        #     'finish_flag': finish_flag
        # }
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
                            'env_idx': env_idx
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
                            'step_time': step_time,
                            'rgb': rgb_info,
                            'depth': depth_info,
                            'position': pos.tolist(),
                            'orientation': quat.tolist()
                        }

                    episode_data[robot_name]['robot_info'] = {
                        "step_time": step_time,
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

            # Save camera information
            all_finish_flag = True
            finish_flag = True
            for episode_data_item in episode_datas:
                if episode_data_item is not None:
                    for robot_name, episode_data in episode_data_item.items():
                        idx = episode_data['env_idx']
                        if 'camera_data' in episode_data:
                            for camera, camera_data in episode_data['camera_data'].items():
                                # Save camera pose
                                pos = camera_data['position']
                                quat = camera_data['orientation']
                                with open(self.pose_save_path_list[idx], 'a') as f:
                                    f.write(f"{pos[0]}\t{pos[1]}\t{pos[2]}\t{quat[0]}\t{quat[1]}\t{quat[2]}\t{quat[3]}\n")

                                # Save images
                                step_time = camera_data['step_time']
                                rgb_image = Image.fromarray(camera_data['rgb'], "RGB")
                                depth_image = camera_data['depth']

                                rgb_filename = os.path.join(self.save_dir_list[idx], f"{camera}_image_step_{step_time}.png")
                                rgb_image.save(rgb_filename)

                                depth_filename = os.path.join(self.save_dir_list[idx], f"{camera}_depth_step_{step_time}.npy")
                                np.save(depth_filename, depth_image)

                            # Save robot information
                            with open(self.robot_save_path_list[idx], 'a') as f:
                                json.dump(episode_data['robot_info'], f)
                                f.write('\n')
                        
                    finish_flag = episode_data['finish_flag']
                all_finish_flag = finish_flag and all_finish_flag
            if all_finish_flag:
                break