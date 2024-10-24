import os
import json
import time
import numpy as np
from PIL import Image
import multiprocessing as mp

def collect_and_send_data(pipe, env, args, task_name, robot_name, camera_list, add_rgb_subframes=True):
    def get_camera_pose():
        # Implement this function based on your environment
        pass

    while True:
        obs = env.get_observations(add_rgb_subframes=add_rgb_subframes)
        camera_pose_dict = get_camera_pose()

        episode_data = {
            'camera_data': {},
            'robot_info': {}
        }

        for camera in camera_list:
            cur_obs = obs[task_name][robot_name][camera]
            camera_pose = camera_pose_dict[camera]
            pos, quat = camera_pose[0], camera_pose[1]

            rgb_info = cur_obs['rgba'][..., :3]
            depth_info = cur_obs['depth']
            max_depth = 10
            depth_info[depth_info > max_depth] = 0

            episode_data['camera_data'][camera] = {
                'step_time': time.time(),
                'rgb': rgb_info,
                'depth': depth_info,
                'position': pos.tolist(),
                'orientation': quat.tolist()
            }

        episode_data['robot_info'] = {
            "step_time": time.time(),
            "position": obs[task_name][robot_name]['position'].tolist(),
            "orientation": obs[task_name][robot_name]['orientation'].tolist()
        }

        pipe.send(episode_data)
        time.sleep(args.data_collection_interval)  # Adjust this interval as needed

def save_episode_data(pipe, args, split, scan, path_id):
    save_dir = os.path.join(args.sample_episode_dir, split, scan, f"id_{str(path_id)}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pose_save_path = os.path.join(save_dir, 'poses.txt')
    cam_save_path = os.path.join(save_dir, 'camera_param.jsonl')
    robot_save_path = os.path.join(save_dir, 'robot_param.jsonl')

    while True:
        episode_data = pipe.recv()

        # Save camera information
        for camera, camera_data in episode_data['camera_data'].items():
            # Save pose
            pos = camera_data['position']
            quat = camera_data['orientation']
            with open(pose_save_path, 'a') as f:
                f.write(f"{pos[0]}\t{pos[1]}\t{pos[2]}\t{quat[0]}\t{quat[1]}\t{quat[2]}\t{quat[3]}\n")

            # Save images
            step_time = camera_data['step_time']
            rgb_image = Image.fromarray(camera_data['rgb'], "RGB")
            depth_image = camera_data['depth']

            rgb_filename = os.path.join(save_dir, f"{camera}_image_step_{step_time}.png")
            rgb_image.save(rgb_filename)

            depth_filename = os.path.join(save_dir, f"{camera}_depth_step_{step_time}.npy")
            np.save(depth_filename, depth_image)

            # Save camera parameters
            camera_info = {
                "camera": camera,
                "step_time": step_time,
                "position": camera_data['position'],
                'orientation': camera_data['orientation']
            }
            with open(cam_save_path, 'a') as f:
                json.dump(camera_info, f)
                f.write('\n')

        # Save robot information
        with open(robot_save_path, 'a') as f:
            json.dump(episode_data['robot_info'], f)
            f.write('\n')

def main():
    # Initialize your environment, args, and other necessary variables here
    env = None  # Replace with your actual environment initialization
    args = None  # Replace with your actual args
    task_name = "your_task_name"
    robot_name = "your_robot_name"
    camera_list = ["camera1", "camera2"]  # Add your camera names
    split = "train"  # or "test", etc.
    scan = "scan_name"
    path_id = "path_id"

    # Create a pipe for communication between processes
    parent_conn, child_conn = mp.Pipe()

    # Create and start the data collection process
    collection_process = mp.Process(target=collect_and_send_data, args=(parent_conn, env, args, task_name, robot_name, camera_list))
    collection_process.start()

    # Create and start the data saving process
    saving_process = mp.Process(target=save_episode_data, args=(child_conn, args, split, scan, path_id))
    saving_process.start()

    # Wait for the processes to finish (you may want to implement a proper termination condition)
    collection_process.join()
    saving_process.join()

if __name__ == "__main__":
    main()