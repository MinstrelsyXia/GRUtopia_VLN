import os, sys
import lmdb
import pickle
from PIL import Image
import numpy as np
import cv2
import zlib
import json
import msgpack_numpy
from collections import defaultdict


class LmdbReader:
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path

    def read_episode_data(self, path_id):
        """Read episode data from the LMDB database."""
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False)  # Open LMDB in readonly mode
        with env.begin() as txn:
            key = f"{path_id}".encode()  # Create the key used to store the data

            # Retrieve the data using the key
            value = txn.get(key)

            if value is not None:
                # Deserialize data using pickle
                # value = zlib.decompress(value)
                # data = pickle.loads(value)
                data = msgpack_numpy.unpackb(value, raw=False)
                return data
            else:
                print(f"No data found for path_id: {path_id}")
                return None
    
    def read_all_episode_data(self):
        """Read all episode data from the LMDB database."""
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False)  # Open LMDB in readonly mode
        all_data = {}  # Dictionary to store all episode data

        with env.begin() as txn:
            with txn.cursor() as cursor:
                for key, value in cursor:
                    key_decoded = key.decode('utf-8')  # Decode the key from bytes to string
                    # Deserialize data using pickle
                    # value = zlib.decompress(value)
                    # data = pickle.loads(value)
                    data = msgpack_numpy.unpackb(value, raw=False)
                    all_data[key_decoded] = data  # Store in the dictionary

        env.close()
        return all_data  # Return all episode data as a dictionary

    def read_all_keys(self):
        """从LMDB数据库中读取所有键值。

        Returns:
            list: 包含所有键值的列表
        """
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False)  # 以只读模式打开LMDB
        keys = []

        with env.begin() as txn:
            with txn.cursor() as cursor:
                for key, _ in cursor:
                    keys.append(key.decode('utf-8'))  # 将字节类型的键值解码为字符串

        env.close()
        return keys
    
    def save_episode_video(self, episode_data, key, output_dir, use_pid=False):
        """Save the episode video to a file."""
        frames = []
        
        # Collect frames from episode data
        rgb_data = episode_data['episode_data']['camera_info']['pano_camera_0']['rgb']
        for frame in rgb_data:
            # Convert the frame to a PIL image and then to a NumPy array
            pil_image = Image.fromarray(frame)
            # save the image

            frames.append(np.array(pil_image))

        # Define output video file path
        if use_pid:
            output_file = os.path.join(output_dir, f"episode_video_{key}_pid.mp4")
        else:
            output_file = os.path.join(output_dir, f"episode_video_{key}.mp4")
        
        # Check the dimensions of the first frame
        if len(frames) > 0:
            height, width, layers = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_file, fourcc, 6, (width, height))

            # Write frames to video
            for frame in frames:
                video_writer.write(frame)

            # Release the video writer
            video_writer.release()
            print(f"Video saved successfully to {output_file}")
        else:
            print("No frames to save to video.")
    
    def analysis_lmdb(self, dataset_root_dir, split, output_json_dir='logs'):
        # load data
        lmdb_data = self. _all_episode_data()
        dataset_data, scans = self.load_vln_dataset(dataset_root_dir, split)
        total_results = {"success": 0, "total": 0, "failure": 0, "path planning": 0, "fall": 0, "stuck": 0, "maximum step": 0}

        success_episode_data = []
        success_lmdb_data = []
        
        # analysis
        scan_completion = defaultdict(lambda: {"success": 0, "total": 0, "failure": 0, "path planning": 0, "fall": 0, "stuck": 0, "maximum step": 0})
        # Iterate through each scan in dataset_data
        for scan, ep_infos in dataset_data.items():
            # Count total episode_ids for the scan
            scan_completion[scan]['total'] = len(ep_infos)
            total_results["total"] += len(ep_infos)
            
            # Check each episode_id in lmdb_data for completion
            for ep_info in ep_infos:
                traj_id = str(ep_info['trajectory_id'])
                if traj_id in lmdb_data:
                    # Here, we assume lmdb_data[episode_id] has a 'completed' status
                    if lmdb_data[traj_id]['finish_status'] == 'success':  # Replace 'completed' with actual status key
                        scan_completion[scan]['success'] += 1
                        total_results["success"] += 1
                        success_episode_data.append(ep_info)
                        success_lmdb_data.append([traj_id, lmdb_data[traj_id]])
                    else:
                        scan_completion[scan]['failure'] += 1
                        scan_completion[scan][lmdb_data[traj_id]['fail_reason']] += 1
                        total_results[lmdb_data[traj_id]['fail_reason']] += 1

        # Write results to a JSON file
        output_json_file = os.path.join(output_json_dir, f'scan_completion_{split}.json')
        with open(output_json_file, 'w') as json_file:
            json.dump(scan_completion, json_file, indent=4)
        
        with open(output_json_file, 'a') as json_file:
            json.dump(total_results, json_file, indent=4)
        
        print(f"Results written to {output_json_file}")
        
        output_success_episode_file = os.path.join(output_json_dir, f'success_episode_data_{split}.json')
        with open(output_success_episode_file, 'w') as json_file:
            json.dump(success_episode_data, json_file, indent=4)

        print(f"Success episode data written to {output_success_episode_file}")
    
        return scan_completion, success_episode_data, success_lmdb_data
    
    def check_exist_scan_and_pathId(self, dataset_root_dir, split, only_recollect_path_planning_fail=False):
        """Check if the scan and path_id exist in the LMDB database."""
        # load data
        lmdb_data = self.read_all_episode_data()
        dataset_data, scans = self.load_vln_dataset(dataset_root_dir, split)
        scan_pathId_list = defaultdict(list)
        for scan, ep_infos in dataset_data.items():
            # Check each episode_id in lmdb_data for completion
            for ep_info in ep_infos:
                traj_id = str(ep_info['trajectory_id'])
                if traj_id in lmdb_data:
                    if lmdb_data[traj_id]['finish_status'] == 'success': 
                        continue
                    else:
                        if only_recollect_path_planning_fail:
                            if lmdb_data[traj_id]['fail_reason'] == 'path planning':
                                scan_pathId_list[scan].append(traj_id)
                        else:
                            scan_pathId_list[scan].append(traj_id)
        return scan_pathId_list
        
    def load_vln_dataset(self, dataset_root_dir, split, filter_same_trajectory=True, filter_stairs=True):
        with open(os.path.join(dataset_root_dir, "gather_data", f"{split}_gather_data.json"), 'r') as f:
            data = json.load(f)
        with open(os.path.join(dataset_root_dir, "gather_data", "env_scan.json"), 'r') as f:
            scan = json.load(f)

        new_data = defaultdict(list)
        if filter_same_trajectory or filter_stairs:
            if filter_same_trajectory:
                trajectory_list = []
            for scan, data_item in data.items():
                for item in data_item:
                    if filter_same_trajectory:
                        if item['trajectory_id'] in trajectory_list:
                            continue
                        else:
                            trajectory_list.append(item['trajectory_id'])

                    if filter_stairs:
                        if 'stair' in item['instruction']['instruction_text']:
                            # use the differences between the z-dim among reference paths to filter stairs
                            height_th = 0.3
                            latest_height = item['reference_path'][0][-1]
                            has_stairs = False
                            for path_id in range(1, len(item['reference_path'])):
                                path = item['reference_path'][path_id]
                                if abs(path[-1] - latest_height) >= height_th:
                                    # stairs
                                    has_stairs = True
                                    break
                                else:
                                    latest_height = path[-1]
                            if has_stairs:
                                continue

                        different_height = False
                        paths = item['reference_path']
                        for path_idx in range(len(paths)-1):
                            if abs(paths[path_idx+1][2] - paths[path_idx][2]) > 0.3:
                                different_height = True
                                break
                        if different_height:
                            continue

                    new_data[scan].append(item)
            data = new_data

        return data, scan
    def save_episode_images(self, episode_data, key, output_dir):
        output_dir = os.path.join(output_dir, key)
        os.makedirs(output_dir, exist_ok=True)
        # episode_data['episode_data']['camera_info']['pano_camera_0']['rgb']: [num, 256,256,3]
        for idx, frame in enumerate(episode_data['episode_data']['camera_info']['pano_camera_0']['rgb']):
            pil_image = Image.fromarray(frame)
            pil_image.save(os.path.join(output_dir, f"frame_{idx:04d}.jpg"))

    def get_target_path_id(self):
        path_id_file = "vlmaps/docker/valid_paths/sub_success_paths_room_id_1212.txt"
        with open(path_id_file, 'r') as f:
            path_id_list = f.readlines()
        return path_id_list
if __name__ == '__main__':
    mode = 'get_images'
    use_pid = True
    
    root_dir = '/ssd/zhaohui/workspace/w61_grutopia_1216'
    pid_lmdb_path = os.path.join(root_dir, 'data/sample_episodes/20241216_sample_episodes')
    original_lmdb_path = os.path.join(root_dir, 'data/sample_episodes/20241216_sample_episodes')

    val_seen_lmdb_path = os.path.join(root_dir, 'data/sample_episodes/20241115_sample_episodes_val_seen/sample_data.lmdb')
    val_unseen_lmdb_path = os.path.join(root_dir, 'data/sample_episodes/20241115_sample_episodes_val_unseen/sample_data.lmdb')
    
    if mode == 'get_images':
        my_lmdb_path = os.path.join(root_dir, 'data/sample_episodes/20241216_sample_episodes/sample_data.lmdb')
        data_collector = LmdbReader(my_lmdb_path)
        all_keys = data_collector.read_all_keys()
        for key in all_keys:
            episode_data = data_collector.read_episode_data(key)
            if episode_data is not None:
                data_collector.save_episode_images(episode_data, key=key, output_dir='logs/images')

    if mode == 'save_video':
        if use_pid:
            lmdb_path = pid_lmdb_path
            print(f"Use pid lmdb: {pid_lmdb_path}")
        else:
            lmdb_path = original_lmdb_path
            print(f"Use original lmdb: {original_lmdb_path}")

        data_collector = LmdbReader(lmdb_path)
        '''1. Load all data'''
        # all_data = data_collector.read_all_episode_data()
        # path_id_list = all_data.keys()
        # path_id_list = [1220, 476, 531, 67, 883, 94]
        # path_id_list = [str(x) for x in path_id_list]
        # for path_id in path_id_list:
        #     episode_data = data_collector.read_episode_data(path_id)
        #     ## save to the video
        #     if episode_data is not None:
        #         data_collector.save_episode_video(episode_data, key=path_id, output_dir='logs/videos', use_pid=use_pid)

        '''2. Load the target path_id'''
        all_keys = data_collector.read_all_keys()
        path_id = '1015'
        episode_data = data_collector.read_episode_data(path_id)
        ## save to the video
        if episode_data is not None:
            data_collector.save_episode_video(episode_data, key=path_id, output_dir='logs/videos', use_pid=use_pid)
    
    elif mode == 'analysis':
        '''3. Analysis the LMDB data'''
        split = 'val_unseen'
        dataset_root_dir = '../VLN/VLNCE/R2R_VLNCE_v1-3_corrected'
        if split == 'val_seen':
            lmdb_path = val_seen_lmdb_path
        elif split == 'val_unseen':
            lmdb_path = val_unseen_lmdb_path
        save_success_video = True

        output_json_dir = os.path.join('/'.join(lmdb_path.split('/')[:-1]), 'analysis')
        os.makedirs(output_json_dir, exist_ok=True)
        data_collector = LmdbReader(lmdb_path)
        scan_completion, success_episode_data, success_lmdb_data = data_collector.analysis_lmdb(dataset_root_dir=dataset_root_dir, split=split, output_json_dir=output_json_dir)

        if save_success_video:
            output_video_dir = os.path.join('/'.join(lmdb_path.split('/')[:-1]), 'success_videos')
            os.makedirs(output_video_dir, exist_ok=True)
            for ep_info in success_lmdb_data:
                data_collector.save_episode_video(episode_data=ep_info[1], key=ep_info[0], output_dir=output_video_dir, use_pid=use_pid)
