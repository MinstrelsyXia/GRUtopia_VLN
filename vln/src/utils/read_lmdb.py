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

        env.close()
    
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
    
    def save_episode_video(self, episode_data, key, output_dir):
        """Save the episode video to a file."""
        frames = []
        
        # Collect frames from episode data
        rgb_data = episode_data['episode_data']['camera_info']['pano_camera_0']['rgb']
        for frame in rgb_data:
            # Convert the frame to a PIL image and then to a NumPy array
            pil_image = Image.fromarray(frame)
            frames.append(np.array(pil_image))

        # Define output video file path
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
    
    def analysis_lmdb(self, dataset_root_dir, split, output_json_file='logs/scan_completion.json'):
        # load data
        lmdb_data = self.read_all_episode_data()
        dataset_data, scans = self.load_vln_dataset(dataset_root_dir, split)
        total_results = {"success": 0, "total": 0, "failure": 0, "path planning": 0, "fall": 0, "stuck": 0, "maximum step": 0}
        
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
                    else:
                        scan_completion[scan]['failure'] += 1
                        scan_completion[scan][lmdb_data[traj_id]['fail_reason']] += 1
                        total_results[lmdb_data[traj_id]['fail_reason']] += 1

        # Write results to a JSON file
        with open(output_json_file, 'w') as json_file:
            json.dump(scan_completion, json_file, indent=4)
        
        with open(output_json_file, 'a') as json_file:
            json.dump(total_results, json_file, indent=4)

        print(f"Results written to {output_json_file}")
    
        return scan_completion
    
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

if __name__ == '__main__':
    mode = 'analysis'
    
    lmdb_path = 'data/sample_episodes/20241105_sample_episodes/sample_data.lmdb'
    data_collector = LmdbReader(lmdb_path)
    if mode == 'save_video':
        '''1. Load all data'''
        # data_collector.read_all_episode_data()
        
        '''2. Load the target path_id'''
        path_id = '3703'
        episode_data = data_collector.read_episode_data(path_id)
        ## save to the video
        data_collector.save_episode_video(episode_data, key=path_id, output_dir='logs/videos')
    
    elif mode == 'analysis':
        '''3. Analysis the LMDB data'''
        data_collector.analysis_lmdb(dataset_root_dir='../VLN/VLNCE/R2R_VLNCE_v1-3', split='train')