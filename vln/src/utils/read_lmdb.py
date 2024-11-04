import os, sys
import lmdb
import pickle
from PIL import Image
import numpy as np
import cv2

class DataCollector:
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
                data = pickle.loads(value)
                return data
            else:
                print(f"No data found for path_id: {path_id}")
                return None

        env.close()
    
    def read_all_episode_data(self):
        """Read all episode data from the LMDB database."""
        env = lmdb.open(self.lmdb_path, readonly=True)  # Open LMDB in readonly mode
        all_data = {}  # Dictionary to store all episode data

        with env.begin() as txn:
            with txn.cursor() as cursor:
                for key, value in cursor:
                    key_decoded = key.decode('utf-8')  # Decode the key from bytes to string
                    # Deserialize data using pickle
                    data = pickle.loads(value)
                    all_data[key_decoded] = data  # Store in the dictionary

        env.close()
        return all_data  # Return all episode data as a dictionary
    
    def save_episode_video(self, episode_data, key, output_dir):
        """Save the episode video to a file."""
        frames = []
        
        # Collect frames from episode data
        for episode in episode_data['episode_data']:
            frame = episode['camera_info']['pano_camera_0']['rgb']
            # Convert the frame to a PIL image and then to a NumPy array
            pil_image = Image.fromarray(frame)
            frames.append(np.array(pil_image))

        # Define output video file path
        output_file = os.path.join(output_dir, f"episode_video_{key}.mp4")
        
        # Check the dimensions of the first frame
        if len(frames) > 0:
            height, width, layers = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_file, fourcc, 24.0, (width, height))

            # Write frames to video
            for frame in frames:
                video_writer.write(frame)

            # Release the video writer
            video_writer.release()
            print(f"Video saved successfully to {output_file}")
        else:
            print("No frames to save to video.")

if __name__ == '__main__':
    lmdb_path = '/ssd/wangliuyi/code/GRUtopia/data/sample_episodes/20241101_sample_episodes/sample_data.lmdb'
    data_collector = DataCollector(lmdb_path)
    '''1. Load all data'''
    # data_collector.read_all_episode_data()
    
    '''2. Load the target path_id'''
    path_id = '5'
    episode_data = data_collector.read_episode_data(path_id)
    ## save to the video
    data_collector.save_episode_video(episode_data, key=path_id, output_dir='logs/videos')