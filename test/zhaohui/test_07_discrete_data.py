
import lmdb
import msgpack_numpy
from PIL import Image
import numpy as np
import os
import cv2

def describe_action(action):
    if action == 1:
        return "向前走0.25米"
    elif action == 2:
        return "左转15°"
    elif action == 3:
        return "右转15°"
    else:
        return "停在原地"

def print_actions(actions):
    for index,action in enumerate(actions):
        print(f"[{index}]==>{describe_action(action)}")

project_path = '/ssd/zhaohui/workspace/w61_grutopia_0107'
lmdb_path_0507 = project_path + '/data/sample_episodes/20250110_dagger/sample_data.lmdb'
env_0508 = lmdb.open(lmdb_path_0507, readonly=True, lock=False)
id=6
key = f"{id}".encode()
with env_0508.begin() as txn:
    value = txn.get(key)
    value = msgpack_numpy.unpackb(value)
    if value is None:
        print(f"value is None")
    else:
        if 'action' in value['episode_data']:
            actions = value['episode_data']['action']
            print_actions(actions)
        frames = []
        rgb_data = value['episode_data']['camera_info']['pano_camera_0']['rgb']
        for frame in rgb_data:
            pil_image = Image.fromarray(frame)
            pil_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            frames.append(pil_image)
        output_file = os.path.join(project_path, f"new.mp4")
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