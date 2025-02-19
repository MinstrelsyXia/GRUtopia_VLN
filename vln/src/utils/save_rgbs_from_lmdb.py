import cv2
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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Collect frames from episode data
        rgb_data = episode_data['episode_data']['camera_info']['pano_camera_0']['rgb']
        for frame in rgb_data:
            # Convert the frame to a PIL image and then to a NumPy array
            pil_image = Image.fromarray(frame)
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
    
    def save_rgb_from_lmdb(self, lmdb_path, output_dir, save_actions=True, max_length=150):
        """从LMDB数据库中提取RGB帧并保存到指定目录。
        
        Args:
            lmdb_path: LMDB数据库路径
            output_dir: 输出目录路径
        """
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 读取LMDB数据
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        
        # 使用cursor.iternext()方法只获取键值
        all_keys = []
        with env.begin() as txn:
            cursor = txn.cursor()
            while cursor.next():
                all_keys.append(cursor.key().decode('utf-8'))
            cursor.close()
        
        print(f"该LMDB数据库共包含 {len(all_keys)} 个样本")
        success_count = 0
        save_count = 0
        save_episode_min_length = 20
        error_log = []
        total_actions = {}

        # 使用获取的keys来读取数据
        with env.begin() as txn:
            for episode_id in all_keys:
                try:
                    value = txn.get(episode_id.encode())
                except Exception as e:
                    print(f"错误：无法获取键值 {episode_id}")
                    continue
                
                if value is None:
                    continue
                
                episode_dir = os.path.join(output_dir, episode_id, 'rgb')
                # 解包数据
                data = msgpack_numpy.unpackb(value, raw=False)
                if 'finish_status' not in data:
                    continue
                
                if data['fail_reason'] == 'fast_fall' or 'pano_camera_0' not in data['episode_data']['camera_info']:
                    continue
                
                rgb_frames = data['episode_data']['camera_info']['pano_camera_0']['rgb'][:max_length]
                if data['finish_status'] == 'success':
                    success_count += 1
                    save_count += 1
                else:
                    sample_length = len(data['episode_data']['camera_info']['pano_camera_0']['rgb'])
                    if sample_length > save_episode_min_length:
                        save_count += 1
                    else:
                        continue
                
                os.makedirs(episode_dir, exist_ok=True)

                # 保存每一帧
                for frame_idx, frame in enumerate(rgb_frames):
                    output_path = os.path.join(episode_dir, f"{frame_idx}.jpg")
                    # 将numpy数组转换为PIL图像并保存
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    Image.fromarray(frame_rgb).save(output_path)
                
                print(f"已保存视频 {episode_id} 的 {len(rgb_frames)} 帧到 {episode_dir}")

                if save_actions:
                    action_json_file = os.path.join(episode_dir, f"actions.json")
                    with open(action_json_file, 'w') as f:
                        total_action = data['episode_data']['action'][:max_length]
                        action = total_action[:-1]
                        action.append(0)
                        if len(action) != len(rgb_frames):
                            print(f"错误：动作长度与RGB帧长度不一致 {episode_id}. len action {len(action)}, len rgb {len(rgb_frames)}")
                            error_log.append([episode_id, len(action), len(rgb_frames)])
                            continue
                        json.dump(action, f)
                    
                    # 把所有action记录到一个文件里
                    total_actions[episode_id] = action
        
        # 把所有action记录到一个文件里
        with open(os.path.join(output_dir, f"total_actions.json"), 'w') as f:
            json.dump(total_actions, f)
        
        env.close()
        print(f"所有RGB帧已保存到 {output_dir}")
        print(f"成功保存 {save_count} 个视频, 其中采集成功的数量是{success_count}")
        print(f"错误日志")
        for error in error_log:
            print(f"episode_id: {error[0]}, len action: {error[1]}, len rgb: {error[2]}")

def extract_frames_from_video(video_dir, video_path, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开视频文件
    cap = cv2.VideoCapture(os.path.join(video_dir, video_path))
    video_idx = video_path.split('.')[0]  # e.g., 1.mp4

    video_output_dir = os.path.join(output_dir, f"{video_idx}")
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
        video_output_dir = os.path.join(video_output_dir, f"rgb")
        os.makedirs(video_output_dir)
        print(f"video output dir: {video_output_dir}")
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return
    
    frame_count = 0
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        # 如果读取失败，退出循环
        if not ret:
            break
            
        # 构建输出文件路径（例如：M/rgb/0.jpg）
        output_path = os.path.join(video_output_dir, f"{frame_count}.jpg")
        
        # 保存帧图像
        cv2.imwrite(output_path, frame)
        
        frame_count += 1
        
        # 打印进度
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count} 帧")
    
    # 释放视频对象
    cap.release()
    print(f"完成！共处理 {frame_count} 帧")

    print(f"RGB 储存目录 {video_output_dir}")

# 使用示例
if __name__ == "__main__":
    mode = 'save_rgbs_from_lmdb'

    if mode == 'save_rgbs_from_video':
        '''1. save rgbs from video'''
        video_dir = "data/videos"
        video_path = "1.mp4"  # 输入视频路径
        output_dir = "data/rgbs/"      # 输出目录
        extract_frames_from_video(video_dir, video_path, output_dir)
    elif mode == 'save_rgbs_from_lmdb':
        # lmdb_path = 'data/sample_episodes/20250211_sixth_floor_sample/sample_data.lmdb'
        lmdb_path = 'data/sample_episodes/20250214_sample_aliengo/sample_data.lmdb'
        output_dir = 'data/aliengo_rgbs_lmdb'
        data_collector = LmdbReader(lmdb_path)
        data_collector.save_rgb_from_lmdb(lmdb_path, output_dir)
