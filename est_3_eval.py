import os
import json

def calculate_metrics(target_dir):
    """计算所有 episode 的平均指标"""
    total_success = 0
    total_tl = 0
    total_spl = 0
    episode_count = 0
    
    # 遍历所有子目录
    for scene_dir in os.listdir(target_dir):
        scene_path = os.path.join(target_dir, scene_dir)
        if os.path.isdir(scene_path):
            for episode_dir in os.listdir(scene_path):
                metric_file = os.path.join(scene_path, episode_dir, 'metric.json')
                if os.path.exists(metric_file):
                    with open(metric_file, 'r') as f:
                        metric = json.load(f)
                        total_success += metric.get('success', 0)
                        total_tl += metric.get('TL', 0)
                        total_spl += metric.get('spl', 0)
                        episode_count += 1
    
    # 计算平均值
    success_rate = total_success / episode_count if episode_count > 0 else 0
    avg_tl = total_tl / episode_count if episode_count > 0 else 0
    avg_spl = total_spl / episode_count if episode_count > 0 else 0
    
    print(f"总 episode 数: {episode_count}")
    print(f"成功率: {success_rate:.4f}")
    print(f"平均轨迹长度: {avg_tl:.4f}")
    print(f"平均 SPL: {avg_spl:.4f}")
    
    return success_rate, avg_tl, avg_spl

# 调用函数计算指标
target_dir = '/ssd/xiaxinyuan/code/w61-grutopia/logs_docker_1213'
calculate_metrics(target_dir)

room_target_dir = 'logs_docker_room'
calculate_metrics(room_target_dir)
