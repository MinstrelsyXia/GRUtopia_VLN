import os
import json
import gzip

# 定义数据
def create_episode(episode_id,start_position, start_rotation, position, reference_path):
    '''
    episode id: num of path.
    usage: ep = create_episode([0, 0, 0], [0, 0, 0, 1], [0, 0, 0], [[0, 0, 0], [0, 0, 0]]); episodes = [ep]; save_episodes(episodes)
    '''
    episode = {
        'episode_id': episode_id,  # 你可以根据需要生成唯一的 episode_id
        'trajectory_id': 0,  # 你可以根据需要生成唯一的 trajectory_id
        'scene_id': 'sixth_floor',  # 你可以根据需要修改 scene_id
        'start_position': start_position,
        'start_rotation': start_rotation,
        'info': {'geodesic_distance': 7.9608235359191895},  # 你可以根据需要修改 geodesic_distance
        'goals': [{'position': position, 'radius': 3.0}],  # 你可以根据需要修改 radius
        'instruction': {
            'instruction_text': 'fake instruction',
            'instruction_tokens': [0,0]
        },
        'reference_path': reference_path
    }
    return episode


def save_episodes(episodes,dataset_root_dir):
    data = {
        'episodes': episodes
    }
    split = 'sixth_floor'
    output_dir = os.path.join(dataset_root_dir, split)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{split}.json.gz")

    # 将数据写入 JSON 文件并压缩为 .gz 格式
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        json.dump(data, f)

    print(f"Data has been written to {output_file}")


