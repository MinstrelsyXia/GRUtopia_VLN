import json
import os
data_split = 'val_unseen'
json_file = f"vlmaps/docker/valid_paths/{data_split}_PReval_gather_data.json"

with open(json_file, 'r') as f:
    data = json.load(f)
    episode_all = []
for item in data:
    traj = data[item]
    for t in traj:
        print(t['episode_id'])
        episode_all.append(t['episode_id'])

print(len(episode_all))
# dump episode into unseen.txt
main_dir = f"vlmaps/docker/valid_paths/{data_split}"
if not os.path.exists(main_dir): 
    os.makedirs(main_dir)   
txt_file = os.path.join(main_dir, "episode_ids.txt")
with open(txt_file, 'w') as f:
    for episode in episode_all:
        f.write(f"{episode}\n")


import gzip
import json
import copy
import os
starrt_pos_dict = {}
episode_id = []
trajectory_id_list = []

target_dir = f'vlmaps/docker/valid_paths/{data_split}'

object_path_file = os.path.join(target_dir, "object.txt")
object_id_file = os.path.join(target_dir, "object_id.txt")
room_path_file = os.path.join(target_dir, "room.txt") 
room_id_file = os.path.join(target_dir, "room_id.txt")
with open(object_path_file, 'w') as f:
    pass
with open(object_id_file, 'w') as f:
    pass
with open(room_path_file, 'w') as f:
    pass
with open(room_id_file, 'w') as f:
    pass
def get_sub_trajectory_id_list():
    with open(txt_file, 'r') as f:
        sub_trajectory_id_list = [int(line.strip()) for line in f.readlines()]
        # 重新按大小排序
        sub_trajectory_id_list.sort()
    return sub_trajectory_id_list


def find_gpu_id(trajectory_id,num_of_gpus,trajectory_id_list):
    for idx, id in enumerate(trajectory_id_list):
        if (id == trajectory_id):
            return idx % num_of_gpus
def judge_stair_exists(path):
    '''
    path_z: ndarray, if max(path_z)- min(path_z) > 1.5, then there is a stair
    '''
    path_z = []
    for path in path:
        path_z.append(path[1])
    return max(path_z)- min(path_z) > 1.5

def load_data(split,num_of_gpus):
    ''' Load data based on VLN-CE
    '''
    # dataset_root_dir = base_data_dir # '../VLN/VLNCE/R2R_VLNCE_v1-3'
    dataset_root_dir = '/ssd/xiaxinyuan/code/VLN/VLNCE/R2R_VLNCE_v1-3'
    scene_id_list = []
    total_scans = []
    load_data = []
    split_data = [[] for _ in range(num_of_gpus)]
    trajectory_id_list = get_sub_trajectory_id_list()
    print(len(trajectory_id_list))
    for split in splits:
            with gzip.open(os.path.join(dataset_root_dir, f"{split}", f"{split}.json.gz"), 'rt', encoding='utf-8') as f:
                data = json.load(f)
            for item in data["episodes"]:
                scene_id = item['scene_id'].split('/')[1]
                if scene_id not in scene_id_list:
                    scene_id_list.append(scene_id)
            for item in data["episodes"]:
                if not judge_stair_exists(item['reference_path'])  and (item['trajectory_id'] in trajectory_id_list):
                    instruction = item['instruction']['instruction_text']
                    scene_id = item['scene_id'].split('/')[1]
                    if not ('room' in instruction or 'hall' in instruction or 'way' in instruction):
                        with open(object_path_file, 'a') as f:
                            f.write(f"{scene_id},{item['trajectory_id']},{item['episode_id']},{item['instruction']['instruction_text']}\n")
                        with open(object_id_file, 'a') as f:
                            f.write(f"{item['episode_id']}\n")
                    else:
                        with open(room_path_file, 'a') as f:
                            f.write(f"{scene_id},{item['trajectory_id']},{item['episode_id']},{item['instruction']['instruction_text']}\n")
                        with open(room_id_file, 'a') as f:
                            f.write(f"{item['episode_id']}\n")
                    

    return split_data
    # return load_data, list(set(total_scans))
# splits = ['train', 'val_seen', 'val_unseen', 'test']
splits = ['val_unseen']
# splits = ['val_seen']
# splits = ['val_seen']
num_of_gpus = 4
for split in splits:
    split_data = load_data(split,num_of_gpus)
    print(split_data)
    
