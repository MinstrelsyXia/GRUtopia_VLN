from pathlib import Path
import json
from tqdm.notebook import tqdm
import os
import gzip
import copy
import numpy as np
from collections import defaultdict

def transform_rotation_z_90degrees(rotation):
    ''' 沿着z轴旋转90度
    '''
    z_rot_90 = [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]  # 90 degrees = pi/2 radians
    w1, x1, y1, z1 = rotation
    w2, x2, y2, z2 = z_rot_90
    revised_rotation = [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ]
    return revised_rotation

def load_data(dataset_root_dir, split):
    ''' Load data based on VLN-CE
    '''
    total_scans = []
    data_loaded = defaultdict(list)
    total_nums = 0
    with gzip.open(os.path.join(dataset_root_dir, f"{split}", f"{split}.json.gz"), 'rt', encoding='utf-8') as f:
        data = json.load(f)
        # Store original data format
        original_data = copy.deepcopy(data)
        for item in data["episodes"]:
            item["original_start_position"] = copy.copy(item["start_position"])
            item["original_start_rotation"] = copy.copy(item["start_rotation"])
            item["start_position"] = [item["original_start_position"][0], -item["original_start_position"][2], item["original_start_position"][1]]
            item["start_rotation"] = [-item["original_start_rotation"][3], item["original_start_rotation"][0], item["original_start_rotation"][2], -item["original_start_rotation"][1]] # [x,y,z,-w] => [w,x,y,z]
            item["start_rotation"] = transform_rotation_z_90degrees(item["start_rotation"])
            item["scan"] = item["scene_id"].split("/")[1]
            item["c_reference_path"] = []
            if "reference_path" in item.keys():
                for path in item["reference_path"]:
                    item["c_reference_path"].append([path[0], -path[2], path[1]])
                item["reference_path"] = item["c_reference_path"]
                del item["c_reference_path"]
            data_loaded[item["trajectory_id"]].append(item)
            total_nums += 1
            total_scans.append(item["scan"])

    print(f"Loaded data with a total of {len(data_loaded)} trajectories from {split}. The total number of instructions is {total_nums}")
    return data_loaded, list(set(total_scans)), original_data

def find_left_right_error(split, statistic_data_folder):
    data = json.load(open(os.path.join(statistic_data_folder, f"{split}_first_action.json")))
    errors = []
    for k,v in data.items():
        gt = v["gt_first_action"].lower()
        llm = v["llm_inst_first_action"].lower()
        if 'around' in llm or 'round' in llm:
            continue
        if "right" in gt and "left" in llm:
            errors.append(k)
        elif "right" in llm and "left" in gt:
            errors.append(k)
    print("left/right error instructions in {}: {}".format(split, len(errors)))
    return errors

class CorrectInstructions:
    def __init__(self, splits, dataset_root_dir, statistic_data_folder, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_root_dir = dataset_root_dir
        self.statistic_data_folder = statistic_data_folder
        self.splits = splits
        self.data_loaded = defaultdict(list)
        self.original_data = {}  # Store original data format
        
        for split in splits:
            loaded_data, _, orig_data = load_data(self.dataset_root_dir, split)
            self.data_loaded[split] = loaded_data
            self.original_data[split] = orig_data

        self.action_instrs = defaultdict(list)
        for split in splits:
            self.action_instrs[split] = json.load(open(os.path.join(statistic_data_folder, f"{split}_first_action.json")))

        self.gt_actions = defaultdict(list)
        for split in splits:
            self.gt_actions[split] = json.load(open(os.path.join(statistic_data_folder, f"{split}_actions_pred.json")))

        print(1)
        
    def save_corrected_data(self, split, corrected_episodes):
        """Save corrected data back to json.gz format"""
        output_data = copy.deepcopy(self.original_data[split])
        
        # Update instructions in original format
        for episode in output_data["episodes"]:
            episode_id = episode["episode_id"]
            if episode_id in corrected_episodes:
                episode["instruction"]["instruction_text"] = corrected_episodes[episode_id]
        
        # Save to new json.gz file
        output_path = self.output_dir / f"{split}.json.gz"
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            json.dump(output_data, f)
        print(f"Saved corrected data to {output_path}")

    def correct_left_right_error(self, split, correct_and_save=False):
        """Correct instructions where left/right directions are wrong"""
        errors = find_left_right_error(split, self.statistic_data_folder)
        if not errors:
            print(f"No left/right errors found in {split}")
            return

        corrected_episodes = {}
        for error_id in errors:
            traj_id, inst_idx = map(int, error_id.split('_'))
            instr_data = self.action_instrs[split][error_id]
            original_text = instr_data['instruction']
            
            # Get ground truth and predicted first actions
            gt_action = instr_data['gt_first_action'].lower()
            pred_action = instr_data['llm_inst_first_action'].lower()
            
            # Replace incorrect direction with correct one
            corrected_text = original_text.lower()
            if 'left' in pred_action and 'right' in gt_action:
                corrected_text = corrected_text.replace('left', 'right', 1)
            elif 'right' in pred_action and 'left' in gt_action:
                corrected_text = corrected_text.replace('right', 'left', 1)
                
            # Find the corresponding episode_id
            for episode in self.data_loaded[split][traj_id]:
                if episode['instruction']['instruction_text'].rstrip() == original_text:
                    episode_id = episode['episode_id']
                    corrected_episodes[episode_id] = corrected_text
                    break
        
        print(f"Found {len(corrected_episodes)} instructions to correct in {split}")
        
        if correct_and_save and corrected_episodes:
            self.save_corrected_data(split, corrected_episodes)
            print(f"Saved corrected left/right instructions for {split}")
        
        return corrected_episodes

    def correct_heading_error(self, split, angle_threshold, Turn180_candidates=None, corrected_episodes=None, correct_and_save=False):
        gt_actions = self.gt_actions[split]
        traj_instrs = self.action_instrs[split]
        dataset = self.data_loaded[split]
        errors = []
        ang_th = angle_threshold // 15
        for traj_id, traj_gt_actions in gt_actions['actions_pred'].items():
            first_angles = traj_gt_actions[:ang_th]
            if all(x == 2 for x in first_angles) or all(x == 3 for x in first_angles):
                inst_nums = len(dataset[int(traj_id)])
                for i in range(inst_nums):
                    traj_instr_id = f"{traj_id}_{i}"
                    find_flag = False
                    if traj_instr_id in traj_instrs:
                        traj_instr = traj_instrs[traj_instr_id]
                        for index, sub_item in enumerate(dataset[int(traj_id)]):
                            if sub_item['instruction']['instruction_text'].rstrip() == traj_instr['instruction']:
                                real_index = index
                                find_flag = True
                                break
                        if find_flag:
                            if 'around' not in traj_instr['llm_inst_first_action'].lower() and \
                                'round' not in traj_instr['llm_inst_first_action'].lower() and \
                                'back' not in traj_instr['llm_inst_first_action'].lower() and \
                                'u-turn' not in traj_instr['llm_inst_first_action'].lower() and \
                                    'until' not in traj_instr['llm_inst_first_action'].lower() and \
                                        'left' not in traj_instr['llm_inst_first_action'].lower() and \
                                            'right' not in traj_instr['llm_inst_first_action'].lower():
                                errors.append(dataset[int(traj_id)][real_index])              
        print(f"heading error instructions in {split}: {len(errors)}")
        
        # check if has same episode_id data
        check_list = []
        for error in errors:
            if error["episode_id"] not in check_list:
                check_list.append(error["episode_id"])
            else:
                print('!!!')
        
        if errors:
            if corrected_episodes is None:
                corrected_episodes = {}
                update_episodes = False
            else:
                update_episodes = True
            
            revised_episode_ids = []
            for err_idx, error in enumerate(errors):
                episode_id = error["episode_id"]
                # Randomly select a Turn180 instruction prefix
                new_prefix = np.random.choice(Turn180_candidates)
                if update_episodes and episode_id in corrected_episodes:
                    original_instruction = corrected_episodes[episode_id]
                else:
                    original_instruction = error["instruction"]["instruction_text"]
                corrected_instruction = new_prefix + original_instruction
                corrected_episodes[episode_id] = corrected_instruction
                if episode_id not in revised_episode_ids:
                    revised_episode_ids.append(episode_id)
                else:
                    print('!!!')
                
        if correct_and_save:
            self.save_corrected_data(split, corrected_episodes)
            print(f"Saved corrected heading instructions for {split} for {len(corrected_episodes)} instructions.")
        
        return corrected_episodes
    
    def correct_data(self, correct_left_right=False, correct_heading=False, correct_and_save=False):
        corrected_episodes = {}
        if correct_left_right:
            corrected_episodes = self.correct_left_right_error(split, correct_and_save=False)
        if correct_heading:
            corrected_episodes = self.correct_heading_error(split, angle_threshold=180, Turn180_candidates=Turn180_candidates, corrected_episodes=corrected_episodes,correct_and_save=False)
        
        if correct_and_save:
            self.save_corrected_data(split, corrected_episodes)
            print(f"Saved corrected instructions for {split}")
            
        return corrected_episodes

if __name__ == "__main__":
    dataset_root_dir = "/isaac-sim/GRUtopia/data/datasets/R2R_VLNCE_v1-3"
    statistic_data_folder = Path("/isaac-sim/GRUtopia/data/datasets/revised/statistics")
    inst_alias = "inst"
    splits = ["train", "val_seen","val_unseen"]
    action_map = {
        0: "stop",
        1: "move forward",
        2: "turn left",
        3: "turn right",
    }
    
    Turn180_candidates = [
        "Turn around. ",
        "Make a 180-degree turn. ",
        "Reverse direction. ",
        "Go back. ",
        "Face the other way. ",
        "Do a U-turn. ",
        "Head back. ",
        "Turn your body around. ",
        "Circle back. "
    ]

    # Add output directory
    output_dir = "data/datasets/revised/corrected"
    os.makedirs(output_dir, exist_ok=True)
    
    correct_instructions = CorrectInstructions(splits, dataset_root_dir, statistic_data_folder, output_dir)
    for split in splits:
        # correct_instructions.correct_heading_error(split, angle_threshold=180, Turn180_candidates=Turn180_candidates, correct_and_save=True)
        # correct_instructions.correct_left_right_error(split, correct_and_save=True)
        correct_instructions.correct_data(correct_left_right=True, correct_heading=True, correct_and_save=True)
