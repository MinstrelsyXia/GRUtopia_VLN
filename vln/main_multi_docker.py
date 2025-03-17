# Author: w61
# Date: 2024/11/12
# Usage: python use multi-docker to run main_sample_episode.py, and NOT use reload function.

import os,sys
import json
from collections import defaultdict

from vln.main_sample_episode import build_dataset
from vln.src.utils.read_lmdb import LmdbReader
from grutopia.core.util.log import log


def task_assign_and_write(args):
    lmdb_reader = LmdbReader(args.lmdb_path)
    if os.path.exists(args.lmdb_path) and args.settings.force_sample_scan:
        # continue sample data from existing lmdb
        scan_pathId_dict = lmdb_reader.check_exist_scan_and_pathId(args.datasets.base_data_dir, args.datasets.splits[0], args.sample_episodes.only_recollect_path_planning_fail)
    else:
        # build new lmdb and sample data from scratch
        scan_pathId_dict = lmdb_reader.load_vln_dataset(args.datasets.base_data_dir, args.datasets.splits[0])
        new_dict = defaultdict(list)
        for k,v in new_dict.items():
            new_dict[k] = v['trajectory_id']
        scan_pathId_dict = new_dict
        
    # Initialize a list to hold the divided dictionaries
    divided_dicts = [{} for _ in range(args.sample_episodes.docker_nums)]

    # Distribute the items in scan_pathId_dict across the divided_dicts
    for index, (key, value) in enumerate(scan_pathId_dict.items()):
        divided_index = index % args.sample_episodes.docker_nums  # Calculate this before using the index
        divided_dicts[divided_index][key] = {}  # Initialize the dictionary for that key
        for v in value:
            divided_dicts[divided_index][key][v] = 'None'

    # Write each divided dictionary to a corresponding JSON file
    for i, divided_dict in enumerate(divided_dicts):
        json_file_path = os.path.join(args.lmdb_pathId_dir, f'scan_pathId_part_{i}.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(divided_dict, json_file, indent=4)
    
    log.info(f"Task assignment and writing to JSON files at {args.lmdb_pathId_dir} done.")
        

if __name__ == "__main__":
    vln_envs, vln_config, sim_config, data_camera_list = build_dataset()
    # 1. assign tasks and write to JSON files
    task_assign_and_write(vln_config)
    
    # 2. Start docker containers
    split = vln_envs.datasets.splits[0]
    lmdb_pathId_dir = vln_envs.lmdb_pathId_dir
    os.system(f"bash vln/scripts/run_multi_dockers.sh {lmdb_pathId_dir} {split}")
