from vln.src.dataset.data_utils_multi_env import load_gather_data
import numpy as np
from vln.src.utils.utils import Config
from vln.src.v2.dataloader.data_reviser import revise_one_data, skip_list

class BasePathKeyDataloader:
    def __init__(
        self,
        base_data_dir,
        split_data_types,
        robot_offset,
        filter_same_trajectory,
        revise_data = True,
    ):
        self.path_key_data = {}
        self.path_key_scan = {}
        self.path_key_split = {}

        args_dict = {
            "datasets":{
                "base_data_dir":base_data_dir,
            }
        }
        for split_data_type in split_data_types:
            load_data_map, _ = load_gather_data(Config(args_dict), split_data_type, filter_same_trajectory=filter_same_trajectory, filter_stairs=True)
            for scan, path_list in load_data_map.items():
                for path in path_list:
                    trajectory_id = path['trajectory_id']
                    if revise_data:
                        if trajectory_id in skip_list:
                            continue
                        path = revise_one_data(path)
                    episode_id = path['episode_id']
                    path_key = f"{trajectory_id}_{episode_id}"
                    path["start_position"] += robot_offset
                    for i, _ in enumerate(path["reference_path"]):
                        path["reference_path"][i] += robot_offset
                    self.path_key_data[path_key] = path
                    self.path_key_scan[path_key] = scan
                    self.path_key_split[path_key] = split_data_type