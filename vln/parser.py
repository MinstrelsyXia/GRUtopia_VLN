import os,sys
import argparse
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSAC_SIM_DIR = os.path.join(os.path.dirname(ROOT_DIR), "isaac-sim-4.0.0")
sys.path.append(ROOT_DIR)
sys.path.append(ISSAC_SIM_DIR)

from grutopia.core.config import SimulatorConfig
from vln.src.utils.utils import dict_to_namespace

def process_args():
    '''Init parser arguments'''
    parser = argparse.ArgumentParser(description="Main function for VLN in GRUtopia")
    parser.add_argument("--env", default="val_seen", type=str, help="The split of the dataset", choices=['train', 'val_seen', 'val_unseen'])
    parser.add_argument("--path_id", default=5593, type=int, help="The number of path id") # 5593
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--test_verbose", action="store_true", default=False)
    parser.add_argument("--wait", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="vis_one_path", help="The mode of the program")
    parser.add_argument("--sim_cfg_file", type=str, default="vln/configs/sim_cfg.yaml")
    parser.add_argument("--vln_cfg_file", type=str, default="vln/configs/vln_cfg.yaml")
    parser.add_argument("--save_obs", action="store_true", default=False)
    parser.add_argument("--windows_head", default=False, action="store_true", help="Open a matplotlib window to show the topdown camera for view the robot's action")
    parser.add_argument("--windows_head_type", default="show", choices=['show', 'save'], help="The type of the window head")
    args = parser.parse_args()

    args.root_dir = ROOT_DIR
    args.log_dir = os.path.join(ROOT_DIR, "logs")
    args.log_image_dir = os.path.join(args.log_dir, "images", str(args.env), str(args.path_id))
    if not os.path.exists(args.log_image_dir):
        os.makedirs(args.log_image_dir)

    '''Init simulation config'''
    sim_config = SimulatorConfig(args.sim_cfg_file)

    '''Init VLN config'''
    with open(args.vln_cfg_file, 'r') as f:
        vln_config = dict_to_namespace(yaml.load(f.read(), yaml.FullLoader))
    # update args into vln_config
    for key, value in vars(args).items():
        setattr(vln_config, key, value)
    return vln_config, sim_config

