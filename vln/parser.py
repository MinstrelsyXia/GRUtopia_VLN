import os,sys
import argparse
import yaml
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSAC_SIM_DIR = os.path.join(os.path.dirname(ROOT_DIR), "isaac-sim-4.0.0")
sys.path.append(ROOT_DIR)
sys.path.append(ISSAC_SIM_DIR)

from grutopia.core.config import SimulatorConfig
from vln.src.utils.utils import dict_to_namespace

def process_args(sim_cfg_file=None, vln_cfg_file=None):
    '''Init parser arguments'''
    parser = argparse.ArgumentParser(description="Main function for VLN in GRUtopia")
    parser.add_argument("--split", default="", type=str, help="The split of the dataset", choices=['train', 'val_seen', 'val_unseen'])
    parser.add_argument("--path_id", default=5593, type=int, help="The number of path id") # 5593
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--test_verbose", action="store_true", default=False)
    parser.add_argument("--save_path_planning", action="store_true", default=False)
    parser.add_argument("--wait", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="vis_one_path", help="The mode of the program")
    parser.add_argument("--scan", type=str, default="", help="The target scan")
    parser.add_argument("--sim_cfg_file", type=str, default="vln/configs/sim_cfg.yaml")
    parser.add_argument("--vln_cfg_file", type=str, default="vln/configs/vln_cfg.yaml")
    parser.add_argument("--save_obs", action="store_true", default=False)
    parser.add_argument("--windows_head", default=False, action="store_true", help="Open a matplotlib window to show the topdown camera for view the robot's action")
    parser.add_argument("--windows_head_type", default="show", choices=['show', 'save'], help="The type of the window head")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--clear_sample_dir", action="store_true", default=False)
    
    # for multi-docker
    # parser.add_argument("--docker_nums", type=int, default=1) # for multi-docker # This should be set in config file
    parser.add_argument("--docker_id", type=int, default=0) # for multi-docker
    parser.add_argument("--lmdb_pathId_dir", type=str)
    args, unknown = parser.parse_known_args()

    '''Init simulation config'''
    if sim_cfg_file is not None:
        args.sim_cfg_file = sim_cfg_file 
    sim_config = SimulatorConfig(args.sim_cfg_file)

    '''Init VLN config'''
    if vln_cfg_file is not None:
        args.vln_cfg_file = vln_cfg_file
    with open(args.vln_cfg_file, 'r') as f:
        vln_config = dict_to_namespace(yaml.load(f.read(), yaml.FullLoader))
    # update args into vln_config
    for key, value in vars(args).items():
        setattr(vln_config, key, value)

    '''Init save directory'''
    vln_config.root_dir = ROOT_DIR
    vln_config.log_dir = os.path.join(ROOT_DIR, "logs")
    try:
        vln_config.log_image_dir = os.path.join(vln_config.log_dir, "images", str(vln_config.split), str(vln_config.path_id), os.getlogin())
    except Exception:
        vln_config.log_image_dir = os.path.join(vln_config.log_dir, "images", str(vln_config.split), str(vln_config.path_id))
    if not os.path.exists(vln_config.log_image_dir):
        os.makedirs(vln_config.log_image_dir)
    
    if "sample_episodes" in vln_config.settings.mode:
        vln_config.sample_episode_dir = os.path.join(ROOT_DIR, "data", "sample_episodes")
        # if os.path.exists(vln_config.sample_episode_dir) and vln_config.settings.force_sample:
        if os.path.exists(vln_config.sample_episode_dir) and args.clear_sample_dir:
            shutil.rmtree(vln_config.sample_episode_dir)
        if not os.path.exists(vln_config.sample_episode_dir):
            os.makedirs(vln_config.sample_episode_dir)
        
        if vln_config.sample_episodes.save_form == 'lmdb':
            lmdb_name_dir = os.path.join(vln_config.sample_episode_dir, vln_config.name)
            vln_config.lmdb_name_dir = lmdb_name_dir
            vln_config.lmdb_path = os.path.join(lmdb_name_dir, "sample_data.lmdb")
            if not os.path.exists(lmdb_name_dir):
                os.makedirs(lmdb_name_dir)
            
            if hasattr(vln_config.sample_episodes, 'docker_nums') and vln_config.sample_episodes.docker_nums > 1:
                vln_config.lmdb_pathId_dir = os.path.join(lmdb_name_dir, 'pathIds')
                if not os.path.exists(vln_config.lmdb_pathId_dir):
                    os.makedirs(vln_config.lmdb_pathId_dir)

    return vln_config, sim_config

