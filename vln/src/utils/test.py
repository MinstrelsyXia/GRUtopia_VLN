import sys
# sys.path.insert(0,'/ssd/zhaohui/workspace/w61_grutopia/')

from grutopia.core.config import SimulatorConfig
sim_config = SimulatorConfig('vln/configs/sample_episodes_sim_cfg.yaml')
config = sim_config.config.tasks[0]

import isaacsim
from omni.isaac.kit import SimulationApp
_simulation_app = SimulationApp({'headless': True, 'anti_aliasing': 0}) # !!!

from omni.isaac.core import World
_world = World(physics_dt=0.005, rendering_dt=0.005, stage_units_in_meters=1.0)
scene = _world.scene
from grutopia_extension.tasks.VLN_task import VLNTask
from vln.src.dataset.data_utils_multi_env import load_scene_usd
class Datasets:
    def __init__(self,mp3d_data_dir):
        self.mp3d_data_dir = mp3d_data_dir
class Args:
    def __init__(self,datasets):
        self.datasets = datasets

from grutopia.core.runner import SimulatorRunner
_runner = SimulatorRunner(config=sim_config)

config.scene_asset_path = load_scene_usd(Args(Datasets('../Matterport3D/data/v1/scans')),'7y3sRwLe3Va')
config.robots[0].position = [-16.267200469970703, -0.7207760214805603, 0.1518409252166748]
config.robots[0].orientation = [0.7071067811865475, -0.0, -0.0, -0.7071067811865476]
task = VLNTask(config, scene)
_world.add_task(task)
_world.reset()


_world.clear()

configscene_asset_path = load_scene_usd(Args(Datasets('../Matterport3D/data/v1/scans')),'5LpN3gDmAk7')
config.robots[0].position = [-16.267200469970703, -0.7207760214805603, 0.1518409252166748]
config.robots[0].orientation = [0.7071067811865475, -0.0, -0.0, -0.7071067811865476]
task = VLNTask(config, scene)
_world.add_task(task)
_world.reset()

print('success')


