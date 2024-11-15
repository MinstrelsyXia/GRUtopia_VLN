
import sys
sys.path.insert(0,'/isaac-sim/GRUtopia/')

import isaacsim
from omni.isaac.kit import SimulationApp
_simulation_app = SimulationApp({'headless': True, 'anti_aliasing': 0})
import numpy as np
from omni.isaac.core import World
from vln.src.dataset.data_utils_multi_env import load_gather_data
from grutopia.core.config import SimulatorConfig
from grutopia_extension.tasks.VLN_task import VLNTask
from vln.src.dataset.data_utils_multi_env import load_scene_usd
from grutopia.core.runner import SimulatorRunner

class Datasets:
    def __init__(self):
        self.mp3d_data_dir = '/isaac-sim/Matterport3D/data/v1/scans'
        self.base_data_dir = '/isaac-sim/VLN/VLNCE/R2R_VLNCE_v1-3'
class Args:
    def __init__(self):
        self.datasets = Datasets()
args = Args()
sim_config = SimulatorConfig('/isaac-sim/GRUtopia/vln/configs/sample_episodes_sim_cfg.yaml')
_runner = SimulatorRunner(config=sim_config)
config = sim_config.config.tasks[0]
_world = World(physics_dt=0.005, rendering_dt=0.005, stage_units_in_meters=1.0)

datas, _ = load_gather_data(args,'train',filter_same_trajectory=True,filter_stairs=True)
for key, data in datas.items():
    _world.clear()
    scene = _world.scene
    position = np.array(data[0]['start_position'])
    orientation = np.array(data[0]['start_rotation'])
    paths_list_origin = data[0]['reference_path']
    paths_list = []
    for i, path in enumerate(paths_list_origin):
        paths_list.append(np.array(path))
    config.scene_asset_path = load_scene_usd(args,key)
    config.robots[0].position = position
    config.robots[0].orientation = orientation
    task = VLNTask(config, scene)
    _world.add_task(task)
    _world.reset()
    for _ in range(10):
        _world.step(render=True)
    print(f"init env {key}")
    current_tasks = _world._current_tasks
    task_name = list(current_tasks.keys())[0]
    task = current_tasks[task_name]
    robot_name = list(task.robots.keys())[0]
    robot = task.robots[robot_name]
    isaac_robot = robot.isaac_robot
    task.set_robot_poses_without_offset(config.robots[0].position, config.robots[0].orientation)
    isaac_robot.set_world_velocity(np.zeros(6))
    isaac_robot.set_joint_velocities(np.zeros(len(isaac_robot.dof_names)))
    isaac_robot.set_joint_positions(np.zeros(len(isaac_robot.dof_names)))
    robot_poses = task.get_robot_poses_without_offset()
    print(f"init robot ,robot_poses:{robot_poses}")

    i = 0
    warm_step = 100
    finish = False
    while _simulation_app.is_running() and not finish:
        i = i + 1
        if i < warm_step:
            action = {'move_along_path': [[paths_list[0]]]}
            robot.apply_action(action)
            _world.step(render=True)
        elif i == warm_step:
            finish = True

print('+++++++++++++++++++++++++++++success++++++++++++++++++++++++++++')



