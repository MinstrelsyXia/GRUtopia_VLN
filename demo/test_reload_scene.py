from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container
file_path = './demo/configs/h1_locomotion.yaml'
sim_config = SimulatorConfig(file_path)
headless = False
webrtc = False
if is_in_container():
    headless = True
    webrtc = True

# sim_config.config.tasks[0].scene_asset_path = '/ssd/wangliuyi/data/Matterport3D/data/v1/scans/1LXtFkjw3qL/matterport_mesh/b94039b4eb8947bdb9ff5719d9173eae/isaacsim_b94039b4eb8947bdb9ff5719d9173eae.usd'
sim_config.config.tasks[0].scene_asset_path = '/isaac-sim/Matterport3D/data/v1/scans/1LXtFkjw3qL/matterport_mesh/b94039b4eb8947bdb9ff5719d9173eae/isaacsim_b94039b4eb8947bdb9ff5719d9173eae.usd'
    
# env = BaseEnv(sim_config, headless=headless, webrtc=webrtc)
env = BaseEnv(sim_config, headless=True, webrtc=False, native=True)
import numpy as np
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from grutopia.core.util import log
task_name = env.config.tasks[0].name
robot_name = env.config.tasks[0].robots[0].name
path = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (3.0, 4.0, 0.0)]
i = 0
move_action = {'move_along_path': [path]}
rotate_action = {'rotate': [euler_angles_to_quat(np.array([0, 0, np.pi]))]}
path_finished = False
actions = {'h1': move_action}
while env.simulation_app.is_running():
    i += 1
    env_actions = []
    env_actions.append(actions)
    obs = env.step(actions=env_actions)
    if not path_finished:
        path_finished = obs[task_name][robot_name]['move_along_path'].get('finished', False)
        if path_finished:
            log.info('start rotate')
            actions['h1'] = rotate_action
            start_rotate = True
    if i % 100 == 0:
        print(i)
        print('available observations for h1: {}'.format(obs[task_name][robot_name].keys()))
        print('current position of h1:{}'.format(obs[task_name][robot_name]['position']))
        print('current orientation of h1: {}'.format(quat_to_euler_angles(obs[task_name][robot_name]['orientation'])))
    if i % 500 == 0:
        # env.config.clear_all_callbacks()
        # env.config.clear_instance()
        # env.runner._world.clear()

        env.reset_env()

        sim_config.config.tasks[0].scene_asset_path = '/ssd/wangliuyi/data/Matterport3D/data/v1/scans/1pXnuDYAj8r/matterport_mesh/2e84c97e728d46babd3270f4e1a0ae3a/isaacsim_2e84c97e728d46babd3270f4e1a0ae3a.usd'
        env.runner.add_tasks(sim_config.config.tasks)
        # env.runner.reset()
        move_action = {'move_along_path': [path]}
        rotate_action = {'rotate': [euler_angles_to_quat(np.array([0, 0, np.pi]))]}
        path_finished = False
        actions = {'h1': move_action}
env.simulation_app.close()