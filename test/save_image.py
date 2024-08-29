import os
from src.envs.config import SimulatorConfig
from src.envs.env import BaseEnv

file_path = '/home/pjlab/.local/share/ov/pkg/isaac-sim-4.0.0/src/configs/pick_and_place/collect_data_render_1.yaml'
sim_config = SimulatorConfig(file_path)

headless = False
webrtc = False

env = BaseEnv(sim_config, headless=headless, webrtc=webrtc)

import numpy as np
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.loggers.data_logger import DataLogger
from src.controllers.pick_and_place import PostGraspPickPlaceController
from src.tasks import FrankaTablePickAndPlaceForEnv
from src.utils.logger import log
from multiprocessing import Pool, Pipe
from multiprocessing.connection import Connection
from pathos import multiprocessing
from pathlib import Path
import h5py
import traceback

def launch_one_controller(pipe: Connection, task_name,  gripper_joint_closed_positions, gripper_joint_opened_positions, asset_root, dt=1 / 30):

    controller = PostGraspPickPlaceController(
        name=task_name + "controller",
        gripper=None,
        gripper_joint_closed_positions=gripper_joint_closed_positions,
        gripper_joint_opened_positions=gripper_joint_opened_positions,
        base_pose=([-0.6, 0.0, 0.5], [1, 0, 0, 0]),
        mesh_path= str(os.path.join(asset_root, "collision_mesh/green_table.obj")),
        urdf= str(os.path.join(asset_root, "urdfs/panda/panda.urdf")),
        srdf=str(os.path.join(asset_root, "urdfs/panda/panda.srdf")),
        dt=dt, 
    )
    while True:
        obs = pipe.recv()
        try:
            current_time = obs["current_world_time"]
            tmp_obs = obs["observations"]
            raw_action:ArticulationAction = controller.forward(
                picking_position=tmp_obs["cube"]["position"],
                placing_position=tmp_obs["target"]["position"],
                current_joint_positions=tmp_obs["robot"]["qpos"],
                gripper_position=tmp_obs["robot"]["gripper"]["position"],
                gripper_orientation=tmp_obs["robot"]["gripper"]["orientation"],
                end_effector_offset=np.array([0, 0.005, 0]),
                current_time=current_time
            )
            if raw_action == -1:
                positions = tmp_obs["robot"]["qpos"]
                controller_done = True
            else:
                positions = raw_action.joint_positions
                controller_done = controller.is_done()
            # log.debug(positions)
            # if np.any(np.isnan(positions)):
            #     log.error("nan in positions")

            action = {"robot": positions} # action 
            if obs["done"] or controller_done:
                print("controller done")
                controller.reset()
        except Exception as e:
            tb = traceback.format_exc()
            print(e)
            pipe.send((e, tb))
            break
        pipe.send((action, controller_done))


i = 0
obs = env.reset()
task_name_lst = []
task_lst = []
obs_lst = []
task_name2pipe = {}
for task_name, task in  env.runner.current_tasks.items():
    task_name_lst.append(task_name)
    task_lst.append(task)
    obs_lst.append(obs[task_name])

rendering_logger = {} # {task_name: {"images": { camera_name :np.ndarray}, "qpos": np.ndarray, "qvel": np.ndarray, "applied_action": np.ndarray}}
with Pool(int(os.cpu_count() * 0.75)) as p:
    for task_name, task in zip(task_name_lst, task_lst):
        task: FrankaTablePickAndPlaceForEnv
        parent_end, child_end = Pipe()
        task_name2pipe[task_name] = parent_end
        print("Joint closed positions: ", task._robot.gripper.joint_closed_positions)
        print("Joint opened positions: ", task._robot.gripper.joint_opened_positions)
        p.apply_async(launch_one_controller, (child_end, task_name, task._robot.gripper.joint_closed_positions, task._robot.gripper.joint_opened_positions, task._asset_root, env.get_dt()))
        if task._render:
            rendering_logger[task_name] = {"images": {camera_name:[] for camera_name in task.camera_names}, "qpos": [], "qvel": [], "applied_action": []}
    while env.simulation_app.is_running():
        actions = []
        for task_name in task_name_lst:
            task: FrankaTablePickAndPlaceForEnv
            task_name2pipe[task_name].send(obs[task_name])
        for task_name, task in zip(task_name_lst, task_lst):
            action, controller_done = task_name2pipe[task_name].recv()
            if isinstance(action, Exception):
                print(controller_done)
                raise action
            actions.append(action)

            # log rendering data
            if task._render:
                render_obs = obs[task_name]["info"]["render"]
                rendering_logger[task_name]["qpos"].append(render_obs["robot"]["qpos"])
                rendering_logger[task_name]["qvel"].append(render_obs["robot"]["qvel"])
                rendering_logger[task_name]["applied_action"].append(action["robot"])
                for camera_name in task.camera_names:
                    rendering_logger[task_name]["images"][camera_name].append(render_obs["images"][camera_name])
            if obs[task_name]["done"] or controller_done:
                if obs[task_name]["reward"] > 0:
                    log.info(f"task {task_name} success!")
                    log_path = env.runner.log_data(task_name)
                    # log rendering data
                    if task._render:
                        log_path = Path(log_path)
                        with h5py.File(log_path.parent / f'{log_path.stem}.hdf5', 'w') as f:
                            f.create_dataset("observations/qpos", data=np.array(rendering_logger[task_name]["qpos"]))
                            f.create_dataset("observations/qvel", data=np.array(rendering_logger[task_name]["qvel"]))
                            f.create_dataset("action", data=np.array(rendering_logger[task_name]["applied_action"]))
                            for camera_name in task.camera_names:
                                f.create_dataset(f"observations/images/{camera_name}", data=np.array(rendering_logger[task_name]["images"][camera_name]))
                del rendering_logger[task_name]
                rendering_logger[task_name] = {"images": {camera_name:[] for camera_name in task.camera_names}, "qpos": [], "qvel": [], "applied_action": []}
                task.individual_reset()
                env.runner.reset_data_logger(task_name)


        obs = env.step(actions)

        i += 1
    print('Done')
    env.simulation_app.close()
    p.terminate()



## controller parallel
# camera
# light
# robots