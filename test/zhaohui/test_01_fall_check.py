###
# 加载一组数据，判断是否摔倒
###
import numpy as np
from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from vln import PROJECT_ROOT_PATH
from vln.src.v2.util.common import (
    check_robot_fall, 
    load_data,
    load_scene_usd,
)


# 需要处理的 path_id 列表
path_id_list = [
    6584
]

headless=False
split='train'
the_scan ='7y3sRwLe3Va'
project_path = PROJECT_ROOT_PATH
base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
mp3d_data_dir = f"{project_path}/../Matterport3D/data/v1/scans"
robot_name = 'aliengo' # h1 / aliengo
if robot_name == "aliengo":
    robot_offset = np.array([0.   , 0.   , 0.50])
else:
    robot_offset = np.array([0.   , 0.   , 1.05])
# 获取数据
data_map = load_data(base_data_dir,split,True,True,)

filtered_path_list = []
for scan_id, path_list in data_map.items():
    if scan_id != the_scan:
        continue
    for one_path in path_list:
        path_id = one_path['trajectory_id']
        if path_id not in path_id_list:
            continue
        one_path["start_position"] += robot_offset
        for i, _ in enumerate(one_path["reference_path"]):
            one_path["reference_path"][i] += robot_offset
        # one_path['start_position'][0] += -0.5
        # one_path['reference_path'][0][0] += -0.5
        # one_path['start_position'][1] += -0.5
        # one_path['reference_path'][0][1] += -0.5
        # one_path['start_position'][2] += 0.3
        # one_path['reference_path'][0][2] += 0.3
        filtered_path_list.append(one_path)

# 加载场景
sim_cfg_file = f'{project_path}/vln/configs/sim_cfg_policy_{robot_name}_eval.yaml'
sim_config = SimulatorConfig(sim_cfg_file)
fall_height_threshold = sim_config.config_dict['tasks'][0]['robots'][0]['fall_height_threshold']
# robot_height = sim_config.config_dict['tasks'][0]['robots'][0]['robot_height']
scene_asset_path = load_scene_usd(mp3d_data_dir, the_scan)
sim_config.config.tasks[0].scene_asset_path = scene_asset_path
path_zero=filtered_path_list[0]
start_position = np.array(path_zero["start_position"])
start_rotation = np.array(path_zero["start_rotation"])
sim_config.config.tasks[0].robots[0].position = start_position
sim_config.config.tasks[0].robots[0].orientation = start_rotation
env = BaseEnv(sim_config, headless=headless, webrtc=False)
# from omni.kit.actions.core import get_action_registry
# get_action_registry().get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera").execute()

from pxr import Gf, UsdLux, UsdGeom
import omni.usd
stage = omni.usd.get_context().get_stage()
distant_light = UsdLux.DistantLight.Define(stage, "/World/distant_light")
distant_light.CreateIntensityAttr(1000)
distant_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

up_disk_light = UsdLux.DiskLight.Define(stage, "/World/up_disk_light")
up_disk_light.CreateIntensityAttr(5000)
up_disk_light.CreateRadiusAttr(50.0)
up_disk_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
UsdGeom.Xformable(up_disk_light).AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
up_disk_light = up_disk_light
up_disk_light_position = UsdGeom.Xformable(up_disk_light).AddTranslateOp()

down_disk_light = UsdLux.DiskLight.Define(stage, "/World/down_disk_light")
down_disk_light.CreateIntensityAttr(5000)
down_disk_light.CreateRadiusAttr(50.0)
down_disk_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
down_disk_light = down_disk_light
down_disk_light_position = UsdGeom.Xformable(down_disk_light).AddTranslateOp()



the_task = env._runner.current_tasks[list(env._runner.current_tasks.keys())[0]]
the_robot = the_task.robots[list(the_task.robots.keys())[0]]
the_isaac_robot = the_robot.isaac_robot

for path in filtered_path_list:
    the_path_id = path['trajectory_id']
    # 设置机器人位置
        
    start_position = np.array(path["start_position"])
    start_rotation = np.array(path["start_rotation"])
    reference_path = path['reference_path']
    end_position = reference_path[-1]
    print(f"11111111{start_position}")

    raise_light = 1
    if robot_name == 'aliengo':
        raise_light+= 0.55 
    z = round(start_position[2], 2)
    up_disk_light_position.Set(Gf.Vec3f(start_position[0],  start_position[1],   -z - raise_light))
    down_disk_light_position.Set(Gf.Vec3f(start_position[0],  start_position[1],  z + raise_light))

    the_task.set_single_robot_poses_without_offset(start_position, start_rotation)
    the_isaac_robot.set_world_velocity(np.zeros(6))
    the_isaac_robot.set_joint_velocities(np.zeros(len(the_isaac_robot.dof_names)))
    the_isaac_robot.set_joint_positions(np.zeros(len(the_isaac_robot.dof_names)))
    the_isaac_robot.set_joint_efforts(np.zeros(len(the_isaac_robot.dof_names)))
    # warm up
    for _ in range(240):
        env.step(actions=[{robot_name:{'stand_still': []}}], add_rgb_subframes=False, render=False)
    # 检查是否摔倒
    # robot_position, robot_rotation = the_isaac_robot.get_world_pose()
    robot_position, robot_rotation = the_task.get_robot_poses_without_offset()
    robot_bottom_z = the_robot.get_ankle_height() - sim_config.config_dict['tasks'][0]['robots'][0]['ankle_height']
    is_fall = check_robot_fall(robot_position, robot_rotation, robot_bottom_z,height_threshold=fall_height_threshold)
    print(f"[scan:{the_scan}][path:{the_path_id}][fall:{is_fall}]")
    while True:
        env.step(actions=[{robot_name:{'stand_still': []}}], add_rgb_subframes=False, render=False)

if(hasattr(env, 'simulation_app')):
    env.simulation_app.close()