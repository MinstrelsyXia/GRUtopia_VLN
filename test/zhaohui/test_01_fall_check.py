###
# 加载一组数据，判断是否摔倒
###
<<<<<<< HEAD
from vln.src.dataset.data_utils_multi_env import load_gather_data
import numpy as np
from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from vln.src.dataset.data_utils_multi_env import load_scene_usd
from grutopia.core.util.log import log

def check_robot_fall(robot_position, robot_rotation, robots_bottom_z, pitch_threshold=35, roll_threshold=15, height_threshold=0.5):
    from omni.isaac.core.utils.rotations import quat_to_euler_angles
    roll, pitch, yaw = quat_to_euler_angles(robot_rotation, degrees=True)
    # Check if the pitch or roll exceeds the thresholds
    if abs(pitch) > pitch_threshold or abs(roll) > roll_threshold:
        is_fall = True
        log.info(f"Robot falls down!!!")
        log.info(f"Current Position: {robot_position}, Orientation: {roll, pitch, yaw}")
    else:
        is_fall = False
    
    # Check if the height between the robot base and the robot ankle is smaller than a threshold
    robot_ankle_z = robots_bottom_z
    robot_base_z = robot_position[2]
    if robot_base_z - robot_ankle_z < height_threshold:
        is_fall = True
        log.info(f"Robot falls down!!!")
        log.info(f"Current Position: {robot_position}, Orientation: {roll, pitch, yaw}")
    return is_fall

class Datasets:
    def __init__(self):
        self.mp3d_data_dir = '/ssd/share/Matterport3D/data/v1/scans'
        self.base_data_dir = '/ssd/share/VLN/VLNCE/R2R_VLNCE_v1-3_corrected'
class Args:
    def __init__(self):
        self.datasets = Datasets()

# 需要处理的 path_id 列表
path_id_list = [
    482
]

headless=False
split='val_unseen'
the_scan ='EU6Fwq7SyZv'
project_path = "/ssd/zhaohui/workspace/w61_grutopia_0102"
args = Args()

# 获取数据

data_map, _ = load_gather_data(args, split, filter_same_trajectory=True, filter_stairs=True)
robot_offset = np.array([0.   , 0.   , 1.05])
=======
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

>>>>>>> w61/spring_festival_2025
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
<<<<<<< HEAD
sim_cfg_file = f'{project_path}/vln/configs/sim_cfg_policy_eval.yaml'
sim_config = SimulatorConfig(sim_cfg_file)
scene_asset_path = load_scene_usd(args, the_scan)
=======
sim_cfg_file = f'{project_path}/vln/configs/sim_cfg_policy_{robot_name}_eval.yaml'
sim_config = SimulatorConfig(sim_cfg_file)
fall_height_threshold = sim_config.config_dict['tasks'][0]['robots'][0]['fall_height_threshold']
# robot_height = sim_config.config_dict['tasks'][0]['robots'][0]['robot_height']
scene_asset_path = load_scene_usd(mp3d_data_dir, the_scan)
>>>>>>> w61/spring_festival_2025
sim_config.config.tasks[0].scene_asset_path = scene_asset_path
path_zero=filtered_path_list[0]
start_position = np.array(path_zero["start_position"])
start_rotation = np.array(path_zero["start_rotation"])
sim_config.config.tasks[0].robots[0].position = start_position
sim_config.config.tasks[0].robots[0].orientation = start_rotation
env = BaseEnv(sim_config, headless=headless, webrtc=False)
<<<<<<< HEAD
=======
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



>>>>>>> w61/spring_festival_2025
the_task = env._runner.current_tasks[list(env._runner.current_tasks.keys())[0]]
the_robot = the_task.robots[list(the_task.robots.keys())[0]]
the_isaac_robot = the_robot.isaac_robot

for path in filtered_path_list:
    the_path_id = path['trajectory_id']
    # 设置机器人位置
<<<<<<< HEAD
    start_position = np.array(path["start_position"])
    start_rotation = np.array(path["start_rotation"])
    print(f"11111111{start_position}")
=======
        
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

>>>>>>> w61/spring_festival_2025
    the_task.set_single_robot_poses_without_offset(start_position, start_rotation)
    the_isaac_robot.set_world_velocity(np.zeros(6))
    the_isaac_robot.set_joint_velocities(np.zeros(len(the_isaac_robot.dof_names)))
    the_isaac_robot.set_joint_positions(np.zeros(len(the_isaac_robot.dof_names)))
    the_isaac_robot.set_joint_efforts(np.zeros(len(the_isaac_robot.dof_names)))
    # warm up
    for _ in range(240):
<<<<<<< HEAD
        env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=False, render=False)
=======
        env.step(actions=[{robot_name:{'stand_still': []}}], add_rgb_subframes=False, render=False)
>>>>>>> w61/spring_festival_2025
    # 检查是否摔倒
    # robot_position, robot_rotation = the_isaac_robot.get_world_pose()
    robot_position, robot_rotation = the_task.get_robot_poses_without_offset()
    robot_bottom_z = the_robot.get_ankle_height() - sim_config.config_dict['tasks'][0]['robots'][0]['ankle_height']
<<<<<<< HEAD
    is_fall = check_robot_fall(robot_position, robot_rotation, robot_bottom_z)
    print(f"[scan:{the_scan}][path:{the_path_id}][fall:{is_fall}]")
    while True:
        env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=False, render=False)
=======
    is_fall = check_robot_fall(robot_position, robot_rotation, robot_bottom_z,height_threshold=fall_height_threshold)
    print(f"[scan:{the_scan}][path:{the_path_id}][fall:{is_fall}]")
    while True:
        env.step(actions=[{robot_name:{'stand_still': []}}], add_rgb_subframes=False, render=False)
>>>>>>> w61/spring_festival_2025

if(hasattr(env, 'simulation_app')):
    env.simulation_app.close()