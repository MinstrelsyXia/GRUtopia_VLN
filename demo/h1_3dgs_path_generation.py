# collect data using 3dgs
'''
1. 将3dgs封装成一个特定的camera.先将camera和scene分开
2. scene 可以单独导入
3. camera要实现: 
    self.get_obs(data_type): rgba, depth, point_cloud
4. camera 要被绑到机器人上
subgoal: 起一个有机器人的环境,可以保存rgb和depth和pcd.
5. 在场景中申请另一个topdown camera, 用于获取场景的俯视图, 生成其occupancy map,通过map自动生成一些可到达点，然后连起来，采集轨迹视频，然后我们再标注instruction
subgoal:测试是否摄像头能跟着移动

'''
import json
import torch
import torchvision
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import hydra
import traceback
#! after sim setup is finished
import open3d as o3d

from grutopia.core.env import BaseEnv
from grutopia.core.config import SimulatorConfig
from grutopia.core.util.log import log
import cv2
class sixth_floor_scene:
    def __init__(self, json_path, sim_config_path, device_number):

        self.json_path = json_path
        self.device_number = device_number
        with open(self.json_path, "r") as json_file:
            lego_json_config = json.load(json_file)

        self.lego_usd_root = lego_json_config["usd_model_root"]
        self.lego_gs_root = lego_json_config["gs_model_root"]
        self.lego_name_list = lego_json_config["model_list"]
        self.lego_device_number = 0
        self.lego_editable = True

        sim_config = SimulatorConfig(sim_config_path) #! scene_usd is empty!
        self.sim_config = sim_config
        self.env = BaseEnv(sim_config, headless=True, webrtc=False)
        self.env.reset()
        
        self.lego_xform_list = [] 
        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.prims import get_prim_at_path
        for lego_name in self.lego_name_list:
            create_prim(usd_path=os.path.join(self.lego_usd_root, lego_name["usd_name"]), 
                        prim_path="/World/" + lego_name["isaac_name"], 
                        position=lego_name["init_position"], 
                        orientation=lego_name["init_orientation"],
                        scale=lego_name["init_scale"])
            self.lego_xform_list.append(XFormPrim("/World/" + lego_name["isaac_name"]))
            print("Create preset usd at /World/" + lego_name["isaac_name"])
        create_prim("/World/light", "DistantLight")
        print("Create preset light at /World/light")
        self.task_name = self.env.config.tasks[0].name
        self.robot_name = self.env.config.tasks[0].robots[0].name
        self.init_agents()
    
    def init_agents(self):
        '''call after self.init_env'''
        self.agents = self.env._runner.current_tasks[self.task_name].robots[self.robot_name].isaac_robot
        self.agent_last_pose = None
        self.agent_init_pose = self.sim_config.config.tasks[0].robots[0].position
        self.agent_init_rotation = self.sim_config.config.tasks[0].robots[0].orientation

        # self.set_agent_pose(self.agent_init_pose, self.agent_init_rotation)
        self.agents.set_joint_velocities(np.zeros(len(self.agents.dof_names)))
        self.agents.set_joint_positions(np.zeros(len(self.agents.dof_names)))
    
    def set_agent_pose(self, position, rotation):
        self.agents.set_world_pose(position, rotation)
    def get_agent_pose(self):
        return self.agents.get_world_pose()


json_path = "thirdparty/landmark_isaacsim_interaction/json_configs/multi_model_sixthfloor.json"
img_path = "thirdparty/landmark_isaacsim_interaction/rendered_imgs/"
file_path = '/cpfs/user/xiaxinyuan/code/aliyun/GRUtopia_VLN/vln/configs/sim_cfg_path_generation.yaml'
device_number = 0
save_dir = os.path.join(img_path, "sixth_floor")
rgb_save_dir = os.path.join(save_dir, "rgb")
depth_save_dir = os.path.join(save_dir, "depth")
pcd_save_dir = os.path.join(save_dir, "pcd")
debug_rgb_save_dir = os.path.join(save_dir, "debug_rgb")
pano_rgb_save_dir = os.path.join(save_dir, "pano_rgb")
os.makedirs(rgb_save_dir, exist_ok=True)
os.makedirs(depth_save_dir, exist_ok=True)
os.makedirs(pcd_save_dir, exist_ok=True)
os.makedirs(debug_rgb_save_dir, exist_ok=True)
os.makedirs(pano_rgb_save_dir, exist_ok=True)

### load data
my_world = sixth_floor_scene(json_path=json_path, sim_config_path=file_path, device_number=device_number)
my_camera = my_world.env._runner.current_tasks['vln_0'].robots['aliengo_0'].sensors['pano_camera_0']
# debug_camera = my_world.env._runner.current_tasks['vln_0'].robots['aliengo_0'].sensors['h1_pano_camera_debug']
pano_camera = my_world.env._runner.current_tasks['vln_0'].robots['aliengo_0'].sensors['topdown_camera_50']
my_camera.set_renderer(my_world.lego_xform_list,my_world.lego_gs_root,my_world.lego_name_list,my_world.lego_device_number,my_world.lego_editable)
# pano_camera.set_renderer(my_world.lego_xform_list,my_world.lego_gs_root,my_world.lego_name_list,my_world.lego_device_number,my_world.lego_editable)

from omegaconf import DictConfig
# from vlmaps.vlmaps.robot.lang_robot import LangRobot
# from vlmaps.vlmaps.navigator.navigator import Navigator

### create a robot:
# path = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (3.0, 4.0, 0.0)]
episode_path = '/cpfs/user/xiaxinyuan/code/aliyun/GRUtopia_VLN/path_generation/test.json'
data = json.load(open(episode_path, 'r'))
path = np.array(data['camera_trajectory'])
path[:,2] += 0.1
i = 0

actions = {'h1_0': {'move_along_path': [path]}}
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.utils.transformations import get_relative_transform

# set inito robot pose:
init_position = path[0]
init_orientation = data['camera_init_orientation']
init_orientation = quat_to_euler_angles(init_orientation)
orientation = [0, 0, init_orientation[2]]
init_orientation = euler_angles_to_quat(orientation)
my_world.set_agent_pose(init_position, init_orientation)

while my_world.env.simulation_app.is_running():
    i += 1
    env_actions = []
    # env_actions.append(actions)
    env_actions.append({'h1_0': {'stand_still': []}})
    obs = my_world.env.step(actions=env_actions)
    # warm up steps:
    if i < 100:
        env_actions.append({'h1_0': {'stand_still': []}})
        continue
    if obs == {}:
        print(f"obs is empty at step {i}")
        continue
    
    if i % 100 == 0:
        rgb_save_path = os.path.join(rgb_save_dir, f"{i}.png")
        depth_save_path = os.path.join(depth_save_dir, f"{i}.png")
        pcd_save_path = os.path.join(pcd_save_dir, f"{i}.pcd")
        # rgb = obs[my_world.task_name][my_world.robot_name]['pano_camera_0']['rgb']
        # depth = obs[my_world.task_name][my_world.robot_name]['pano_camera_0']['depth']
        # pcd = obs[my_world.task_name][my_world.robot_name]['pano_camera_0']['pointcloud']
        # rgb_save = np.transpose(rgb, (1, 2, 0))  # 转换为 [H,W,3]
        data = my_camera.get_data(add_rgb_subframes=True, render=True)
        rgb = data['rgba']
        depth = data['depth']
        pcd = data['pointcloud']
        rgb_save = np.transpose(rgb, (2, 0, 1))  # 转换为 [H,W,3]
        # pano_data = pano_camera.get_data(add_rgb_subframes=True, render=True)
        # pano_rgb = pano_data['rgba']
        # pano_depth = pano_data['depth']
        # pano_pcd = pano_data['pointcloud']
        # use torchvision to save image
        # cv2.imwrite(rgb_save_path, rgb_save)
        # torchvision.utils.save_image(torch.tensor(rgb_save), rgb_save_path)

        rgb_save = np.transpose(rgb, (2, 0, 1)).astype(np.float32)  
        # 确保数值在 0-1 范围内
        if rgb_save.max() > 1.0:
            rgb_save = rgb_save / 255.0
        torchvision.utils.save_image(torch.tensor(rgb_save), rgb_save_path)
        
        torchvision.utils.save_image(torch.tensor(depth), depth_save_path)
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        pano_rgb_save_path = os.path.join(pano_rgb_save_dir, f"{i}.png")
        o3d.io.write_point_cloud(pcd_save_path, pcd_o3d)
        # torchvision.utils.save_image(torch.tensor(pano_rgb), pano_rgb_save_path)
        


my_world.env.simulation_app.close()


import open3d as o3d
from vlmaps.application_my.utils import get_dummy_2d_grid, downsample_pc, visualize_pc
from vlmaps.application_my.isaac_robot_docker import IsaacSimLanguageRobot, build_dataset



class Generate_Obstacle_Map(IsaacSimLanguageRobot):
    def __init__(self, config: DictConfig, sim_config,vln_config,split):
        super().__init__(config, sim_config, vln_config, split=None)


    def get_pc(self, camera,depth):
        grid_2d =  get_dummy_2d_grid(depth.shape[1],depth.shape[0])
        pc = camera.get_world_points_from_image_coords(grid_2d, depth.flatten())
        pc_downsampled = downsample_pc(pc, 150)
        pcd_global = o3d.geometry.PointCloud()
        pcd_global.points = o3d.utility.Vector3dVector(pc_downsampled)
        self.PCD_GLOBAL+=pcd_global
        # visualize_pc(PCD_GLOBAL,headless=False, save_path = "1.jpg")
        return pc_downsampled

    def update_map(self, camera):
        camera_pose = camera.get_world_pose()
        camera_position = camera_pose[0]
        camera_yaw = camera_pose[1]
        camera_yaw = quat_to_euler_angles(camera_yaw)
        depth = obs[my_world.task_name][my_world.robot_name]['camera']['depth']
        pcd = self.get_pc(camera_position, camera_yaw, depth)
        self.my_map.update_map_with_pc(
            pc= self.PCD_GLOBAL,
            camera_position = camera_position,
            camera_orientation= camera_yaw,
            max_depth = 11,
        )
    def _setup_sim(self, sim_config, json_path):
        """
        Setup IsaacSim simulator, load IsaacSim scene and relevant mesh data
        """
        # 加载 lego 模型配置
        with open(json_path, "r") as json_file:
            lego_json_config = json.load(json_file)

        lego_usd_root = lego_json_config["usd_model_root"]
        lego_gs_root = lego_json_config["gs_model_root"]
        lego_name_list = lego_json_config["model_list"]


        self.init_env(sim_config, headless=self.vln_config.headless)
        self.init_omni_env()


        # 创建场景物体
        self.lego_xform_list = []
        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.prims import get_prim_at_path
        
        for lego_name in lego_name_list:
            create_prim(
                usd_path=os.path.join(lego_usd_root, lego_name["usd_name"]),
                prim_path="/World/" + lego_name["isaac_name"],
                position=lego_name["init_position"],
                orientation=lego_name["init_orientation"],
                scale=lego_name["init_scale"]
            )
            self.lego_xform_list.append(XFormPrim("/World/" + lego_name["isaac_name"]))
            print("Create preset usd at /World/" + lego_name["isaac_name"])
        
        create_prim("/World/light", "DistantLight")
        print("Create preset light at /World/light")

        self.sim_config.config.tasks[0].scene_asset_path = lego_usd_root + "/" + lego_name_list[0]["usd_name"]
        init_position = lego_json_config['model_list'][0]["init_position"]
        init_orientation = lego_json_config['model_list'][0]["init_orientation"] 
        self.sim_config.config.tasks[0].robots[0].position = init_position
        self.sim_config.config.tasks[0].robots[0].orientation = init_orientation 
        self.vlmaps_data_dir = self.vlmaps_data_dir 
        self.test_file_save_dir = self.test_dir 
        self.vln_config.log_image_dir = self.test_file_save_dir
        if not os.path.exists(self.test_file_save_dir):
            os.makedirs(self.test_file_save_dir, exist_ok=True)
        self.nav_save_dir = self.test_file_save_dir + "/nav"
        if not os.path.exists(self.nav_save_dir):
            os.makedirs(self.nav_save_dir, exist_ok=True)
        self.init_agents()
        self.init_cam_occunpancy_map(robot_prim=self.agents.prim_path, start_point=init_position)
        self.init_occupancy_map()

    def setup_scene(self):
        """
        Setup the simulator, load scene data and prepare
        the LangRobot interface for navigation
        from LangRobot
        """

        # trajectory_id:37; scene_id:s8pcm....glb; scan_id: s8...
        item = self._setup_sim(self.sim_config, self.json_path) # from VLNdataloader init_one_path

        # for vlmap
        # self.setup_map(self.vlmaps_data_dir)
        self.setup_camera()
        self.eval_helper.setup_task(item)

    def move_to_closets_frontier(self):
        ## get closest frontier
        frontier = self.get_frontiers()#! set frontier_type to closest
        angle = self.get_angle(frontier,coord='uv')
                        # transfer to [0,360]
        angle = (angle + 360) % 360
        self.turn(angle,threshold = 0.05)
        self.move_to(frontier,type='obs')

    def update_all_maps(self):
        topdown_map = self.GlobalTopdownMap(self.vln_config, self.item['scan']) 
        freemap, camera_pose = self.get_global_free_map(verbose=self.vln_config.test_verbose) 
        topdown_map.update_map(freemap, camera_pose, verbose=self.vln_config.test_verbose) 
        self.get_surrounding_free_map(verbose = True)  
        # todo: get pc and max_depth
        self.update_obstacle_map(pc,max_depth)
        self.eval_helper.add_pos(self.agents.get_world_pose()[0])

@hydra.main(
    version_base=None,
    config_path="../config_my",
    config_name="vlmap_dataset_cfg_docker.yaml",
)
def main(config: DictConfig) -> None:
    try:
        vln_envs, vln_config, sim_config, data_camera_list = build_dataset(config.vln_config)
        robot = IsaacSimLanguageRobot(config, sim_config, vln_config=vln_config, split=None)
        
        robot.setup_scene()

        while robot.env.simulation_app.is_running():
            robot.eval_helper.add_pos(robot.agents.get_world_pose()[0])
            robot.warm_up(200)
            while True:
                robot.move_to_closets_frontier()

            #! for debuging
            # goal_obs = robot.ObstacleMap._xy_to_px(robot.eval_helper.goals[:,:2])
            # robot.move_to(goal_obs[1],'obs')
            ''' if the target is reached, then raise EarlyFound and stop the exploration'''
            for cat_i, subgoal in enumerate(parsed_instructions):
                if cat_i >= skip_flag:
                    log.info(f"Executing {subgoal}") # "self.move_to_object('hallway')"
                    try:
                        robot.test_movement(subgoal)
                        skip_flag = skip_flag + 1
                    except EarlyFound as e:
                        log.info(f"{e}. Found object early, stopping exploration.")
                        skip_flag = skipped_i
                        break # break from 'for'
            
            ''' execute the last spatial instruction'''
            for subgoal in parsed_instructions[skip_flag+1:]:
                log.info(f"Executing {subgoal}")
                robot.test_movement(subgoal)

            # robot.env.simulation_app.close()
            last_scene_name = scene_name
            robot.eval_helper.add_pos(robot.agents.get_world_pose()[0])
            robot.save_metric()
            robot.clear_maps()
            break # break from 'while simulator is running'

    except Exception as e:
        log.error(f"Unexpected error: {e}")
        log.error("Traceback: %s", traceback.format_exc())
        ''' restart, and save the episode no matter it is finished or not'''
        if robot.env.simulation_app.is_running():
            try:
                robot.save_metric()
                robot.clear_maps()
                # 确保父目录存在
                os.makedirs(os.path.dirname(config.last_scan_file), exist_ok=True)
                
            except Exception as e:
                log.error(f"Unexpected error while writing file: {e}")
            # sys.exit(1)
            if robot.env is not None and robot.env.simulation_app.is_running():
                robot.env.simulation_app.close()
        log.error(f"Unexpected error: {e}")
        log.error("Traceback: %s", traceback.format_exc())
# if __name__ == "__main__":
#     main()
