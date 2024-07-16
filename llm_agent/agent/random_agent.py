import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from pxr import Gf
from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat

from llm_agent.utils.utils_omni import get_camera_data, get_face_to_instance_by_2d_bbox
from llm_agent.utils.llm_agent import LLM_Agent
from llm_agent.npc.npc import NPC

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def random_agent(runner, npc: NPC, agent: LLM_Agent, task_config):
    start_time = time.time()
    pbar = tqdm(total=task_config['max_step'])
    current_position, current_orientation = runner.get_obj('h1').get_world_pose()
    initial_position = current_position
    # Initialize result
    result_dict = {'fall_times': 0,
                   'path_length': 0, 
                   'path': [{'step_time':0, 
                             'position':list(current_position), 
                             'orientation':list(current_orientation)
                             }]
                    }
    
    # Taks Begins
    last_time = 0
    step_time = 0
    stop_flag = False
    obs = runner.step(actions={}, render = False)
    
    # Random choose navigation goal #
    free_points = np.column_stack(np.where((1 - (agent.bev_map.occupancy_map == 0))==1))
    random_goal = free_points[np.random.randint(0, free_points.shape[0]), :]

    # Move to the navigation goal #
    agent.bev_map.step_time = step_time
    nav_path, node_type = agent.navigate_p2p(current_position, random_goal, task_config['verbose'])
    while not stop_flag:
        if len(nav_path)==0:
            break
        # Move one simulation step
        step_time += 1
        pbar.update(1)
        actions = {"h1": {'move_along_path': [nav_path]}}
        if step_time % 100 == 0 or step_time < 3:
            # if robot falls down or stuck in smw, reset the robot
            offset = np.linalg.norm(np.mean([path['position'][:2] for path in result_dict['path']][-5:], axis = 0)-current_position[:2]) + np.linalg.norm(np.mean([path['orientation'] for path in result_dict['path']][-5:], axis = 0)-obs['h1']['orientation']) 
            if (current_position[2] < task_config['fall_threshold']) or (len(result_dict['path'])>=5 and offset<0.05): # robot falls down
                runner._robots['h1'].isaac_robot.set_world_pose(position=np.array(initial_position), orientation=euler_angles_to_quat(np.array([0, 0, 0])))
                runner._robots['h1'].isaac_robot.set_joint_velocities(np.zeros(len(runner._robots['h1'].isaac_robot.dof_names)))
                runner._robots['h1'].isaac_robot.set_joint_positions(np.zeros(len(runner._robots['h1'].isaac_robot.dof_names)))
                result_dict['fall_times']+=1

                current_position, _ = runner.get_obj('h1').get_world_pose()
                nav_path, node_type = agent.navigate_p2p(current_position, random_goal, task_config['verbose'])
            
            plt.close('all')
            if task_config['verbose']:
                print('time per step:', (time.time()-last_time)/200)
                print('current_position:', current_position)

            agent.bev_map.step_time = step_time
            obs = runner.step(actions=actions, render = True)
            rgb, depth = agent.bev_map.update_occupancy_and_candidates(update_candidates=False, verbose=task_config['verbose'])
            landmarks = npc.update_seen_objects()
            last_position = current_position
            current_position = obs['h1']['position']
            navigate_info = obs['h1']['move_along_path']

            if task_config['verbose']:
                rgb_image_behind = get_camera_data("/World/h1/torso_link/h1_camera_debug_01", (512, 512), ['rgba'])['rgba']
                rgb_image_front = get_camera_data("/World/h1/torso_link/h1_camera_debug", (512, 512), ['rgba'])['rgba']
                if rgb is not None and rgb.size>0 and rgb_image_behind is not None and rgb_image_front.size>0:
                    rgb_image_behind = rgb_image_behind[..., :3]
                    rgb_image_front = rgb_image_front[..., :3]
                    plt.imsave(os.path.join(ROOT_DIR, 'images', str(step_time), 'debug_behind.jpg'), rgb_image_behind)
                    plt.imsave(os.path.join(ROOT_DIR, 'images', str(step_time), 'debug_front.jpg'), rgb_image_front)
                    if rgb is not None:
                        plt.imsave(os.path.join(ROOT_DIR, 'images', str(step_time), 'rgb.jpg'), rgb)
                    if depth is not None:
                        plt.imsave(os.path.join(ROOT_DIR, 'images', str(step_time), 'depth.jpg'), depth)
            
            # if current position to targe meet collision, replan a new path
            next_target = nav_path[navigate_info['current_index']+1] if len(nav_path)>navigate_info['current_index']+1 else navigate_info['current_point']
            if agent.is_collision(navigate_info['current_point'], next_target):
                nav_path, node_type = agent.navigate_p2p(current_position, random_goal, task_config['verbose'])
            
            # update result
            result_dict['path'].append({'step_time':step_time, 'position':list(obs['h1']['position']), 'orientation':list(obs['h1']['orientation'])})
            last_time = time.time()
        else:
            obs = runner.step(actions=actions, render = False)
            last_position = current_position
            current_position = obs['h1']['position']
            navigate_info = obs['h1']['move_along_path']

        result_dict['path_length'] += np.linalg.norm(last_position[:2]-current_position[:2])

        # Judge if the simulation stop condition has been reached
        if step_time > task_config['max_step']: # max step is reached
            stop_flag = True


    agent.bev_map.step_time = step_time
    obs = runner.step(actions={}, render = True)
    rgb, depth = agent.bev_map.update_occupancy_and_candidates(update_candidates=False, verbose=task_config['verbose'])
    landmarks = npc.update_seen_objects()
    semantic_labels = get_camera_data('/World/h1/torso_link/h1_camera_whole_view', npc.camera_params['resolution'], ["bbox"])['bbox']
    if semantic_labels['data'] is not None and semantic_labels['data'].size>0:
        landmarks = get_face_to_instance_by_2d_bbox(semantic_labels['data'], semantic_labels["info"]['idToLabels'], npc.camera_params['resolution'])
        total = {key.lower(): i for i, key in enumerate(npc.dialogue_graph.spatial_relations.keys())}
        landmarks = [(list(npc.dialogue_graph.spatial_relations.keys())[total[i]]) for i in landmarks if i in total]
        
    end_time = time.time()
    result_dict['time'] = end_time-start_time
    result_dict['success_view'] = npc.target in landmarks
    result_dict['last_view'] = landmarks
    result_dict['path'].append({'step_time':step_time, 'position':list(obs['h1']['position']), 'orientation':list(obs['h1']['orientation'])})
    plt.imsave(os.path.join(agent.bev_map.save_path, 'last_view.jpg'), rgb)
    plt.imsave(os.path.join(agent.bev_map.save_path, 'final_occupancy.jpg'), agent.bev_map.occupancy_map)
    return result_dict