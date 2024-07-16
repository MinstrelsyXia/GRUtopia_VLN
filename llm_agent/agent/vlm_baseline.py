import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat

from llm_agent.utils.utils_omni import get_camera_data
from llm_agent.utils.llm_agent import LLM_Agent
from llm_agent.npc.npc import NPC

def vlm_baseline(runner, npc: NPC, agent: LLM_Agent, task_config):
    start_time = time.time()
    pbar = tqdm(total=task_config['max_step'])
    current_position, current_orientation = runner.get_obj('h1').get_world_pose()
    initial_position = current_position
    # initialize result
    result_dict = {'dialogue': [], 
                   'dialogue_turn': 0, 
                   'candidates_reduced': [], 
                   'fall_times': 0,
                   'path_length': 0, 
                   'path': [{'step_time':0, 
                             'position':list(current_position), 
                             'orientation':list(current_orientation)
                             }],
                    'actions': []
                    }
    res_candidates = [(obj_id, relation) for obj_id, relation in npc.dialogue_graph.spatial_relations.items()]
    res_candidates = npc.dialogue_graph.filter_candidates(npc.history_answer[0], res_candidates)
    result_dict['original_candidates'] = len(res_candidates)
    # Taks Begins
    dialogue = []
    turning_time = 0
    last_time = 0
    step_time = 0
    stop_flag = False
    rgb_image = None
    while not stop_flag:
        while rgb_image is None or rgb_image.size==0:
            actions = {"h1": {'move_along_path': [[tuple(current_position)]]}}
            step_time += 1
            pbar.update(1)
            agent.bev_map.step_time = step_time
            obs = runner.step(actions=actions, render = True)
            rgb_image, _ = agent.bev_map.update_occupancy_and_candidates(update_candidates=False, verbose=task_config['verbose'])
            landmarks = npc.update_seen_objects()
            last_position = current_position
            current_position = obs['h1']['position']
            navigate_info = obs['h1']['move_along_path']
        
        # Make decision #
        agent.bev_map.step_time = step_time
        obs = runner.step(actions={}, render = True)
        rgb_image, _ = agent.bev_map.update_occupancy_and_candidates(update_candidates=False, verbose=task_config['verbose'])
        landmarks = npc.update_seen_objects()
        last_position = current_position
        current_position = obs['h1']['position']
        navigate_info = obs['h1']['move_along_path']

        if task_config['verbose']:
            rgb_image_behind = get_camera_data("/World/h1/torso_link/h1_camera_debug_01", (512, 512), ['rgba'])['rgba']
            rgb_image_front = get_camera_data("/World/h1/torso_link/h1_camera_debug", (512, 512), ['rgba'])['rgba']
            if rgb_image is not None and rgb_image.size>0 and rgb_image_behind is not None and rgb_image_front.size>0:
                rgb_image_behind = rgb_image_behind[..., :3]
                rgb_image_front = rgb_image_front[..., :3]
                plt.imsave('debug_behind.jpg', rgb_image_behind)
                plt.imsave('debug_front.jpg', rgb_image_front)
                plt.imsave('rgb.jpg', rgb_image)

        goal_info = ';'.join(dialogue) if len(dialogue)>0 else None
        action = agent.make_decision_baseline(rgb_image, agent.question, goal_info, turning_time)
        content = agent.get_action_baseline(action, obs['h1']['position'], quat_to_euler_angles(obs['h1']['orientation'])[2])
        result_dict['actions'].append({'step_time': step_time, 'action': action})
        # Execute the decision #
        if action[0] in list(range(1,10)):
            turning_time = 0
            # Navigate to navigation goal
            nav_path, node_type = agent.navigate_p2p(current_position, content, task_config['verbose'])
            while not stop_flag:
                # move one simulation step
                step_time += 1
                pbar.update(1)
                if len(nav_path)==0:
                    break
                actions = {"h1": {'move_along_path': [nav_path]}}
                if step_time % 100 == 0:
                    # if robot falls down or stuck in smw, reset the robot
                    offset = np.linalg.norm(np.mean([path['position'][:2] for path in result_dict['path']][-5:], axis = 0)-current_position[:2]) + np.linalg.norm(np.mean([path['orientation'] for path in result_dict['path']][-5:], axis = 0)-obs['h1']['orientation']) 
                    if (current_position[2] < task_config['fall_threshold']) or (len(result_dict['path'])>=5 and offset<0.05): # robot falls down
                        runner._robots['h1'].isaac_robot.set_world_pose(position=np.array(initial_position), orientation=euler_angles_to_quat(np.array([0, 0, 0])))
                        runner._robots['h1'].isaac_robot.set_joint_velocities(np.zeros(len(runner._robots['h1'].isaac_robot.dof_names)))
                        runner._robots['h1'].isaac_robot.set_joint_positions(np.zeros(len(runner._robots['h1'].isaac_robot.dof_names)))
                        result_dict['fall_times']+=1

                        current_position, _ = runner.get_obj('h1').get_world_pose()
                        nav_path, node_type = agent.navigate_p2p(current_position, content, task_config['verbose'])

                    plt.close('all')
                    if task_config['verbose']:
                        print('time per step:', (time.time()-last_time)/200)
                        print('current_position:', current_position)

                    agent.bev_map.step_time = step_time
                    obs = runner.step(actions=actions, render = True)
                    rgb_image, _ = agent.bev_map.update_occupancy_and_candidates(update_candidates=False, verbose=task_config['verbose'])
                    landmarks = npc.update_seen_objects()
                    last_position = current_position
                    current_position = obs['h1']['position']
                    navigate_info = obs['h1']['move_along_path']
                    
                    # if current position to targe meet collision, replan a new path
                    next_target = nav_path[navigate_info['current_index']+1] if len(nav_path)>navigate_info['current_index']+1 else navigate_info['current_point']
                    if agent.is_collision(navigate_info['current_point'], next_target):
                        nav_path, node_type = agent.navigate_p2p(current_position, content, task_config['verbose'])
                        
                    # update result
                    result_dict['path'].append({'step_time':step_time, 'position':list(obs['h1']['position']), 'orientation':list(obs['h1']['orientation'])})
                    last_time = time.time()
                else:
                    obs = runner.step(actions=actions, render = False)
                    last_position = current_position
                    current_position = obs['h1']['position']
                    navigate_info = obs['h1']['move_along_path']

                result_dict['path_length'] += np.linalg.norm(last_position[:2]-current_position[:2])

                # judge if the simulation stop condition has been reached
                if step_time > task_config['max_step']:
                    stop_flag = True

        # Navigation goal is a candidate and has been reached, ask NPC a question
        elif action[0] in [10, 11]:
            turning_time+=1
            while abs(quat_to_euler_angles(obs['h1']['orientation'])[2] - content)>0.02:
                step_time += 1
                pbar.update(1)
                actions['h1'] = {'rotate': [euler_angles_to_quat(np.array([0, 0, content]))]}
                if step_time % 100 == 0:
                    if task_config['verbose']:
                        print('time per step:', (time.time()-last_time)/200)
                    
                    obs = runner.step(actions=actions, render = True)
                    rgb_image, _ = agent.bev_map.update_occupancy_and_candidates(update_candidates=False, verbose=task_config['verbose'])
                    landmarks = npc.update_seen_objects()
                    last_position = current_position
                    current_position = obs['h1']['position']

                    last_time = time.time()
                    result_dict['path'].append({'step_time':step_time, 'position':list(obs['h1']['position']), 'orientation':list(obs['h1']['orientation'])})
                else:
                    obs = runner.step(actions=actions, render = False)
                    last_position = current_position
                    current_position = obs['h1']['position']

                if step_time > task_config['max_step']: # max step is reached
                    stop_flag = True
                    break
                if obs['h1']['position'][2] < task_config['fall_threshold']: # robot falls down
                    break
        elif action[0] == 13:
            # Ask for more goal information
            landmarks = npc.update_seen_objects()
            # get answer
            npc_answer = npc.generate_response(content, landmarks)
            if task_config['verbose']:
                print(npc_answer)
            result_dict['dialogue'].append({'step_time':step_time, 'question': content, "answer": npc_answer})
            dialogue.append(npc_answer)
            # update evaluation data
            new_res_candidates = npc.filter_num(npc.history_answer[0]['cate'], npc_answer, res_candidates)
            result_dict['candidates_reduced'].append(len(res_candidates)-len(new_res_candidates))
            res_candidates = new_res_candidates
            result_dict['dialogue_turn']+=1
            plt.imsave(os.path.join(agent.bev_map.save_path, str(result_dict['dialogue_turn'])+'_view.jpg'), rgb_image)
        else:
            stop_flag = True

        if result_dict['dialogue_turn'] >= task_config['max_dialogue']:
            stop_flag = True

    agent.bev_map.step_time = step_time
    obs = runner.step(actions={}, render = True)
    rgb_image, _ = agent.bev_map.update_occupancy_and_candidates(update_candidates= False, verbose=task_config['verbose']) 
    landmarks = npc.update_seen_objects()
    landmarks.extend(npc.update_seen_objects('/World/h1/torso_link/h1_camera_whole_view'))
    landmarks = list(set(landmarks))
    
    end_time = time.time()
    result_dict['time'] = end_time-start_time
    result_dict['success_view'] = npc.target in landmarks
    result_dict['last_view'] = landmarks
    result_dict['left_candidates'] = res_candidates
    result_dict['path'].append({'step_time':step_time, 'position':list(obs['h1']['position']), 'orientation':list(obs['h1']['orientation'])})
    plt.imsave(os.path.join(agent.bev_map.save_path, 'last_view.jpg'), rgb_image)
    plt.imsave(os.path.join(agent.bev_map.save_path, 'final_occupancy.jpg'), agent.bev_map.occupancy_map)
    return result_dict
