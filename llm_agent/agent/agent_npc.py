import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from pxr import Gf
from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat

from llm_agent.utils.utils_omni import get_camera_data
from llm_agent.utils.llm_agent import LLM_Agent
from llm_agent.npc.npc import NPC

def agent_npc(runner, npc: NPC, agent: LLM_Agent, task_config):
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
                             }]
                    }
    res_candidates = [(obj_id, relation) for obj_id, relation in npc.dialogue_graph.spatial_relations.items()]
    for _, value in agent.goal_info.items():
        if isinstance(value, list) and len(value)>0:
            instruction = value[0]
    res_candidates = npc.filter_num(agent.goal, instruction, res_candidates)
    result_dict['original_candidates'] = len(res_candidates)
    
    # First Explore Environment Around
    step_time = 0
    rotation_goals = [(quat_to_euler_angles(current_orientation)[2] + degree)%(2*np.pi) - (2*np.pi) if (quat_to_euler_angles(current_orientation)[2] + degree)%(2*np.pi) > np.pi else (quat_to_euler_angles(current_orientation)[2] + degree)%(2*np.pi) for degree in np.linspace(2*np.pi, 0, 10, endpoint=False)]
    while len(rotation_goals)>0:
        rotation_goal= rotation_goals.pop()
        while abs(quat_to_euler_angles(current_orientation)[2] - rotation_goal) > 0.1:
            step_time += 1
            pbar.update(1)
            actions = {
                "h1": {
                    'rotate': [euler_angles_to_quat(np.array([0, 0, rotation_goal]))],
                },
            }
            if step_time%100==0 or step_time <= 3:
                agent.bev_map.step_time = step_time
                obs = runner.step(actions=actions, render = True)
                rgb, depth = agent.update_memory(dialogue_result=None, update_candidates= True, verbose=task_config['verbose']) 
            else:
                obs = runner.step(actions=actions, render = False)
            current_orientation = obs['h1']['orientation']
            if (obs['h1']['position'][2] < task_config['fall_threshold']) or step_time>1000: # robot falls down
                break

    # Taks Begins
    last_time = 0
    dialogue = None
    stop_flag = False
    while not stop_flag:
        # make decision
        content = agent.make_decision()

        nav_path, node_type = agent.navigate_p2p(current_position, content['goal'], task_config['verbose'])
        ori_len = len(agent.bev_map.candidates)
        while not stop_flag :
            # move one simulation step
            step_time += 1
            pbar.update(1)
            if len(nav_path)==0:
                if content['type']=='candidate':
                    break
                else:
                    target_node = agent.node_to_sim(agent.transfer_to_node(content['goal']))
                    nav_path = agent.sample_points_between_two_points(list(current_position), target_node, step=1)
                    nav_path = [tuple(i) for i in nav_path]
                    if len(nav_path)>1:
                        nav_path.pop(0)
                    else:
                        nav_path = [tuple(target_node)]
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
                    nav_path, node_type = agent.navigate_p2p(current_position, content['goal'], task_config['verbose'])

                plt.close('all')
                print('time per step:', (time.time()-last_time)/200)
                print('current_position:', current_position)

                agent.bev_map.step_time = step_time
                obs = runner.step(actions=actions, render = True)
                rgb, depth = agent.update_memory(dialogue_result=dialogue, verbose=task_config['verbose']) 
                dialogue = None
                landmarks = npc.update_seen_objects()
                last_position = current_position
                current_position = obs['h1']['position']
                navigate_info = obs['h1']['move_along_path']

                if task_config['verbose']:
                    if not os.path.exists(os.path.join('images', str(step_time))):
                        os.makedirs(os.path.join('images', str(step_time)))
                    rgb_image_behind = get_camera_data("/World/h1/torso_link/h1_camera_debug_01", (512, 512), ['rgba'])['rgba']
                    rgb_image_front = get_camera_data("/World/h1/torso_link/h1_camera_debug", (512, 512), ['rgba'])['rgba']
                    if rgb is not None and rgb.size>0 and rgb_image_behind is not None and rgb_image_front.size>0:
                        rgb_image_behind = rgb_image_behind[..., :3]
                        rgb_image_front = rgb_image_front[..., :3]
                        plt.imsave(os.path.join('images', str(step_time), 'debug_behind_'+str(step_time)+'.jpg'), rgb_image_behind)
                        plt.imsave(os.path.join('images', str(step_time), 'debug_front_'+str(step_time)+'.jpg'), rgb_image_front)
                        
                
                # if current position to targe meet collision, replan a new path
                next_target = nav_path[navigate_info['current_index']+1] if len(nav_path)>navigate_info['current_index']+1 else navigate_info['current_point']
                if agent.is_collision(navigate_info['current_point'], next_target):
                    nav_path, node_type = agent.navigate_p2p(current_position, content['goal'], task_config['verbose'])
                    
                # update result
                result_dict['path'].append({'step_time':step_time, 'position':list(obs['h1']['position']), 'orientation':list(obs['h1']['orientation'])})
                last_time = time.time()
            else:
                obs = runner.step(actions=actions, render = False)
                last_position = current_position
                current_position = obs['h1']['position']
                navigate_info = obs['h1']['move_along_path']

            result_dict['path_length'] += np.linalg.norm(last_position[:2]-current_position[:2])
            # judge if the navigation stop condition has been reached
            if (content['type']=='frontier' and len(agent.bev_map.candidates)>ori_len) or navigate_info['finished']:
                break

            # judge if the simulation stop condition has been reached
            if step_time > task_config['max_step']:
                stop_flag = True

        # Navigation goal is a candidate and has been reached, ask NPC a question
        if content['type']=='candidate':
            # rotate towards target
            while abs(quat_to_euler_angles(obs['h1']['orientation'])[2] - np.arctan2(content['goal'][1]-obs['h1']['position'][1], content['goal'][0]-obs['h1']['position'][0]))>0.02:
                step_time += 1
                pbar.update(1)
                actions['h1'] = {'rotate': [euler_angles_to_quat(np.array([0, 0, np.arctan2(content['goal'][1]-obs['h1']['position'][1], content['goal'][0]-obs['h1']['position'][0])]))]}

                obs = runner.step(actions=actions, render = False)
                last_position = current_position
                current_position = obs['h1']['position']

                if step_time > task_config['max_step']: # max step is reached
                    stop_flag = True
                    break
                if obs['h1']['position'][2] < task_config['fall_threshold']: # robot falls down
                    break

            # get objects in current view
            agent.bev_map.step_time = step_time
            obs = runner.step(actions=actions, render = True)
            rgb, depth = agent.update_memory(dialogue_result=dialogue, verbose=task_config['verbose']) 
            landmarks = npc.update_seen_objects()
            landmarks.extend(npc.update_seen_objects('/World/h1/torso_link/h1_camera_whole_view'))
            landmarks = list(set(landmarks))
            last_position = current_position
            current_position = obs['h1']['position']

            # dialogue
            if result_dict['dialogue_turn'] >= task_config['max_dialogue']:
                stop_flag  = True
            else:
                # get answer
                npc_answer = npc.generate_response(f"Is what I am facing the goal {agent.bev_map.goal}?", landmarks)
                if task_config['verbose']:
                    print(npc_answer)
                result_dict['dialogue'].append({'step_time':step_time, 'question': f"Is what I am facing the goal {agent.bev_map.goal}?", "answer": npc_answer})
                if "yes" in npc_answer.lower():
                    stop_flag = True
                    # update evaluation data
                    result_dict['candidates_reduced'].append(len(res_candidates)-1)
                else:
                    new_res_candidates = npc.filter_num(agent.goal, npc_answer, res_candidates)
                    # update evaluation data
                    result_dict['candidates_reduced'].append(1 + len(res_candidates)-len(new_res_candidates))
                    res_candidates = new_res_candidates
                result_dict['dialogue_turn'] += 1
                
            if result_dict['dialogue_turn'] < task_config['max_dialogue'] and not stop_flag:
                # Ask for more goal information
                # get answer
                agent_question = agent.ask_question()
                npc_answer = npc.generate_response(agent_question, landmarks)
                if task_config['verbose']:
                    print(npc_answer)
                dialogue = {'step_time':step_time, 'question': agent_question, "answer": npc_answer}
                # update evaluation data
                result_dict['dialogue'].append(dialogue)
                new_res_candidates = npc.filter_num(agent.goal, npc_answer, res_candidates)
                result_dict['candidates_reduced'].append(len(res_candidates)-len(new_res_candidates))
                res_candidates = new_res_candidates
                result_dict['dialogue_turn']+=1
            plt.imsave(os.path.join(agent.bev_map.save_path, str(result_dict['dialogue_turn'])+'_view.jpg'), rgb)

    agent.bev_map.step_time = step_time
    obs = runner.step(actions={}, render = True)
    rgb, depth = agent.update_memory(update_candidates= False, verbose=task_config['verbose']) 
    landmarks = npc.update_seen_objects()
    landmarks.extend(npc.update_seen_objects('/World/h1/torso_link/h1_camera_whole_view'))
    landmarks = list(set(landmarks))

    end_time = time.time()
    result_dict['time'] = end_time-start_time
    result_dict['success_view'] = npc.target in landmarks
    result_dict['last_view'] = landmarks
    result_dict['left_candidates'] = res_candidates
    result_dict['path'].append({'step_time':step_time, 'position':list(obs['h1']['position']), 'orientation':list(obs['h1']['orientation'])})
    plt.imsave(os.path.join(agent.bev_map.save_path, 'last_view.jpg'), rgb)
    plt.imsave(os.path.join(agent.bev_map.save_path, 'final_occupancy.jpg'), agent.bev_map.occupancy_map)
    return result_dict
