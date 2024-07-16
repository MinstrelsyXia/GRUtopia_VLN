import os
import sys
# sys.path.insert(0, '/home/pjlab/miniconda3/envs/llm_agent/lib/python3.10/site-packages')
import typing_extensions
from typing_extensions import override
sys.path.pop(0)
import json
import yaml
import isaacsim
from omni.isaac.kit import SimulationApp
# simulation_app = SimulationApp({"headless": True})
simulation_app = SimulationApp({"headless": False})
import omni
from pxr import Gf
from tqdm import tqdm
import matplotlib.pyplot as plt
from grutopia.core.config import SimulatorConfig
class SimulatorConfig_evaluate(SimulatorConfig):
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.config = None
        self.validate()
from grutopia.core.runner import SimulatorRunner
from llm_agent.npc.npc import NPC
from llm_agent.agent.agent_npc import agent_npc
from llm_agent.agent.random_agent import random_agent
from llm_agent.agent.vlm_baseline import vlm_baseline
from llm_agent.utils.utils import Evaluator, load_file
from llm_agent.utils.llm_agent import LLM_Agent
from llm_agent.utils.large_model import Llm_intern, Llm_aliyun, Vlm_intern, Vlm_gpt4o, Vlm_qwen
import argparse
import logging
def main():
    parser = argparse.ArgumentParser(description="arguments")
    # parser.add_argument('--scene-name', type=str, default='MVUCSQAKTKJ5EAABAAAAACI8_usd', help='scene name')
    parser.add_argument('--scene-name', type=str, default='/wangliuyi/Matterport3D/data/v1/scans/1LXtFkjw3qL/matterport_mesh/b94039b4eb8947bdb9ff5719d9173eae/b94039b4eb8947bdb9ff5719d9173eae.usd', help='scene name')
    parser.add_argument('--mode', type=str, default='test', help='test or validation')
    parser.add_argument('--agent-type', type=str, default='agent', help='agent baseline or agent_random')
    parser.add_argument('--continue-last', type=bool, default=False, help='continue last experiment')
    parser.add_argument('--dialogue-turn', type=int, default=3, help='1-5')
    parser.add_argument('--llm', type=str, default='gpt', help='gpt, internlm, chatglm, llama, qwenlm')
    parser.add_argument('--vlm', type=str, default='gpt', help='gpt, internvl, qwenvl, llava')
    args = parser.parse_args()
    # load config
    config_dict = load_file("llm_agent/evaluate.yaml")
    task_dict, simulate_dict = config_dict['evaluate'], config_dict['simulator']
    agent_type, continue_last = args.agent_type, args.continue_last
    task_dict['max_dialogue']=args.dialogue_turn
    # initialize agent
    with open('llm_agent/agent/agent_config.yaml', 'r') as file:
        agent_config = yaml.safe_load(file)

    # load llm and vlm
    verbose =  agent_config.pop('verbose')
    gpt_4o = Vlm_gpt4o(azure = True, verbose=verbose)

    # load agent
    agent_llm = args.llm
    agent_vlm = args.vlm 
    if agent_type.split('_')[0] == 'agent':
        agent_config['planner_config']['last_scope']= 1
        if agent_llm == 'gpt':
            agent = LLM_Agent(gpt_4o, gpt_4o, **agent_config)
        elif agent_llm == 'internlm':
            internlm = Llm_intern(verbose)
            agent = LLM_Agent(internlm, gpt_4o, **agent_config)
        elif agent_llm == 'chatglm':
            chatglm = Llm_aliyun('chatglm3-6b', verbose = verbose)
            agent = LLM_Agent(chatglm, gpt_4o, **agent_config)
        elif agent_llm == 'llama':
            llama = Llm_aliyun('llama3-8b-instruct', verbose = verbose)
            agent = LLM_Agent(llama, gpt_4o, **agent_config)
        elif agent_llm == 'qwenlm':
            qwenlm = Llm_aliyun('qwen-7b-chat', verbose= verbose)
            agent = LLM_Agent(qwenlm, gpt_4o, **agent_config)

    if agent_type.split('_')[0] == 'baseline':
        agent_config['planner_config']['last_scope']= 0
        if agent_vlm == 'gpt':
            agent = LLM_Agent(None, gpt_4o, **agent_config)
        elif agent_vlm == 'internvl':
            internvl = Vlm_intern(verbose)
            agent = LLM_Agent(None, internvl, **agent_config)
        elif agent_vlm == 'qwenvl':
            qwenvl = Vlm_qwen(verbose)
            agent = LLM_Agent(None, qwenvl, **agent_config)
        elif agent_vlm == 'llava':
            agent = LLM_Agent(None, llava, **agent_config)

    # scenes data path
    data_path = 'evaluate_data/data/meta'
    scenes_path = os.path.join('evaluate_data/splits', args.mode, 'scenes')
    scenes_name = [args.scene_name]
    scene_info_paths = [
        {
            'scene_name': scene_name.split('_')[0],
            'usd_path': os.path.join(scenes_path, scene_name, 'start_result_dialogue_new_all_mesh.usd'),
            'tasks_path': os.path.join(data_path, scene_name, 'episodes_v4_mini.json'),
            'model_mapping_path': os.path.join(data_path, scene_name, 'model_mapping.json'),        
            'object_dict_path': os.path.join(data_path, scene_name, 'object_dict.json')
        }
        for scene_name in scenes_name
    ]
    # result path
    if agent_type.split('_')[0] == 'agent':
        result_path = os.path.join('result', agent_type + agent_llm + '_' + str(args.dialogue_turn))
    elif agent_type.split('_')[0] == 'baseline':
        result_path = os.path.join('result', agent_type + agent_vlm + '_' + str(args.dialogue_turn))

    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"Directory created: {result_path}")    
    #################################################### DEBUG #################################################### 
    debug_root_path = os.path.join(result_path, 'Debug')
    logging.basicConfig(filename=os.path.join(result_path, args.scene_name + '_log.txt'), filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    ###############################################################################################################
    # object data
    object_caption = load_file('evaluate_data/splits/all/object_captions_score_sort.json')
    object_caption_embeddings = load_file('evaluate_data/splits/all/object_captions_embeddings.pkl')

    # evaluate agent
    money = 0
    evaluator = Evaluator(agent_type)
    for scene_info_path in scene_info_paths:
        # load scene info
        scene_name = scene_info_path['scene_name']
        scene_path = scene_info_path['usd_path']
        tasks = load_file(scene_info_path['tasks_path'])
        model_mapping = load_file(scene_info_path['model_mapping_path'])
        spatial_relations = load_file(scene_info_path['object_dict_path'])
        npc = NPC(gpt_4o, object_caption, spatial_relations, object_caption_embeddings, model_mapping, agent_config['map_config']['camera_parameters'])
        
        # initialize scene
        if continue_last:
            result_names = os.listdir(result_path)
        simulate_dict['scene']['path'] = scene_path
        config = SimulatorConfig_evaluate(simulate_dict).config
        runner = SimulatorRunner(config)
        stage = omni.usd.get_context().get_stage()
        robot = stage.GetPrimAtPath(config.robots[0].prim_path)
        for object_idx, (target_object, rounds) in enumerate(tqdm(tasks.items())):
            # initialize object
            task_info = {}
            result_path_preffix = os.path.join(result_path, scene_name + '_' + '_'.join(target_object.split('/')))
            for task_idx, turn in enumerate(rounds):
                # sift task
                result_save_path = result_path_preffix + '_' + str(task_idx) + '.json'
                if continue_last and os.path.basename(result_save_path) in result_names:
                    continue
                current_position = (turn['start_point'][0], turn['start_point'][1], 1.05)
                task_info['shortest_path_length'] = turn['distance']
                # if task_info['shortest_path_length'] <=3 or task_info['shortest_path_length'] >=10:
                #     continue
                # initialize task
                task_info['position'] = turn['target_point']
                try:
                    if args.mode == 'validation':
                        question = turn['dialogue'][0]['dialog']
                    elif args.mode == 'test':
                        question = turn['dialog'][0]['dialog']
                except:
                    continue
                target = {'model': model_mapping[target_object], 'object': target_object}
                robot.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*current_position))
                # task_begin
                if agent_type == 'agent_random':
                    agent.reset()
                    #################################################### DEBUG #################################################### 
                    agent.bev_map.save_path = os.path.join(debug_root_path, scene_name + '_' + '_'.join(target_object.split('/')) + '_' + str(task_idx), 'anno')
                    if not os.path.exists(agent.bev_map.save_path):
                        os.makedirs(agent.bev_map.save_path)
                        print(f"Directory created: {agent.bev_map.save_path}")
                    ###############################################################################################################
                    npc.reset(target['object'])
                    npc.history_answer.append({'cate': target_object.split('/')[0]  })
                    result_dict = random_agent(runner, npc, agent, task_dict)
                elif agent_type == 'agent':
                    agent.reset(question)
                    #################################################### DEBUG #################################################### 
                    agent.bev_map.save_path = os.path.join(debug_root_path, scene_name + '_' + '_'.join(target_object.split('/')) + '_' + str(task_idx), 'anno')
                    if not os.path.exists(agent.bev_map.save_path):
                        os.makedirs(agent.bev_map.save_path)
                        print(f"Directory created: {agent.bev_map.save_path}")
                    ###############################################################################################################
                    npc.reset(target['object'])
                    npc.history_answer.append({'cate': agent.goal})
                    try:
                        result_dict = agent_npc(runner, npc, agent, task_dict)
                    except ZeroDivisionError as e:
                        logging.exception(f"Error: {e} - Error occurred in episode: {os.path.basename(result_save_path)}")
                        continue
                elif agent_type == 'baseline':
                    agent.reset_baseline(question)
                    #################################################### DEBUG #################################################### 
                    agent.bev_map.save_path = os.path.join(debug_root_path, scene_name + '_' + '_'.join(target_object.split('/')) + '_' + str(task_idx), 'anno')
                    if not os.path.exists(agent.bev_map.save_path):
                        os.makedirs(agent.bev_map.save_path)
                        print(f"Directory created: {agent.bev_map.save_path}")
                    ###############################################################################################################
                    npc.reset(target['object'])
                    npc.history_answer.append({'cate': target_object.split('/')[0]})
                    try:
                        result_dict = vlm_baseline(runner, npc, agent, task_dict)
                    except ZeroDivisionError as e:
                        logging.exception(f"Error: {e} - Error occurred in episode: {os.path.basename(result_save_path)}")
                        continue
                elif agent_type == 'no_agent':
                    agent.reset(question)
                    result_dict = no_agent(runner, agent, task_dict)
                #################################################### DEBUG #################################################### 
                if result_dict.get('debug'):
                    debug_dict = result_dict.pop('debug')
                    for info_name, image_list in debug_dict.items():
                        debug_save_path = os.path.join(debug_root_path, scene_name + '_' + '_'.join(target_object.split('/')) + '_' + str(task_idx), info_name)
                        if not os.path.exists(debug_save_path):
                            os.makedirs(debug_save_path)
                            print(f"Directory created: {debug_save_path}")
                        for step_time, image in image_list:
                            plt.imsave(os.path.join(debug_save_path, str(step_time)+ '.jpg'), image)
                ###############################################################################################################
                # save task result
                evaluate_result = evaluator.update(result_dict, task_info)
                new_data = {'scene': scene_path, 'target': target_object, 'question': question, 'task_info': task_info, 'result': result_dict, 'evaluate_result': evaluate_result, 'money': gpt_4o.token_calculator.calculate_money(0.001275, 5e-6, 15e-6, 13e-8) - money}
                money = gpt_4o.token_calculator.calculate_money(0.001275, 5e-6, 15e-6, 13e-8)
                json_data = json.dumps(new_data, indent=4)
                with open(result_save_path, 'w') as file:
                    file.write(json_data)
                print('\n', result_dict)
                

    evaluator.report(os.path.join(result_path, 'result.json'))

if __name__ == '__main__':
    main()