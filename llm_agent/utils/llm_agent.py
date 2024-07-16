import re
import os
import io
import random
random.seed(2024)
from copy import deepcopy
import numpy as np
from PIL import Image
from typing import Optional
import time
import matplotlib.pyplot as plt
from llm_agent.utils.BEVmap import BEVMap
from llm_agent.utils.path_planner import Node, PathPlanning

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# An agent that does not surpport multiple rounds of QA 
class LLM_Agent:
    def __init__(self, llm, vlm, map_config: dict, planner_config: dict, k_frames_memory=10):
        # llm and vlm
        self.llm = llm  # The large language model.
        self.vlm = vlm

        # memory module params
        self.bev_map = BEVMap(llm, vlm, **map_config)  # Bird's Eye View map
        self.goal_info = {'category': None, 'location': [], 'spatial': [], 'appearance': [], 'others': []}  # Information about the goal object
        self.k_frames_memory = k_frames_memory  # Number of past frames to remember
        self.recent_observation_memory = []  # Store the last k RGB images

        # decision module params
        self.goal_reached = False  # Goal has been reached or not  
        self.navigation_goal_reached = False  # Navigation goal has been reached or not
        planner_config['agent_radius'], planner_config['last_scope'], planner_config['extend_length'],planner_config['consider_range'] = planner_config['agent_radius']/map_config['voxel_size'], planner_config['last_scope']/map_config['voxel_size'], planner_config['extend_length']/map_config['voxel_size'],planner_config['consider_range']/map_config['voxel_size']
        self.planner_config = planner_config  # The params of planner
        self.decision_time = 0  # Decision making times
    
    def reset(self,question = None):
        self.bev_map.reset()
        self.goal_info = {'category': None, 'location': [], 'spatial': [], 'appearance': [], 'others': []}
        self.recent_observation_memory = []
        self.goal_reached = False
        self.navigation_goal_reached = False
        self.decision_time = 0
        if question:
            self.give_question(question)

    ######################## Initialize the Task ########################
    def give_question(self, question:str):
        self.question = question

        # extract goal
        self.goal = self.llm.get_answer('get_goal', question = question)
        print(f'The goal of this episode is {self.goal}')
        self.bev_map.set_goal(self.goal)

        # add new goal info
        self.goal_info['category'] = self.goal
        goal_info = self.llm.get_answer('extract_info', question = question, answer = '')
        info_type = self.llm.get_answer('get_info_type', info = goal_info)
        if 'location' in info_type.lower():
            self.goal_info['location'].append(goal_info)
        elif 'spatial' in info_type.lower():
            self.goal_info['spatial'].append(goal_info)
        elif 'appearance' in info_type.lower():
            self.goal_info['appearance'].append(goal_info)
        else:
            self.goal_info['others'].append(goal_info)
   
    ######################## Memory Module ########################
    def update_memory(self, dialogue_result: Optional[dict] = None, verbose = False, update_candidates = True):
        """
        Updates the BEV map content with a new RGBD image and recent observation memory with a new RGB image.
        Optionally, updates the goal information based on dialogue results with an NPC.
        """
        # Update the BEVmap
        rgb_image, depth_image = self.bev_map.update_occupancy_and_candidates(update_candidates=update_candidates, verbose = verbose)
        
        # Update the recent observation memory
        if rgb_image is not None:
            if len(self.recent_observation_memory) >= self.k_frames_memory:
                self.recent_observation_memory.pop(0)
            self.recent_observation_memory.append(rgb_image)
        
        # If dialogue_result is provided, update the goal info
        if dialogue_result is not None:
            # Code to update the goal info based on the dialogue result
            goal_info = self.llm.get_answer('extract_info', question = dialogue_result['question'], answer = dialogue_result['answer'])
            info_type = self.llm.get_answer('get_info_type', info = goal_info)
            if 'location' in info_type.lower():
                self.goal_info['location'].append(goal_info)
            elif 'spatial' in info_type.lower():
                self.goal_info['spatial'].append(goal_info)
            elif 'appearance' in info_type.lower():
                self.goal_info['appearance'].append(goal_info)
            else:
                self.goal_info['others'].append(goal_info)

        return rgb_image, depth_image
    
    ######################## Decision Module ########################
    def make_decision(self):
        """
        Makes a decision based on candidate and frontier information.
        """
        # if self.goal_reached:
        #     answer = self.answer_question()
        #     return "Answer", answer
        # elif self.navigation_goal_reached:
        #     question = self.ask_question()
        #     return "Ask", question
        # else:
        #     return "Navigate", self.choose_target()
        return self.choose_target()
    
    def reset_baseline(self, question = None):
        self.bev_map.reset()
        self.question = question
        
    def make_decision_baseline(self, current_view, task, goal_info, turning_time):
        """
        Makes a decision based on candidate and frontier information.
        """
        image_buf = io.BytesIO()
        Image.fromarray(current_view.astype(np.uint8)).save(image_buf, format='PNG')
        answer = self.vlm.get_answer('make_decision', task = task, goal_info = goal_info, image = image_buf, turning_time = turning_time)
        pattern = r"\d+"
        matches = re.search(pattern, answer)
        if matches:
            action = int(matches.group())
            if action in list(range(1,14)):
                if action == 13:
                    # try:
                    #     question = answer.split(':')[1]
                    # except:
                    #     question = 'Could you please tell me more information about the goal object?' 
                    # return action, question
                    return action, 'Could you please tell me more information about the goal object?'
                else:
                    return action, None
            else:
                return 1, None
        else:
            return 1, None
        
    def get_action_baseline(self, action: tuple, current_position: np.ndarray, current_orientation: float):
        question = action[1]
        action = action[0]
        if action == 1:
            x_goal = current_position[0] + np.cos(current_orientation) * 2
            y_goal = current_position[1] + np.sin(current_orientation) * 2
            return np.array([x_goal, y_goal, current_position[2]])
        elif action == 2:
            x_goal = current_position[0] + np.cos(current_orientation) * 4
            y_goal = current_position[1] + np.sin(current_orientation) * 4
            return np.array([x_goal, y_goal, current_position[2]])
        elif action == 3:
            x_goal = current_position[0] + np.cos(current_orientation) * 6
            y_goal = current_position[1] + np.sin(current_orientation) * 6
            return np.array([x_goal, y_goal, current_position[2]])
        elif action == 4:
            x_goal = current_position[0] + np.cos(current_orientation + np.pi / 4) * 2
            y_goal = current_position[1] + np.sin(current_orientation + np.pi / 4) * 2
            return np.array([x_goal, y_goal, current_position[2]])
        elif action == 5:
            x_goal = current_position[0] + np.cos(current_orientation + np.pi / 4) * 4
            y_goal = current_position[1] + np.sin(current_orientation + np.pi / 4) * 4
            return np.array([x_goal, y_goal, current_position[2]])
        elif action == 6:
            x_goal = current_position[0] + np.cos(current_orientation + np.pi / 4) * 6
            y_goal = current_position[1] + np.sin(current_orientation + np.pi / 4) * 6
            return np.array([x_goal, y_goal, current_position[2]])
        elif action == 7:
            x_goal = current_position[0] + np.cos(current_orientation - np.pi / 4) * 2
            y_goal = current_position[1] + np.sin(current_orientation - np.pi / 4) * 2
            return np.array([x_goal, y_goal, current_position[2]])
        elif action == 8:
            x_goal = current_position[0] + np.cos(current_orientation - np.pi / 4) * 4
            y_goal = current_position[1] + np.sin(current_orientation - np.pi / 4) * 4
            return np.array([x_goal, y_goal, current_position[2]])
        elif action == 9:
            x_goal = current_position[0] + np.cos(current_orientation - np.pi / 4) * 6
            y_goal = current_position[1] + np.sin(current_orientation - np.pi / 4) * 6
            return np.array([x_goal, y_goal, current_position[2]])
        elif action == 10:
            goal_orientation = current_orientation + np.pi/2
            goal_orientation = goal_orientation % (2 * np.pi)
            if goal_orientation > np.pi:
                goal_orientation -= 2 * np.pi
            return goal_orientation
        elif action == 11:
            goal_orientation = current_orientation - np.pi/2
            goal_orientation = goal_orientation % (2 * np.pi)
            if goal_orientation > np.pi:
                goal_orientation -= 2 * np.pi
            return goal_orientation
        elif action == 13: 
            return question
        else:
            return 'STOP'

    def choose_target(self)->Node:
        """
        Chooses the best target based on goal_info. 
        If no suitable candidate is found, it chooses a frontier point to explore.
        """
        candidates = [(i, candidate) for i, candidate in enumerate(self.bev_map.candidates) if candidate.get('not_the_goal', False)==False]
        # Score each candidate and find the best one
        if len(candidates)>0:
            candidate_ids = [case[0] for case in candidates]
            description = [f"{case[0]}: {case[1]['description']}" for case in candidates]
            candidate_description = '/n'.join(description)
            all_strings = []
            for _, value in self.goal_info.items():
                if isinstance(value, list):
                    all_strings.extend(value)
                    all_strings.append('')
            goal_info = '\n'.join(all_strings[:-1])
            result = self.llm.get_answer('choose_candidate', description = candidate_description, goal = self.goal, goal_info = goal_info)
            best_index = random.choice(candidate_ids)
            if result:
                match = re.search(r"^\D*(\d+)", result)
                if match:
                    candidate_index = int(match.group(1))
                    best_index = candidate_index if candidate_index in candidate_ids else best_index
                    
            self.bev_map.candidates[best_index]['not_the_goal'] = True
            # self.decision_time+=1
            return {'type': 'candidate', 'goal': self.bev_map.candidates[best_index]['centroids']}
        else:
            # If no candidate is suitable, choose a frontier point
            best_frontier = self.bev_map.get_frontier()
            return {'type': 'frontier', 'goal': best_frontier}

    def navigate_p2p(self, current, target, verbose = False) -> list:
        """
        Sends the target to P2P Navigation for pathfinding.
        """
        # Code to navigate to the next target point
        current = self.transfer_to_node(current)
        target = self.transfer_to_node(target)
        # refresh the map before navigation
        quad_tree = deepcopy(self.bev_map.quad_tree_root)
        radius = int(np.ceil(self.planner_config['agent_radius']))
        x, y = int(max(current.x - radius, 0)), int(max(current.y - radius, 0))
        width, height = int(current.x + radius - x), int(current.y + radius - y)
        quadtree_map = 1 - (self.bev_map.occupancy_map == 0)
        quadtree_map[y: y + height, x: x + width] = np.ones((height, width))
        quad_tree.update(quadtree_map, x, y, width, height)

        path_planner = PathPlanning(quad_tree, quadtree_map, **self.planner_config) # Navigation method
        # path_planner = PathPlanning(self.bev_map.quad_tree_root, 1-(self.bev_map.occupancy_map==0), **self.planner_config) # Navigation method
        node, node_type= path_planner.rrt_star(current, target)
        if verbose:
            if not os.path.exists(os.path.join(ROOT_DIR, 'images', str(self.bev_map.step_time))):
                os.makedirs(os.path.join(ROOT_DIR, 'images', str(self.bev_map.step_time)))
            path_planner.plot_path(node, current, target, os.path.join(ROOT_DIR, 'images', str(self.bev_map.step_time), 'path_'+str(self.bev_map.step_time)+'.jpg'))
        path = []
        while node.parent is not None:
            path.append(self.node_to_sim(node))
            node = node.parent
        
        final_path = []
        path.reverse()
        start_point = self.node_to_sim(current)
        for end_point in path:
            sampled_points = self.sample_points_between_two_points(start_point, end_point)
            final_path.extend(sampled_points)
            start_point = end_point
        # if node_type != 0:
        #     final_path.append(self.node_to_sim(target))
        final_path = [tuple(i) for i in final_path]
        if len(final_path)>0:
            final_path.pop(0)
        return final_path, node_type
    
    def sample_points_between_two_points(self, start, end, step=1):
        x1, y1, z1 = start
        x2, y2, z2 = end
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        unit_vector = ((x2 - x1) / distance, (y2 - y1) / distance)

        num_samples = int(distance / step)
        sampled_points = [[
            x1 + n * step * unit_vector[0], 
            y1 + n * step * unit_vector[1],
            z1
         ] for n in range(num_samples + 1)]

        return sampled_points

    def node_to_sim(self, node):
        if isinstance(node, Node):
            return [(node.x-self.bev_map.quadtree_width/2)*self.bev_map.voxel_size, (node.y-self.bev_map.quadtree_height/2)*self.bev_map.voxel_size, 1.05]
        if len(list(node))==2:
            return [(node[1]-self.bev_map.quadtree_width/2)*self.bev_map.voxel_size, (node[0]-self.bev_map.quadtree_height/2)*self.bev_map.voxel_size, 1.05]
        else:
            raise TypeError(f"Point must be a Node or has length of 2 or 3, but got {type(node).__name__}")

    def transfer_to_node(self, point):
        if isinstance(point, Node):
            return point
        elif len(list(point))==2:
            return Node(point[1], point[0])
        elif len(list(point))==3:
            return Node(point[0]/self.bev_map.voxel_size+self.bev_map.quadtree_width/2, point[1]/self.bev_map.voxel_size+self.bev_map.quadtree_height/2)
        else:
            raise TypeError(f"Point must be a Node or has length of 2 or 3, but got {type(point).__name__}")
        
    def is_collision(self, current, target) -> bool:
        # format input
        current = self.transfer_to_node(current)
        target = self.transfer_to_node(target)
        
        path_planner = PathPlanning(self.bev_map.quad_tree_root, self.bev_map.occupancy_map==2, **self.planner_config) # Navigation method
        # path_planner = PathPlanning(self.bev_map.quad_tree_root, 1-(self.bev_map.occupancy_map==0), **self.planner_config) # Navigation method
        return not path_planner.collision_free(current, target)
    
    ######################## Speak Module ########################
    def answer_question(self) -> str:
        """
        Answers a question using past RGB images.
        """
        # Code to answer the question based on the recent observation memory
        pass

    def ask_question(self) -> str:
        """
        Asks for more information about the goal object.
        """
        description = "\n".join([';'.join(candidate['description'])  for candidate in self.bev_map.candidates if len(candidate['description'])>0 and candidate['not_the_goal']==False])

        all_strings = []
        for _, value in self.goal_info.items():
            if isinstance(value, list):
                all_strings.extend(value)
                all_strings.append('')
        goal_info = '\n'.join(all_strings[:-1])

        result = self.llm.get_answer('ask_question', description = description, goal = self.goal, goal_info = goal_info)
        
        if not result:
            result = 'Could you please tell me more information about the goal object?' 
        return result

