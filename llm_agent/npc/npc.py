import os
import sys
import re
import numpy as np
from collections import defaultdict
np.random.seed(2024)
current_directory = os.path.dirname(os.path.abspath(__file__))
npc_prompt_file = os.path.join(current_directory, 'prompt.txt')
# with open('llm_agent/npc/prompt.txt', 'r', encoding='utf-8') as file:
with open(npc_prompt_file, 'r', encoding='utf-8') as file:
    system_message = file.read()
from llm_agent.utils.utils_omni import get_camera_data, get_face_to_instance_by_2d_bbox

def calc_similarity(e1: np.ndarray, e2: np.ndarray):
    return (e1 * e2).sum() / (np.linalg.norm(e1) * np.linalg.norm(e2))

class Dialogue_Graph:
    def __init__(self, llm, attribute_set, spatial_relations, attr_embedding, model_mapping) -> None:
        self.llm = llm
        self.attribute_set = attribute_set
        self.spatial_relations = spatial_relations
        self.attr_embedding = attr_embedding
        self.id2cate = {}
        self._preprocess(model_mapping)
        self.reset()
        self.candidates = {}

    def _preprocess(self, model_mapping):
        temp_attribute_set = {}
        for obj_id in self.spatial_relations:
            key = model_mapping.get(obj_id)
            if not key in self.attribute_set:
                continue
            temp_attribute_set[obj_id] = self.attribute_set[key]
        
        self.attribute_set = temp_attribute_set

        embedding = {}
        for oid, attrs in self.attr_embedding.items():
            for cp, eb in attrs:
                embedding[cp] = eb
        
        self.attr_embedding = embedding
        
        for k, v in self.spatial_relations.items():
            self.id2cate[k] = v['category']

    def reset(self):
        self.candidates = {}
        self.nodes = {}
        self.visited = set()
        self.sampled = set()
    
    def add_node(self, node_id):
        assert not node_id in self.nodes, f"The node {node_id} has already been added."
        if not node_id in self.attribute_set:
            attributes = []
        else:
            attributes = self.attribute_set[node_id]
        relations = self.spatial_relations[node_id]["nearby_objects"]
        node = {}
        node["node_id"] = node_id
        node["attributes"] = attributes
        node["relations"] = relations
        node["remain"] = len(attributes) + len(relations)
        
        self.nodes[node_id] = node

    def _node_probablity(self, nodes):  
        p = np.array([self.nodes[n]["remain"] for n in nodes])
        p = p / p.sum()
        return p

    def sample_node(self):
        nodes = list(self.nodes.keys())
        p = self._node_probablity(nodes)
        node = np.random.choice(nodes, p=p)

        return node
    

    def sample_attribute(self, node_id):
        assert node_id in self.nodes, f"The node {node_id} is not in the node set."
        node = self.nodes[node_id]
        attributes = node["attributes"]
        attr_score = [_[0] for _ in attributes]
        attr_content = [_[1] for _ in attributes]
        # for _ in attr_content:
        #     attr = _
        #     key = node_id + "/" + attr
        #     if not key in self.visited:
        #         self.visited.add(key)
        #         break
        if len(attr_content) == 0:
            return None
        attr_score = -np.log(np.array(attr_score))
        p = attr_score / attr_score.sum()
        cnt = 0
        while cnt < 50:
            attr = np.random.choice(attr_content, p=p)
            key = node_id + "/" + attr
            if not key in self.visited:
                self.visited.add(key)
                break
            cnt += 1
        # assert cnt < 50, "Execeed the maximum sample time. No attribute is sampled."
        if cnt == 50:
            return None
        self.nodes[node_id]["remain"] -= 1
        return attr
        
    def sample_relation(self, node_id):
        assert node_id in self.nodes, f"The node {node_id} is not in the node set."
        node = self.nodes[node_id]
        relations = node["relations"]
        relations_by_type = defaultdict(list)
        for target_object, [relation, dist] in relations.items():
            target_object_cate = target_object.split('/')[0]
            if target_object_cate == 'other':
                continue

            if relation == "near" and dist > 0.5:
                continue
            
            relations_by_type[relation].append((target_object, dist))
            
        relation_types = list(relations_by_type.keys())
        if len(relation_types) == 0:
            return None, None

        mean_dist = []
        for type in relation_types:
            dists = [_[1] for _ in relations_by_type[type]]
            mean_dist.append(np.mean(dists))
        
        mean_dist = np.array(mean_dist) + 1
        p = 1.0 / mean_dist
        p = p / p.sum()
        
        cnt = 0
        while cnt < 50:
            cnt += 1
            sampled_type = np.random.choice(relation_types, p=p)
            relations = relations_by_type[sampled_type]
            # keys = []
            flag = False
            relations_filtered = []
            object_to_add = []
            for relation in relations:
                target_object, _ = relation
                target_object_cate = target_object.split('/')[0]
                key = node_id + "/" + target_object
                if not key in self.visited:
                    flag = True
                if not target_object in self.nodes:
                    if target_object_cate != "other" and target_object_cate != "others":
                        relations_filtered.append(relation)
                        object_to_add.append(target_object)

                    self.nodes[node_id]["remain"] -= 1
                    # self.add_node(target_object)
                    
                # keys.append(key)
                self.visited.add(key)
                

            if flag and len(relations_filtered) != 0:
                for obj_id in object_to_add:
                    self.add_node(obj_id)
                break

            # assert cnt < 50, "Execeed the maximum sample time. No relation is sampled."
        if cnt == 50:
            return None, None

        length = len(relations_filtered)
        rel_idx = np.random.randint(length)
        sampled_relation = relations_filtered[rel_idx]
        return [sampled_relation], sampled_type

    def sample_event(self):
        res = {}
        node_id = self.sample_node()
        res["node_id"] = node_id
        if np.random.rand() > 0.8: # sample an attribute
            attr = self.sample_attribute(node_id)
            if attr is None:
                relations, relation_type = self.sample_relation(node_id)
                res["event_type"] = "relation"
                res["event"] = (relation_type, relations)
            else:
                res["event_type"] = "attribute"
                res["event"] = attr
        else:
            relations, relation_type = self.sample_relation(node_id)
            if relations is None:
                attr = self.sample_attribute(node_id)
                res["event_type"] = "attribute"
                res["event"] = attr
            else:
                res["event_type"] = "relation"
                res["event"] = (relation_type, relations)
        
        return res
    
    def extract_info(self, event):
        node_id = event["node_id"]
        cate = node_id.split('/')[0]
        if event["event_type"] == 'attribute':
            attr = event["event"]
            info = f"""The target object is a {cate}. {attr}"""
        else:
            relation_type, relations = event["event"]
            info = """["""
            cnt = defaultdict(int)
            for obj, dist in relations:
                obj_cate = obj.split('/')[0]
                cnt[obj_cate] += 1
            for key, num in cnt.items():
                info += f""" \"{num} {key} is/are {relation_type} the {cate}.\","""
            info += "]"
        return info, event["event_type"]
    
    def extract_info_v2(self, info):
        if not "cate" in info:
            info['cate'] = 'object'
        cate = info['cate']
        res_str = ""
        if "room" in info:
            room = info["room"]
            splits = room.split('_')
            if splits[-1] in [str(_) for _ in range(10)]:
                splits = splits[:-1]
            room = ' '.join(splits)
            res_str += f"""This object is in {room}."""
        if "relation" in info:
            cnt = defaultdict(int)
            true_or_false = {}
            for _, relation_type, obj_cate in info['relation']:
                cnt[obj_cate] += 1
                true_or_false[obj_cate] = _
            for key, num in cnt.items():
                tof = true_or_false[key]
                
                if key == 'nothing':
                    if tof:
                        res_str += f"""nothing is {relation_type} the {cate}."""
                    else:
                        res_str += f"""something is {relation_type} the {cate}."""
                else:
                    if tof:
                        res_str += f""" There is/are {num} {key} {relation_type} the {cate}."""
                    else:
                        res_str += f""" There is/are no {key} {relation_type} the {cate}."""
                        
                    
        if "appearance" in info:
            for attr in info["appearance"]:
                res_str += attr

        return res_str
            
    def find_difference(self, candidates, current_id=None):
        if current_id is None or not current_id in candidates:
            candidates = list(candidates)
            id_idx = np.random.randint(len(candidates))
            current_id = candidates[id_idx]

        candidates = [(obj_id, self.spatial_relations[obj_id]) for obj_id in candidates]

        cates = set()
        rooms = set()
        relations = defaultdict(list)

        current_relation = {}

        for (obj_id, relation) in candidates:
            cate = relation["category"]
            room = relation["room"]
            if cate != "others":
                cates.add(cate)
            rooms.add(room)
            relation_sets = {"near": defaultdict(int)}

            for target, (rel_type, dist) in relation['nearby_objects'].items():
                target_cate = target.split('/')[0]
                if target_cate != 'other' and target_cate != 'others':
                    if not rel_type in relation_sets:
                        relation_sets[rel_type] = defaultdict(int)
                    
                    
                    relation_sets[rel_type][target_cate] += 1
                    if rel_type != "near":
                        if dist < 1:
                            relation_sets["near"][target_cate] += 1
            
            for key, item in relation_sets.items():
                relations[key].append(item)
                if obj_id == current_id:
                    current_relation[key] = item
            # print(relations)
        # category
        if len(cates) > 1:
            l = [len(_) for _ in cates]
            l = min(l)
            all = set()
            for _ in cates:
                all.add(_[:l].lower())

            if len(all) > 1:
                return "category", cates, current_id # {current_id} is in {room}
        
        # rooms
        temp_room = set([_[:4] for _ in rooms])
        if len(temp_room) > 1:
            return "room", rooms, current_id # {current_id} is in {room}
        
        # print(relations)
        # relations
        for rel_type, target_list in relations.items():
            # print(rel_type, target_list)
            # if len(target_list) < len(candidates):
            n = len(candidates) - len(target_list)
            for i in range(n):
                target_list.append(defaultdict(int))
            
            if not rel_type in current_relation:
                return "relation", (rel_type, "nothing"), current_id
            
            target_dict_current = current_relation[rel_type]
            
            for i, target_dict_a in enumerate(target_list):
                for key in target_dict_current.keys():
                    if key not in target_dict_a:
                        return "relation", (rel_type, key), current_id
            
        return "appearance", None, current_id

    def get_more_info(self, diff, node_id):
        item_rel = self.spatial_relations[node_id]
        item_attr = self.attribute_set[node_id]
        diff_type, diff_content, _ = diff
        if diff_type == 'category':
            return { "cate": item_rel["category"] }
        
        if diff_type == 'room':
            return { "room": item_rel["room"] }
        
        if diff_type == 'relation':
            rel_type_target, target_cate = diff_content
            
            relations = item_rel["nearby_objects"]
            flag = False
            rel_type_set = set()
            for obj_id, (rel_type, dist) in relations.items():
                cate = obj_id.split('/')[0]
                rel_type_set.add(rel_type)
                if rel_type_target == rel_type and target_cate == cate:
                    flag = True
                    break
                if rel_type_target == 'near' and dist < 1 and target_cate == cate:
                    flag = True
                    break

            if target_cate == 'nothing':
                if rel_type_target in rel_type_set:
                    return {"relation": [(False, rel_type_target, "nothing")]}
                else:
                    return {"relation": [(True, rel_type_target, "nothing")]}


            return {"relation": [(flag, rel_type_target, target_cate)]}
        attr = None     
        len_item_attr = len(item_attr)
        attr_idx = np.random.randint(len_item_attr)
        _, attr = item_attr[attr_idx]
        assert attr is not None
        # while len(self.sampled) < len_item_attr:
        #     attr_idx = np.random.randint(len_item_attr)
        #     _, attr = item_attr[attr_idx]
        #     if not attr in self.sampled:
        #         self.sampled.add(attr)
        #         break
        
        return {"appearance": [attr]}
    
    def filter_candidates(self, infos, candidates=None):
        '''
            infos: 
            {
                "cate": str, optional
                "room": [(flag, room), ...]optional
                "relation": [(has_or_not, relation_type, cate), ...]
                "appearance": [attr1, attr2]
            }
            candidates: optional, [obj_id, ]
        '''
        res_candidates = set()

        if candidates is None:
            candidates = [(obj_id, relation) for obj_id, relation in self.spatial_relations.items()]

        for obj_id, relation in candidates:
            if obj_id in self.attribute_set:
                attrs = self.attribute_set[obj_id]
            else:
                attrs = []

            if 'cate' in infos:
                cate_info = infos['cate']
                cate_item = relation['category']
                min_len = min(len(cate_info), len(cate_item))
                if not cate_info[:min_len].lower() == cate_item[:min_len].lower():
                    # if cate_item != 'others':
                    continue

            if 'room' in infos:
                # room_info = infos['room']
                # room_item = relation['room']
                # min_len = min(len(room_info), len(room_item))
                # if not room_info[:min_len].lower() == room_item[:min_len].lower():
                #     continue
                room_info = infos['room']
                room_item = relation['room']
                may_be_candidate = True
                for r_info in room_info:
                    # min_len = min(len(r_info[1]), len(room_item))
                    if r_info[0]==False:
                        # if r_info[1][:min_len].lower() == room_item[:min_len].lower():
                        #     may_be_candidate = False
                        #     break
                        if r_info[1].lower() in room_item.lower():
                            may_be_candidate = False
                            break
                    else:
                        # if r_info[1][:min_len].lower() != room_item[:min_len].lower():
                        #     may_be_candidate = False
                        #     break
                        if r_info[1].lower() not in room_item.lower():
                            may_be_candidate = False
                            break
                if not may_be_candidate:
                    continue 
            
            if 'relation' in infos:
                relation_info = infos['relation']
                relation_item = relation['nearby_objects']
                flag = True
                for rel_info in relation_info:
                    if flag == False:
                        break
                    
                    has_or_not, rel_a, cate_a = rel_info
                    if has_or_not:
                        flag_has = False
                        rel_b_set = set()
                        for key, (rel_b, dist) in relation_item.items():
                            rel_b_set.add(rel_b)
                            cate_b = key.split('/')[0]
                            if rel_a == rel_b and cate_a == cate_b:
                                flag_has = True
                                break

                            if rel_a == 'near' and dist < 1 and cate_a == cate_b:
                                flag_has = True
                                break
                        
                        if cate_a == 'nothing': # if has nothing rel_a the object
                            if rel_a in rel_b_set: # there should not be any rel_a in this candidate
                                flag = False    
                        else:
                            if not flag_has:
                                flag = False
                                # break
                    else:
                        rel_b_set = set()
                        for key, (rel_b, dist) in relation_item.items():
                            rel_b_set.add(rel_b)
                            cate_b = key.split('/')[0]
                            if rel_a == rel_b and cate_a == cate_b:
                                flag = False
                                break

                            if rel_a == 'near' and dist < 1 and cate_a == cate_b:
                                flag = False
                                break

                        if cate_a == 'nothing': # does not have nothing, means has something
                            if not rel_a in rel_b_set:
                                flag = False


                if not flag:
                    continue
            
            if 'appearance' in infos:
                app_info = infos['appearance']
                app_item = attrs
                flag = True
                for app_a in app_info:
                    eb_a = self.llm.get_embedding(app_a)
                    flag_has = False
                    for score, app_b in app_item:
                        eb_b = self.attr_embedding[app_b]
                        if calc_similarity(eb_a, eb_b) > 0.9:
                            flag_has = True
                            break
                    if not flag_has:
                        flag = False
                        break

                if not flag:
                    continue

                
            res_candidates.add(obj_id)
        res_candidates = [(candidate, self.spatial_relations[candidate]) for candidate in list(res_candidates)]
        return res_candidates
    
    def close_to_eachother(self, candidates):
        candidates = list(candidates)
        flag = False
        for i, obj_id in enumerate(candidates):
            relation = self.spatial_relations[obj_id]["nearby_objects"]
            close_to = True
            for obj_id_b in candidates[i+1:]:
                if not obj_id_b in relation:
                    close_to = False
                    break
                
                _, dist = relation[obj_id_b]
                if dist > 0.1:
                    close_to = False
                    break


            if close_to:
                flag = True
                break
        
        return flag

    def get_difference(self, current, target):
        current.append(target)
        candidates = set(current)
        diff = self.find_difference(candidates, current_id=target)
        difference_info = self.get_more_info(diff=diff, node_id=target)
        if category := difference_info.get("cate"):
            return f"The object I'm looking for is a {category}.", difference_info
        elif room := difference_info.get("room"):
            if room.startswith("bedroom") or room.startswith('toilet'):
                room = room.split('_')[0]
            elif room in ["corridor", "kitchen"]:
                pass
            elif room in ["living_room", "dining_room"]:
                room = " ".join(room.split('_'))
            return f"The {self.id2cate[target]} is located in the {room}.", difference_info
        elif relation := difference_info.get("relation"):
            flag, rel_type_target, target_cate = relation[0]
            if flag:
                if target_cate != "nothing":
                    return f"There is a {target_cate} {rel_type_target} the {self.id2cate[target]}.", difference_info
                else:
                    return f"There is nothing {rel_type_target} the {self.id2cate[target]}.", difference_info
            else:
                if target_cate != "nothing":
                    return f"There is no {target_cate} {rel_type_target} the {self.id2cate[target]}.", difference_info
                else:
                    return f"There is something {rel_type_target} the {self.id2cate[target]}.", difference_info
        elif appearance := difference_info.get("appearance"):
            return f"{appearance[0]}", difference_info        
        else:
            raise ValueError("No difference found.")

class NPC:
    def __init__(self, llm, object_caption, spatial_relations, object_caption_embeddings, model_mapping,camera_params):
        self.dialogue_graph = Dialogue_Graph(llm, object_caption, spatial_relations, object_caption_embeddings, model_mapping)
        self.history_answer = []
        self.seen_object = set()
        self.target = None
        self.camera_params = camera_params
        self.llm = llm
    
    def generate_response(self, prompt, landmarks):
        meta_instruction = system_message.format(object=self.get_object_info(self.target))
        if landmarks is not None:
            objects = ', '.join(landmarks)
            prompt += f" \"{objects}\" are the objects in my current view. "
        messages = [
            {"role": "system", "content": meta_instruction},
            {"role": "user", "content": prompt},
        ]
        message = self.llm.vlm.chat.completions.create(
                model='gpt-4o',
                messages=messages,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
        ).choices[0].message.content

        if 'get_difference' in message.lower():
            npc_answer, origin_info = self.dialogue_graph.get_difference(landmarks, self.target)
            self.history_answer.append(origin_info)
            return npc_answer
        return message
    
    def reset(self, target):
        self.history_answer = []
        self.seen_object = set()
        self.target = target
    
    def update_seen_objects(self, prim_path = None):
        if prim_path:
            prim_path = prim_path
        else:
            prim_path = self.camera_params['camera']
        semantic_labels = get_camera_data(prim_path, self.camera_params['resolution'], ["bbox"])['bbox']
        if semantic_labels['data'] is not None and semantic_labels['data'].size>0:
            landmarks = get_face_to_instance_by_2d_bbox(semantic_labels['data'], semantic_labels["info"]['idToLabels'], self.camera_params['resolution'])
            total = {key.lower(): i for i, key in enumerate(self.dialogue_graph.spatial_relations.keys())}
            landmarks = [(list(self.dialogue_graph.spatial_relations.keys())[total[i]]) for i in landmarks if i in total]
            left_landmarks = self.dialogue_graph.filter_candidates(self.history_answer, [(i, self.dialogue_graph.spatial_relations[i]) for i in landmarks if i in self.dialogue_graph.spatial_relations])
            self.seen_object = self.seen_object.union([obj_id for obj_id, info in left_landmarks])
            return landmarks
        return []
    
    def get_object_info(self, object_name: str):
    
        base_info = f"The target object's ID is \"{object_name}\". It is in the {self.dialogue_graph.spatial_relations[object_name]['room']}."
        appearance_info = '\n'.join(
            [
                f"{appearance[1]}"
                for appearance in self.dialogue_graph.attribute_set[object_name]
            ]
        )
        nearby_object_info = '\n'.join(
            [
                f"a {self.dialogue_graph.spatial_relations[k]['category']} is {v[0]} it."
                for k, v in self.dialogue_graph.spatial_relations[object_name]['nearby_objects'].items()
            ]
        )
        
        return f"""## Information about the object you want to find
                {base_info}

                ## Its appearances
                {appearance_info}

                ## Nearby the object, there are:
                {nearby_object_info}
                """

    def filter_num(self, category, npc_answer, candidates):
        if npc_answer:
            language_prompt = (
                    f"Please parse the information related to goal object {category} in Input as format like this:"
                    +"\nroom: (yes or no, room), ..."
                    +"\nrelation: (yes or no, relation_type, category), ..."
                    +"\nappearance: (attribute), ..."
                    +"\n\nHere are some examples:"
                    +"\nInput: No. The chair I am trying to find is not located in the bedroom."
                    +"\nOutput:"
                    +"\nroom: (no, bedroom)" 
                    +"\n\nInput: There is a vase on the cabinet.(The goal object is cabinet)"
                    +"\nOutput:"
                    +"\nrelation: (yes, in, vase)"
                    +"\n\nInput: It has a rectangular shape with rounded edges."
                    +"\nOutput:"
                    +"\nappearance: (It has a rectangular shape with rounded edges.)"
                    +"\n\nInput: No. It's not the cabinet I'm looking for."
                    +"\nOutput:"
                    +"\n-1"
                    +f"\n\nInput:{npc_answer}"
                    +"\nOutput:" 
                )
            response = self.llm.vlm.chat.completions.create(
                model='gpt-4o',  # Make sure to use the correct model name
                messages=[{"role": "user", 
                            "content": language_prompt
                            }
                        ],
                max_tokens=100
            )
            result = response.choices[0].message.content if response.choices else None
            print(result)
            if result == None or "-1" in result:
                npc_answer = {}
                npc_answer['cate'] = category
                res_candidates = self.dialogue_graph.filter_candidates(npc_answer, candidates)
                return candidates
            pattern = re.compile(r'(\w+):((?: \(.+?\),?)+)')
            result_dict = {}
            matches = pattern.findall(result)
            for match in matches:
                key = match[0].strip()
                values = re.findall(r'\((.+?)\)', match[1].strip())
                if key in result_dict:
                    result_dict[key].extend(values)
                else:
                    result_dict[key] = values
            print(result_dict)
            npc_answer = {}
            npc_answer['cate'] = category
            if result_dict.get('room') is not None:
                try:
                    npc_answer['room'] = [(item.split(', ')[0] == 'yes',item.split(', ')[1]) for item in result_dict['room']]
                except:
                    result_dict.pop('room')
            if result_dict.get('relation') is not None:
                try:
                    npc_answer['relation'] = [(item.split(', ')[0] == 'yes', item.split(', ')[1], item.split(', ')[2]) for item in result_dict['relation']]
                except:
                    result_dict.pop('relation')
            if result_dict.get('appearance') is not None:
                try:
                    npc_answer['appearance'] = result_dict['appearance']
                except:
                    result_dict.pop('appearance')
            res_candidates = self.dialogue_graph.filter_candidates(npc_answer, candidates)
            return res_candidates
        else:
            return candidates
