from typing import List
from numpy import ndarray
from interactive_map import InteractiveMap

from utils.habitat_utils import (
    tf2agent_state,
    make_cfg,
    display_map,
    display_sample,
    save_obs,
    save_state,
    get_position_floor_objects,
)
from utils.mapping_utils import (
    load_map,
    load_pose,
    get_new_mask_pallete,
    get_new_pallete,
    grid_id2pos,
    pos2grid_id,
)

class InteractiveMapGru(InteractiveMap):
    def __init__(self, data_dir, map_config):
        super().__init__(data_dir, map_config)
    
    def set_agent_state(self, agent_state: habitat_sim.AgentState):
        return NotImplementedError
    
    def play_actions(self, start_agent_state: habitat_sim.AgentState, actions_list: List[str], waitkey: bool = False) -> List[ndarray]:
        return NotImplementedError
    
    def save_actions_obs(self, save_dir: str, start_agent_state: habitat_sim.AgentState, actions_list: List[str]):
        return NotImplementedError

    def save_obs_pose(self, save_dir: str, obss: List[dict], poses: List):
        return NotImplementedError


