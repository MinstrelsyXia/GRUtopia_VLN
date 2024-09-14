import os
from pathlib import Path
from typing import Dict, List, Tuple, Union
from scipy.spatial.transform import Rotation as R

import cv2
import numpy as np
from PIL import Image





def keyboard_control_fast():
    k = cv2.waitKey(1)
    if k == ord("a"):
        action = "turn_left"
    elif k == ord("d"):
        action = "turn_right"
    elif k == ord("w"):
        action = "move_forward"
    elif k == ord("q"):
        action = "stop"
    elif k == ord(" "):
        return k, "record"
    elif k == -1:
        return k, None
    else:
        return -1, None
    return k, action


def show_rgb(obs):
    bgr = cv2.cvtColor(obs["color_sensor"], cv2.COLOR_RGB2BGR)
    cv2.imshow("rgb", bgr)


def save_state(root_save_dir, sim_setting, agent_state, save_count):
    save_name = sim_setting["scene"].split("/")[-1].split(".")[0] + f"_{save_count:06}.txt"
    save_dir = os.path.join(root_save_dir, "pose")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    pos = agent_state.position
    quat = [
        agent_state.rotation.x,
        agent_state.rotation.y,
        agent_state.rotation.z,
        agent_state.rotation.w,
    ]
    with open(save_path, "w") as f:
        f.write(f"{pos[0]}\t{pos[1]}\t{pos[2]}\t{quat[0]}\t{quat[1]}\t{quat[2]}\t{quat[3]}")


def save_states(save_dir, agent_states):
    save_path = Path(save_dir) / "poses.txt"
    print(save_path)

    with open(save_path, "w") as f:
        sep = ""
        for agent_state in agent_states:
            pos = agent_state.position
            quat = [
                agent_state.rotation.x,
                agent_state.rotation.y,
                agent_state.rotation.z,
                agent_state.rotation.w,
            ]
            f.write(f"{sep}{pos[0]}\t{pos[1]}\t{pos[2]}\t{quat[0]}\t{quat[1]}\t{quat[2]}\t{quat[3]}")
            sep = "\n"


def save_obs(
    root_save_dir: Union[str, Path], sim_setting: Dict, observations: Dict, save_id: int, obj2cls: Dict
) -> None:
    """
    save rgb, depth, or semantic images in the observation dictionary according to the sim_setting.
    obj2cls is a dictionary mapping from object id to semantic id in habitat_sim.
    rgb are saved as .png files of shape (width, height) in sim_setting.
    depth are saved as .npy files where each pixel stores depth in meters.
    semantic are saved as .npy files where each pixel stores semantic id.

    """
    root_save_dir = Path(root_save_dir)
    if sim_setting["color_sensor"]:
        # save rgb
        save_name = f"{save_id:06}.png"
        save_dir = root_save_dir / "rgb"
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / save_name
        obs = observations["color_sensor"][:, :, [2, 1, 0]] / 255
        cv2.imwrite(str(save_path), observations["color_sensor"][:, :, [2, 1, 0]])

    if sim_setting["depth_sensor"]:
        # save depth
        if sim_setting["depth_sensor"]:
            save_name = f"{save_id:06}.npy"
            save_dir = root_save_dir / "depth"
            os.makedirs(save_dir, exist_ok=True)
            save_path = save_dir / save_name
            obs = observations["depth_sensor"]
            with open(save_path, "wb") as f:
                np.save(f, obs)

    if sim_setting["semantic_sensor"]:
        # save semantic
        if sim_setting["semantic_sensor"]:
            save_name = f"{save_id:06}.npy"
            save_dir = root_save_dir / "semantic"
            os.makedirs(save_dir, exist_ok=True)
            save_path = save_dir / save_name
            obs = observations["semantic_sensor"]
            obs = cvt_obj_id_2_cls_id(obs, obj2cls)
            with open(save_path, "wb") as f:
                np.save(f, obs)





def cvt_obj_id_2_cls_id(semantic: np.ndarray, obj2cls: Dict) -> np.ndarray:
    h, w = semantic.shape
    semantic = semantic.flatten()
    u, inv = np.unique(semantic, return_inverse=True)
    return np.array([obj2cls[x][0] for x in u])[inv].reshape((h, w))


def set_agent_state(p: np.array, q: np.array):
    """p (3,1), q (4, 1): xyzw"""
    return NotImplementedError

def tf2agent_state(tf: np.array):
    p = tf[:3, 3]
    r = R.from_matrix(tf[:3, :3])
    quat = r.as_quat()  # xyzw
    state = set_agent_state(p, quat)
    return state



def display_sample(
    sim_setting,
    rgb_obs,
    semantic_obs=np.array([]),
    depth_obs=np.array([]),
    lidar_depths=list(),
    obj2cls=dict(),
    bbox_2d_dict=dict(),
    waitkey=True,
):
    d3_40_colors_rgb: np.ndarray = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)

    rgb_obs = np.array(rgb_obs[:, :, [2, 1, 0]])
    obs = rgb_obs / 255.0
    if depth_obs.shape[0] > 0:
        depth_obs_div_10 = np.repeat(depth_obs[:, :, None] / 10, 3, axis=2)
        obs = np.concatenate([obs, depth_obs_div_10], axis=1)

    if semantic_obs.shape[0] > 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        semantic_img = np.asarray(semantic_img)[:, :, :3].astype(float) / 255
        obs = np.concatenate([obs, semantic_img], axis=1)

    if obj2cls and semantic_obs.shape[0] > 0:
        obj_ids = np.unique(semantic_obs)
        cls_ids = [obj2cls.get(i) for i in obj_ids]

    cv2.imshow("observations", obs)
    if waitkey:
        k = cv2.waitKey(0)
    else:
        k = cv2.waitKey(1)

    return k


def get_position_floor_objects(semantic_scene, position, h_thres, concept_type="object"):
    """
    get the objects on the same floor as the agent
    type: object or region
    """
    if concept_type == "object":
        objects = semantic_scene.objects
    elif concept_type == "region":
        objects = semantic_scene.regions
    same_floor_obj_list = []
    for obj_i, obj in enumerate(objects):
        if concept_type == "object":
            obj_h = obj.obb.center[1]
        elif concept_type == "region":
            obj_h = obj.aabb.center[1]
        if obj_h - position[1] < h_thres:
            same_floor_obj_list.append(obj)
    return same_floor_obj_list


def get_agent_floor_objects(semantic_scene, agent, h_thres):
    agent_pos = agent.get_state().position
    return get_position_floor_objects(semantic_scene, agent_pos, h_thres)
