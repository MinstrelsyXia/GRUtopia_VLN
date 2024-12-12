import numpy as np
from grutopia.core.util.log import log
from vln.src.local_nav.path_planner import AStarPlanner
from vln.src.dataset.data_utils_multi_env import load_gather_data
import os
from enum import Enum
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import lmdb
import msgpack_numpy
import torch
import time
from vln.src.discrete.fall_helper import fall_by_problematic_data, get_amend_offset

class StuckChecker:
    def __init__(self, offset, isaac_robot):
        self.offset = offset
        self.last_iter = 0
        position, rotation = isaac_robot.get_world_pose()
        self.agent_last_position = position
        self.agent_last_rotation = rotation

    def check_robot_stuck(self, robot_position, robot_rotation, cur_iter, max_iter=300, threshold=0.2, rotation_threshold = math.pi / 12):
        ''' Check if the robot is stuck
        '''
        robot_position = robot_position - self.offset
        if (cur_iter - self.last_iter) <= max_iter:
            return False
        from omni.isaac.core.utils.rotations import quat_to_euler_angles
        position_diff = np.linalg.norm(robot_position[:2] - self.agent_last_position[:2])
        rotation_diff = abs(quat_to_euler_angles(robot_rotation)[2] - quat_to_euler_angles(self.agent_last_rotation)[2])
        if position_diff < threshold and rotation_diff < rotation_threshold:
            return True
        else:
            self.position_diff = 0
            self.rotation_diff = 0
            self.last_iter = cur_iter
            self.agent_last_position = robot_position
            self.agent_last_rotation = robot_rotation
            return False

class AStarDiscretePlanner(AStarPlanner):
    class Action(Enum):
        stop = 0
        forward = 1
        turn_left = 2
        turn_right = 3
        
    
    class Node:
        def __init__(self, x, y, cost, parent_index, angle):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index
            self.angle = angle

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)+ "," + str(self.angle)

    def __init__(
            self, 
            map_width=500, 
            map_height=500, 
            aperture=200, 
            step_unit_meter=0.25, 
            angle_unit=15, 
            max_step=10000
        ):
        self.resolution = 1
        self.max_step = max_step
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = map_width, map_height
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        self.aperture = aperture
        self.step_unit_meter = step_unit_meter
        self.angle_unit = angle_unit
        if 360 % angle_unit != 0:
            raise ValueError("angle_unit needs to be divided by 360 degrees")
        self.x_step_pixels = step_unit_meter * 10 * map_width / aperture
        self.y_step_pixels = step_unit_meter * 10 * map_height / aperture 

    def get_motions(self, yaw):
        motion = []
        base_angle = round(yaw * (180 / math.pi))
        for i in range(360 // self.angle_unit):
            angle = base_angle + i * self.angle_unit
            if angle > 360:
                angle = angle - 360
            dx = self.x_step_pixels * math.cos(math.radians(angle))
            dy = self.y_step_pixels * math.sin(math.radians(angle))
            dx = round(dx)
            dy = round(dy)
            cost = 1 #math.hypot(dx, dy)
            motion.append([dx, dy, cost, angle])
        return motion

    def get_motion_tensor(self, yaw):
        base_angle = round(yaw * (180 / math.pi))
        num_angle = 360 // self.angle_unit
        angle_t = torch.tensor([base_angle + i * self.angle_unit for i in range(num_angle)]).cuda()
        radians_t = angle_t * torch.pi / 180
        sin_t = torch.sin(radians_t)
        cos_t = torch.cos(radians_t)
        origin_x = torch.tensor([self.x_step_pixels for _ in range(num_angle)]).cuda()
        origin_y = torch.tensor([self.y_step_pixels for _ in range(num_angle)]).cuda()
        dx_t = torch.round(origin_x * cos_t)
        dy_t = torch.round(origin_y * sin_t)
        cost_t = torch.tensor([1 for _ in range(num_angle)]).cuda()
        return dx_t, dy_t, cost_t, angle_t

    def get_cost_tensor(self, cost_map, current_x_t, current_y_t, next_x_t, next_y_t, goal_node, current_angle, next_angle_t, dilation = 5):
        # terrain_cost for: away from wall
        def min_with_dilation(t, dilation, self_min):
            min_t = torch.clone(t)
            min_t[t < self_min] = self_min
            min_t[t >= self_min + dilation] -= dilation
            return min_t
        def max_with_dilation(t, dilation, self_max):
            max_t = torch.clone(t)
            max_t[t > self_max] = self_max
            max_t[t <= self_max - dilation] += dilation
            return max_t
        min_x = torch.min(current_x_t, next_x_t)
        max_x = torch.max(current_x_t, next_x_t)
        min_y = torch.min(current_y_t, next_y_t)
        max_y = torch.max(current_y_t, next_y_t)
        min_x = min_with_dilation(min_x,dilation,self.min_x)
        min_y = min_with_dilation(min_y,dilation,self.min_y)
        max_x = max_with_dilation(max_x,dilation,self.max_x)
        max_y = max_with_dilation(max_y,dilation,self.max_y)
        cost_map_t = torch.tensor(cost_map).cuda()
        terrain_cost = [ 
            torch.mean(
                cost_map_t[int(min_x[i].item()):int(max_x[i].item()) + 1,:]
                    [:, int(min_y[i].item()):int(max_y[i].item()) + 1]
            ).item() 
            for i in range(current_x_t.shape[0])]
        terrain_cost_t = torch.tensor(terrain_cost).cuda()
        terrain_cost_t = torch.where(torch.isnan(terrain_cost_t), 255 + 12, terrain_cost_t)
        terrain_cost_t = torch.round(terrain_cost_t)
        # cos_cost for: facing the goal direction
        goal_node_x_t = torch.tensor([goal_node.x for _ in range(current_x_t.shape[0])]).cuda()
        goal_node_y_t = torch.tensor([goal_node.y for _ in range(current_x_t.shape[0])]).cuda()
        sg_vect_x_t = goal_node_x_t - current_x_t
        sg_vect_y_t = goal_node_y_t - current_y_t
        sg_vector_t = torch.cat((sg_vect_x_t.unsqueeze(0), sg_vect_y_t.unsqueeze(0)), dim=0).T
        se_vect_x_t = next_x_t - current_x_t
        se_vect_y_t = next_y_t - current_y_t
        se_vector_t = torch.cat((se_vect_x_t.unsqueeze(0), se_vect_y_t.unsqueeze(0)), dim=0).T
        cos_theta_t = torch.nn.functional.cosine_similarity(sg_vector_t, se_vector_t, dim=1)
        cos_cost_t = torch.tensor([ 1 for _ in range(current_x_t.shape[0])]).cuda() - cos_theta_t
        # angle_cost for: try not to turn
        distance = pow((current_x_t[0].item() - goal_node.x), 2) + pow((current_y_t[0].item() - goal_node.y), 2)
        start_angle_t = torch.tensor([ current_angle for _ in range(current_x_t.shape[0])]).cuda()
        angle_diff_t = torch.max(start_angle_t, next_angle_t) - torch.min(start_angle_t, next_angle_t)
        tmp = torch.clone(angle_diff_t) * 2
        tmp[angle_diff_t > 180] = 360
        angle_diff_t = tmp - angle_diff_t
        angle_cost_t = torch.round(angle_diff_t / 15)
        if distance <= 8:
            angle_cost_t = torch.round(angle_cost_t / 12)
        return terrain_cost_t + cos_cost_t + angle_cost_t

    def get_cost(self, cost_map, start_node, end_node, goal_node, dilation = 5):
        start_x = start_node.x
        start_y = start_node.y
        end_x = end_node.x
        end_y = end_node.y

        if end_x >= self.max_x or end_y >= self.max_y:
            return 255 + 12
        
        def min_with_dilation(xs,dilation,min_value):
            if min(xs) < min_value:
                return min_value
            if min(xs) - dilation < min_value:
                return min(xs)
            return min(xs) - dilation
        def max_with_dilation(xs,dilation,max_value):
            if max(xs) > max_value:
                return max_value
            if max(xs) + dilation > max_value:
                return max(xs)
            return max(xs) + dilation
        min_x = min_with_dilation((start_x,end_x),dilation,self.min_x)
        max_x = max_with_dilation((start_x,end_x),dilation,self.max_x)
        min_y = min_with_dilation((start_y,end_y),dilation,self.min_y)
        max_y = max_with_dilation((start_y,end_y),dilation,self.max_y)
        cost = np.mean(cost_map[min_x:max_x + 1,:][:, min_y:max_y + 1])
        if math.isnan(cost):
            cost = 500
            log.error(f"math.isnan(cost) min_x:{min_x},max_x:{max_x},min_y:{min_y},max_y:{max_y},cost_map:{cost_map[min_x:max_x + 1,:][:, min_y:max_y + 1]}")
        cost = round(cost)
        # sg_vector = np.array([goal_node.x - start_node.x, goal_node.y - start_node.y])
        # se_vector = np.array([end_node.x - start_node.x, end_node.y - start_node.y])
        # dot_product = np.dot(sg_vector, se_vector)
        # norm1 = np.linalg.norm(sg_vector)
        # norm2 = np.linalg.norm(se_vector)
        # cos_theta = dot_product / (norm1 * norm2)

        angle_diff = max(start_node.angle,end_node.angle) - min(start_node.angle,end_node.angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        start_distance = pow((start_node.x - goal_node.x), 2) + pow((start_node.y - goal_node.y), 2)
        end_distance = pow((end_node.x - goal_node.x), 2) + pow((end_node.y - goal_node.y), 2)
        angle_cost = angle_diff // 15
        if start_distance <= 8:
            angle_cost = angle_cost // 12
        # cost = cost + (1 - cos_theta) + angle_cost
        distance_cost = end_distance / start_distance
        cost = cost + angle_cost + distance_cost
        return cost


    def calc_final_path_and_actions(self, goal_node, closed_set):
        # generate final course
        x = self.calc_grid_position(goal_node.x, self.min_x)
        y = self.calc_grid_position(goal_node.y, self.min_y)
        points = [(x, y)]
        actions = []
        last_node = goal_node
        # index = 0
        # print(f"====[{index}][{goal_node.angle}]")
        while last_node.parent_index != -1:
            # index = index + 1
            current_node = closed_set[last_node.parent_index]
            # print(f"====[{index}][{current_node.angle}]")
            x = self.calc_grid_position(current_node.x, self.min_x)
            y = self.calc_grid_position(current_node.y, self.min_y)
            points.append((x, y))
            actions.append(self.Action.forward.value)
            angle_diff = max(current_node.angle,last_node.angle) - min(current_node.angle,last_node.angle)
            if angle_diff == 0:
                pass
            elif angle_diff <= 180:
                if last_node.angle > current_node.angle:
                    actions.extend([self.Action.turn_left.value for  _ in range(angle_diff // self.angle_unit)])
                else:
                    actions.extend([self.Action.turn_right.value for  _ in range(angle_diff // self.angle_unit)])
            else:
                if last_node.angle > current_node.angle:
                    actions.extend([self.Action.turn_right.value for  _ in range((360 - angle_diff) // self.angle_unit)])
                else:
                    actions.extend([self.Action.turn_left.value for  _ in range((360 - angle_diff) // self.angle_unit)])
            last_node = current_node
        # from begin to end
        points.reverse() 
        actions.reverse()

        return points, actions

    def planning(self, sx, sy, gx, gy, obs_map, yaw, min_final_meter=6) -> tuple[list[tuple[float, float]], bool]:
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        # 照片里的角度跟实际角度相反
        if yaw > math.pi:
            yaw = yaw - math.pi
        else:
            yaw = yaw + math.pi

        angle = round(yaw * (180 / math.pi))
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1, angle)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1, 0)

        if obs_map[goal_node.x, goal_node.y] == 255:
            log.warning("Goal is in the obstacle.")
            return [], [], False

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node
        motions = self.get_motions(yaw)
        self.obstacle_map = obs_map
        cost_map = np.where(obs_map == 0, 240, obs_map)
        cost_map = np.where(cost_map == 2, 0, cost_map)

        step = 0
        while step < self.max_step:
            step += 1
            if len(open_set) == 0:
                log.info("Path Planning failed! Open set is empty..")
                break

            c_id = min(open_set,key=lambda o: open_set[o].cost)
            current = open_set[c_id]

            to_final_dis = self.calc_heuristic(current, goal_node)
            if to_final_dis <= min_final_meter:
                # print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.angle = current.angle
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for _, motion in enumerate(motions):
                next_x = current.x + motion[0]
                next_y = current.y + motion[1]
                next_cost = motion[2]
                next_angle = motion[3]
                next_node = self.Node(
                    next_x,
                    next_y,
                    -1,
                    c_id,
                    next_angle)
                obs_cost = self.get_cost(cost_map,current,next_node,goal_node)
                next_node.cost = current.cost + next_cost + obs_cost
                n_id = self.calc_grid_index(next_node)
                # If the node is not safe, do nothing
                if not self.verify_node(next_node):
                    continue
                if n_id in closed_set:
                    continue
                if n_id not in open_set:
                    open_set[n_id] = next_node  # discovered a new node
                else:
                    if open_set[n_id].cost > next_node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = next_node
        find_flag = True
        if step == self.max_step:
            log.info("Cannot find path. Return the path to the nearest node")
            goal_node = current
            find_flag = False
        actions = []
        points, actions = self.calc_final_path_and_actions(goal_node, closed_set)
        
        return points, actions, find_flag

class SampleDataCache:
    def __init__(self, lmdb_path, path_id, instruction):
        if not os.path.exists(lmdb_path):
            os.makedirs(lmdb_path)
        self.lmdb_path = lmdb_path
        self.path_id = path_id
        self.episode_total_data = []
        self.actions = []
        self.instruction = instruction

    def norm_depth(self, depth_info, min_depth=0, max_depth=10):
        depth_info[depth_info > max_depth] = max_depth
        depth_info = (depth_info - min_depth) / (max_depth - min_depth)
        return depth_info
    
    def collect_observation(self, obs ,step , process, camera_pose, robot_pose):
        from omni.isaac.core.utils.rotations import quat_to_euler_angles,euler_angles_to_quat
        task_name = 'vln_0'
        robot_name = 'h1_0'
        camera = 'pano_camera_0'
        episode_data = {
            'camera_info': {},
            'robot_info': {},
            'step': step,
            'progress': process
        }
        cur_obs = obs[task_name][robot_name][camera]
        pos, quat = camera_pose[0], camera_pose[1]
        _,_, yaw = quat_to_euler_angles(quat)
        rgb_info = cur_obs['rgba'][..., :3]
        depth_info = self.norm_depth(cur_obs['depth'])
        episode_data['camera_info'][camera] = {
            'rgb': rgb_info,
            'depth': depth_info,
            'position': pos.tolist(),
            'orientation': quat.tolist(),
            'yaw': yaw
        }
        pos, quat = robot_pose[0], robot_pose[1]
        _,_, yaw = quat_to_euler_angles(quat)
        episode_data['robot_info'] = {
            "position": pos.tolist(),
            "orientation": quat.tolist(),
            "yaw": yaw
        }
        self.episode_total_data.append(episode_data)

    def collect_action(self, action):
        self.actions.append(action)

    def merge_data(self, episode_datas ,actions):
        camera_info_dict = {}
        robot_info_list = {
            "position": [],
            "orientation": [],
            "yaw": [],
        }
        progress_list = []
        step_list = []

        for episode_data in episode_datas:
            for camera, info in episode_data['camera_info'].items():
                if camera not in camera_info_dict:
                    camera_info_dict[camera] = {
                        "rgb": [],
                        "depth": [],
                        "position": [],
                        "orientation": [],
                        "yaw": [],
                    }
                
                camera_info_dict[camera]["rgb"].append(info["rgb"])
                camera_info_dict[camera]["depth"].append(info["depth"])
                camera_info_dict[camera]["position"].append(info["position"])
                camera_info_dict[camera]["orientation"].append(info["orientation"])
                camera_info_dict[camera]["yaw"].append(info["yaw"])

            robot_info_list["position"].append(episode_data["robot_info"]["position"])
            robot_info_list["orientation"].append(episode_data["robot_info"]["orientation"])
            robot_info_list["yaw"].append(episode_data["robot_info"]["yaw"])
            
            step_list.append(episode_data["step"])
            progress_list.append(episode_data["progress"])

        for camera, info in camera_info_dict.items():
            for key, values in info.items():
                camera_info_dict[camera][key] = np.array(values)

        for key, values in robot_info_list.items():
            robot_info_list[key] = np.array(values)
        
        collate_data = {
            'camera_info': camera_info_dict,
            'robot_info': robot_info_list,
            'progress': np.array(progress_list),
            'step': np.array(step_list),
            'action': actions,
        }
        
        return collate_data

    def save_data(self, result):
        finish_flag = result
        if result != 'success':
            finish_flag = 'fail'
        lmdb_file = os.path.join(self.lmdb_path, "sample_data.lmdb")
        database = lmdb.open(lmdb_file, map_size=1 * 1024 * 1024 * 1024 * 1024, max_dbs=0)
        with database.begin(write=True) as txn:
            key = f"{self.path_id}".encode()
            episode_datas = self.merge_data(self.episode_total_data, self.actions)
            data_to_store = {
                'episode_data': episode_datas,
                'finish_status': finish_flag,
                'fail_reason': result,
                'instruction': self.instruction,
            }
            serialized_data = msgpack_numpy.packb(data_to_store, use_bin_type=True)
            txn.put(key, serialized_data)
        database.close()
        self.episode_total_data = []
        self.actions = []

def pixel_to_world(pixel_pose,camera_pose,aperture,width,height):
    cx, cy = camera_pose[0]*10/aperture*width, -camera_pose[1]*10/aperture*height
    px = height - pixel_pose[0] + cx - height/2
    py = pixel_pose[1] + cy - width/2

    world_x = px/10/height*aperture
    world_y = -py/10/width*aperture

    return [world_x, world_y]

def world_to_pixel(world_pose, camera_pose,aperture,width,height):
    cx, cy = camera_pose[0]*10/aperture*width, -camera_pose[1]*10/aperture*height

    X, Y = world_pose[0]*10/aperture*width, -world_pose[1]*10/aperture*height
    pixel_x = width - (X - cx + width/2)
    pixel_y = Y - cy + height/2

    return [pixel_x, pixel_y]

def vis_nav_path(start_pixel, goal_pixel, points, occupancy_map, img_save_path='path_planning.jpg'):
    cmap = mcolors.ListedColormap(['white', 'green', 'gray', 'black'])
    bounds = [0, 1, 3, 254, 256]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=(10, 10))
    # plt.imshow(occupancy_map, cmap='binary', origin='lower')
    plt.imshow(occupancy_map, cmap=cmap, norm=norm, origin='upper')

    # Plot start and goal points
    plt.plot(start_pixel[1], start_pixel[0], 'ro', markersize=6, label='Start')
    plt.plot(goal_pixel[1], goal_pixel[0], 'bo', markersize=6, label='Goal')

    # Plot the path
    if len(points) > 0:
        path = np.array(points)
        plt.plot(path[:, 1], path[:, 0], 'xb-', linewidth=1, markersize=5, label='Path')

    # Customize the plot
    plt.title('Path planning')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.colorbar(label='Occupancy (0: Free, 1: Occupied)')

    # Save the plot
    plt.savefig(img_save_path, pad_inches=0, bbox_inches='tight', dpi=100)
    log.info(f"Saved path planning visualization to {img_save_path}")
    plt.close()

def describe_action(action):
    if action == 1:
        return "向前走0.25米"
    elif action == 2:
        return "左转15°"
    elif action == 3:
        return "右转15°"
    elif action == 4:
        return "停在原地"

def print_actions(actions):
    for index,action in enumerate(actions):
        print(f"[{index}]==>{describe_action(action)}")

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

def get_real_points(yaw , points,actions,camera_pose,aperture,width,height):
    point_index = 0
    current_real_point = pixel_to_world(points[point_index],camera_pose,aperture,width,height)
    current_yaw = yaw
    real_points = []
    for action in actions:
        if action == 2:#left
            current_yaw = current_yaw + (math.pi / 12)
            real_points.append(current_yaw)
        elif action == 3:#right
            current_yaw = current_yaw - (math.pi / 12)
            real_points.append(current_yaw)
        else:#forward
            point_index = point_index + 1
            current_real_point = pixel_to_world(points[point_index],camera_pose,aperture,width,height)
            real_points.append(current_real_point)
    return real_points

def plan_and_get_actions_discrete(goal, occupancy_map, topdown_map, robot_position, robot_rotation, 
                         offset, aperture, width, height, path_planner):
    from omni.isaac.core.utils.rotations import quat_to_euler_angles
    
    camera_pose = occupancy_map.topdown_camera.get_world_pose()[0] - offset
    freemap, _ = occupancy_map.get_global_free_map(robot_pos=robot_position, robot_height=1.55, update_camera_pose=False, verbose=False)
    topdown_map.update_map(freemap, camera_pose, verbose=False, env_idx=0, update_map=True)
    _, _, yaw = quat_to_euler_angles(robot_rotation)
    occupancy_map, _ = topdown_map.get_map(robot_position, return_camera_pose=True)
    start_pixel = world_to_pixel(robot_position,camera_pose,aperture,width,height)
    goal_pixel = world_to_pixel(goal,camera_pose,aperture,width,height)
    # torch.save(occupancy_map,'/ssd/zhaohui/workspace/w61_grutopia_1118/test/zhaohui/occupancy_map.pt')
    start_time = time.time()
    points,actions, find_flag = path_planner.planning(
        start_pixel[0], 
        start_pixel[1],
        goal_pixel[0], 
        goal_pixel[1],
        obs_map=occupancy_map,
        yaw = yaw,
    )
    end_time = time.time()
    log.info(f"path_planning 耗时：{(end_time - start_time)} s")
    if not find_flag:
        return [], [], find_flag
    real_points = get_real_points(yaw,points,actions,camera_pose,aperture,width,height)
    # file_name = f"path_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg" 
    # vis_nav_path(
    #     start_pixel, 
    #     goal_pixel, 
    #     points, 
    #     occupancy_map, 
    #     img_save_path=os.path.join('/ssd/zhaohui/workspace/w61_grutopia_1206/test/zhaohui/', file_name)
    # )
    return actions, real_points, find_flag

def plan_and_get_actions_continuous(goal, occupancy_map, topdown_map, robot_position, robot_rotation, 
                         offset, aperture, width, height, path_planner):  
    camera_pose = occupancy_map.topdown_camera.get_world_pose()[0] - offset
    freemap, _ = occupancy_map.get_global_free_map(robot_pos=robot_position, robot_height=1.55, update_camera_pose=False, verbose=False)
    topdown_map.update_map(freemap, camera_pose, verbose=False, env_idx=0, update_map=True)
    occupancy_map, _ = topdown_map.get_map(robot_position, return_camera_pose=True)
    start_pixel = world_to_pixel(robot_position,camera_pose,aperture,width,height)
    goal_pixel = world_to_pixel(goal,camera_pose,aperture,width,height)
    # torch.save(occupancy_map,'/ssd/zhaohui/workspace/w61_grutopia_1118/test/zhaohui/occupancy_map.pt')
    start_time = time.time()
    points, find_flag = path_planner.planning(
        start_pixel[0], 
        start_pixel[1],
        goal_pixel[0], 
        goal_pixel[1],
        obs_map=occupancy_map,
        min_final_meter=1,
        vis_path=False
    )
    end_time = time.time()
    log.info(f"path_planning 耗时：{(end_time - start_time)} s")
    # file_name = f"path_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg" 
    # vis_nav_path(
    #     start_pixel, 
    #     goal_pixel, 
    #     points, 
    #     occupancy_map, 
    #     img_save_path=os.path.join('/ssd/zhaohui/workspace/w61_grutopia_1206/test/zhaohui/', file_name)
    # )
    if find_flag:
        transfer_paths = []
        for node in points:
            world_coords = pixel_to_world(node,camera_pose,aperture,width,height)
            transfer_paths.append([world_coords[0], world_coords[1], robot_position[2]])
    else:
        transfer_paths = None
    if transfer_paths is not None and len(transfer_paths)>1:
        transfer_paths.pop(0)
    return transfer_paths


def get_env_actions(current_action, robot_position, robot_rotation):
    from omni.isaac.core.utils.rotations import quat_to_euler_angles,euler_angles_to_quat
    roll, pitch, yaw = quat_to_euler_angles(robot_rotation)
    if current_action == 1: # forward
        dx = 0.25 * math.cos(yaw)
        dy = 0.25 * math.sin(yaw)
        goal = np.array([robot_position[0] + dx, robot_position[1] + dy, robot_position[2]])
        env_actions =[{'h1':{'move_to_point': [goal]}}]
    elif current_action == 2: #turn_left
        euler_angles = np.array([roll, pitch, yaw + math.pi / 12])
        env_actions =[{'h1':{'rotate': [euler_angles_to_quat(euler_angles)]}}]
    elif current_action == 3: #turn_right
        euler_angles = np.array([roll, pitch, yaw - math.pi / 12])
        env_actions =[{'h1':{'rotate': [euler_angles_to_quat(euler_angles)]}}]
    else: #stay_still
        env_actions =[{'h1':{'stand_still': [[]]}}]
    return env_actions

def get_env_actions_aggregation(action_aggregation, robot_position, robot_rotation):
    action_count = len(action_aggregation)
    action_type = action_aggregation[0]
    from omni.isaac.core.utils.rotations import quat_to_euler_angles,euler_angles_to_quat
    roll, pitch, yaw = quat_to_euler_angles(robot_rotation)
    if action_type == 1: # forward
        dx = 0.25 * action_count * math.cos(yaw)
        dy = 0.25 * action_count * math.sin(yaw)
        goal = np.array([robot_position[0] + dx, robot_position[1] + dy, robot_position[2]])
        env_actions =[{'h1':{'move_to_point': [goal]}}]
    elif action_type == 2: #turn_left
        euler_angles = np.array([roll, pitch, yaw + (math.pi / 12) * action_count])
        env_actions =[{'h1':{'rotate': [euler_angles_to_quat(euler_angles)]}}]
    elif action_type == 3: #turn_right
        euler_angles = np.array([roll, pitch, yaw - (math.pi / 12) * action_count])
        env_actions =[{'h1':{'rotate': [euler_angles_to_quat(euler_angles)]}}]
    else: #stay_still
        env_actions =[{'h1':{'stand_still': [[]]}}]
    return env_actions

def is_one_action_finished(current_action, obs):
    if len(obs) <= 0:
        return False
    action_name = 'stand_still'
    if current_action == 1:
        action_name = 'move_to_point'
    elif current_action == 2 or current_action == 3:
        action_name = 'rotate'
    return obs['vln_0']['h1_0'][action_name]['finished']

def is_action_finished(action_name, obs):
    if len(obs) <= 0:
        return False
    return obs['vln_0']['h1_0'][action_name]['finished']

def is_halfway_sample_needed(
    action_aggregation,
    action_start_position,
    action_start_rotation,
    robot_position,
    robot_rotation,
    halfway_sample_count,
):
    from omni.isaac.core.utils.rotations import quat_to_euler_angles
    halfway_sample = False
    action_type = action_aggregation[0]
    if halfway_sample_count >= len(action_aggregation) - 1:
        return False
    if action_type == 1: # forward
        x1,x2 = action_start_position[0],robot_position[0]
        y1,y2 = action_start_position[1],robot_position[1]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if distance >= 0.25 * (halfway_sample_count + 1):
            halfway_sample = True
    elif action_type == 2: #turn_left
        _, _, start_yaw = quat_to_euler_angles(action_start_rotation)
        _, _, robot_yaw = quat_to_euler_angles(robot_rotation)
        if (robot_yaw - start_yaw) >= (math.pi / 12) * (halfway_sample_count + 1):
            halfway_sample = True
    elif action_type == 3: #turn_right
        _, _, start_yaw = quat_to_euler_angles(action_start_rotation)
        _, _, robot_yaw = quat_to_euler_angles(robot_rotation)
        if (start_yaw - robot_yaw) >= (math.pi / 12) * (halfway_sample_count + 1):
            halfway_sample = True
    return halfway_sample

def need_sample(lmdb_path, path_id, retry_list=[]):
    if not os.path.exists(lmdb_path):
        return True
    env = lmdb.open(lmdb_path, readonly=True, lock=False) 
    with env.begin() as txn:
        key = f"{path_id}".encode()
        value = txn.get(key)
        if value is None:
            return True
        value = msgpack_numpy.unpackb(value)
        if value['finish_status'] == 'success':
            return 'success' in retry_list
        else:
            return value['fail_reason'] in retry_list

def transform_rotation_z_90degrees(rotation):
    ''' 沿着z轴旋转90度
    '''
    z_rot_90 = [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]  # 90 degrees = pi/2 radians
    w1, x1, y1, z1 = rotation
    w2, x2, y2, z2 = z_rot_90
    revised_rotation = [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ]
    return revised_rotation

def get_sample_data_by_scan(base_data_dir, lmdb_path, split, scan, retry_list=[]):
    class Datasets:
        def __init__(self):
            self.base_data_dir = base_data_dir
    class Args:
        def __init__(self):
            self.datasets = Datasets()
    scan_map, _ = load_gather_data(Args(), split, filter_same_trajectory=True, filter_stairs=True)
    sample_path_list = []
    robot_offset = np.array([0.   , 0.   , 0.975])
    for scan_id, path_list in scan_map.items():
        if scan_id != scan:
            continue
        for one_path in path_list:
            one_path["start_position"] += robot_offset
            for i, _ in enumerate(one_path["reference_path"]):
                one_path["reference_path"][i] += robot_offset
            one_path['start_rotation'] = transform_rotation_z_90degrees(one_path['start_rotation'])
            path_id = one_path['trajectory_id']
            if fall_by_problematic_data(path_id):
                amend_offset = get_amend_offset(path_id)
                one_path['start_position'][0] = one_path['start_position'][0] + amend_offset[0]
                one_path['start_position'][1] = one_path['start_position'][1] + amend_offset[1]
                one_path['start_position'][2] = one_path['start_position'][2] + amend_offset[2]
                one_path['reference_path'][0][0] = one_path['reference_path'][0][0] + amend_offset[0]
                one_path['reference_path'][0][1] = one_path['reference_path'][0][1] + amend_offset[1]
                one_path['reference_path'][0][2] = one_path['reference_path'][0][2] + amend_offset[2]
            sample_path_list.append(one_path)
    # 数据过滤
    filtered_sample_path_list = []
    for one_path in sample_path_list:
        path_id = one_path['trajectory_id']
        if need_sample(lmdb_path, path_id, retry_list):
            filtered_sample_path_list.append(one_path)
    return filtered_sample_path_list