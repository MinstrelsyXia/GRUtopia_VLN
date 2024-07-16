import numpy as np
np.random.seed(2024)
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import Polygon
from scipy.ndimage import distance_transform_edt

from collections import deque
######################## find nearest unoccupied ###############################################
def is_valid(occupancy_map, x, y, r):
    return np.all(occupancy_map[max(y-r, 0): min(y+r, occupancy_map.shape[0]), max(x-r, 0): min(x+r, occupancy_map.shape[1])])==0

def find_nearest_free_space(occupancy_map, current_point:tuple, r: int):
    height, width = occupancy_map.shape
    queue = deque([current_point]) # (y, x)
    visited = set([current_point])
    
    while queue:
        x, y = queue.popleft()
        
        if is_valid(occupancy_map, x, y, r):
            return Node(x, y)  # Found a valid free space
        
        # Add adjacent points (up, down, left, right) to the queue
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (ny, nx) not in visited and occupancy_map[ny, nx] == 1:
                visited.add((ny, nx))
                queue.append((ny, nx))
    
    return None

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None

class QuadTreeNode:
    def __init__(self, x, y, width, height, map_data, depth=0, max_depth=10, threshold=1):
        self.initialize(x, y, width, height, map_data, depth, max_depth, threshold)

    def initialize(self, x, y, width, height, map_data, depth, max_depth, threshold):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.children = []
        self.depth = depth
        self.occupied = False
        self.max_depth = max_depth
        self.threshold = threshold

        self.split(map_data)

    def split(self, map_data):
        """
        Split the current node into four children if the stopping condition is not met
        and the current region is not entirely occupied or free.
        """
        # Stopping condition: node is at max depth or the area is entirely occupied/free
        stop_condition = self.depth >= self.max_depth
        area_data = map_data[self.y:self.y+self.height, self.x:self.x+self.width]
        free_area = np.count_nonzero(area_data == 1)  # Assuming 1 represents free space
        total_area = self.width * self.height
        free_ratio = free_area / total_area
        
        if stop_condition or free_ratio == 1.0 or free_ratio == 0.0:
            # If stopping condition is met or the region is entirely free or occupied
            # Set node as occupied if free area doesn't reach the threshold, otherwise, it's free
            self.occupied = free_ratio < self.threshold
        else:
            # Node should be split further
            half_width = self.width // 2
            half_height = self.height // 2
            if half_width==0 and half_height!=0:
                self.children = [
                QuadTreeNode(self.x + half_width, self.y, self.width - half_width, half_height, map_data, self.depth + 1, self.max_depth, self.threshold),
                QuadTreeNode(self.x + half_width, self.y + half_height, self.width - half_width, self.height - half_height, map_data, self.depth + 1, self.max_depth, self.threshold)
                ]
            elif half_width!=0 and half_height==0:
                self.children = [
                QuadTreeNode(self.x, self.y + half_height, half_width, self.height - half_height, map_data, self.depth + 1, self.max_depth, self.threshold),
                QuadTreeNode(self.x + half_width, self.y + half_height, self.width - half_width, self.height - half_height, map_data, self.depth + 1, self.max_depth, self.threshold)
                ]
            elif half_width==0 and half_height==0:
                self.occupied = free_ratio < self.threshold
            else:
                self.children = [
                QuadTreeNode(self.x, self.y, half_width, half_height, map_data, self.depth + 1, self.max_depth, self.threshold),
                QuadTreeNode(self.x + half_width, self.y, self.width - half_width, half_height, map_data, self.depth + 1, self.max_depth, self.threshold),
                QuadTreeNode(self.x, self.y + half_height, half_width, self.height - half_height, map_data, self.depth + 1, self.max_depth, self.threshold),
                QuadTreeNode(self.x + half_width, self.y + half_height, self.width - half_width, self.height - half_height, map_data, self.depth + 1, self.max_depth, self.threshold)
                ]
    ######################## update the map ###############################################
    def update(self, map_data, x, y, width, height):
        """
        Update the quadtree with new map data for a specified region.
        """
        # Check if the update region intersects with this node
        if (x + width < self.x or x > self.x + self.width or
                y + height < self.y or y > self.y + self.height):
            # No intersection with the region to update
            return

        # If this node is a leaf node or at max depth, reinitialize it based on the new map data
        if not self.children or self.depth == self.max_depth:
            self.initialize(self.x, self.y, self.width, self.height, map_data, self.depth, self.max_depth, self.threshold)
        else:
            # Otherwise, recursively update all children
            for child in self.children:
                child.update(map_data, x, y, width, height)
        self.merge()

    def can_merge(self):
        """
        Check if all children have the same occupied state and are leaf nodes,
        which allows them to be merged into this node.
        """
        if not self.children:
            return False
        
        first_child_state = self.children[0].occupied
        for child in self.children:
            # If any child has children of its own (is not a leaf)
            # or has a different occupied state, we cannot merge
            if child.children or child.occupied != first_child_state:
                return False
        return True

    def merge(self):
        """
        Merge all child nodes into this node if they have the same occupied state
        and are leaf nodes.
        """
        if self.can_merge():
            # Set this node's occupied state to that of its children
            self.occupied = self.children[0].occupied
            # Remove all children since they are now merged into this node
            self.children = []
    ######################## query collision ###############################################
    def query(self, vertices):
        """
        Determine if the object defined by 'vertices' is in a free space.
        
        Parameters:
        vertices -- A list of clockwise-ordered vertices defining the object.
        
        Returns:
        collision -- A boolean indicating if there is a collision (False if free).
        bounding_box -- A tuple (y, x, width, height) of the minimal area covering the object.
        """
        # Convert vertices list to a Path for easy checking
        path = mpath.Path(vertices)
        
        # Calculate bounding box of the vertices
        min_y, min_x = np.min(vertices, axis=0)
        max_y, max_x = np.max(vertices, axis=0)
        height = max_y - min_y
        width = max_x - min_x
        
        # Check for collision recursively starting from the root
        collision = self.check_collision(path)
        
        return not collision, (min_x, min_y, width, height)

    def check_collision(self, path):
        """
        Recursively check if the given path collides with any occupied node.
        This optimized version first checks if the path intersects the current node's area.
        If there's no intersection, it returns False without further checking its children.
        """
        # Define the corners of the current node's area
        corners = [(self.y, self.x), 
                (self.y, self.x + self.width), 
                (self.y + self.height, self.x + self.width), 
                (self.y + self.height, self.x),
                (self.y, self.x)]
        node_path = mpath.Path(corners)
        
        # Check if the given path intersects with the current node's area
        # If there's no intersection, no need to check this node or its children
        if not path.intersects_path(node_path):
            return False
        
        # If this is a leaf node
        if not self.children:
            # Only return True if the leaf node is occupied
            return self.occupied
        else:
            # If it's not a leaf node, recursively check all children nodes
            for child in self.children:
                if child.check_collision(path):
                    return True
            return False

    ######################## visualization ###############################################
    def visualize_quad_tree(self, ax, level=0):
        if self.children:
            for child in self.children:
                child.visualize_quad_tree(ax, level + 1)
        else:
            rect = plt.Rectangle((self.x, self.y), self.width, self.height, 
                                edgecolor='black', lw=0.1, facecolor='white' if self.occupied else 'none')
            ax.add_patch(rect)

    def plot_quad_tree(self, map_data, fig_size=(10,12), fontsize=20, if_save=False, if_show=False):
        fig, axs = plt.subplots(1, 2, figsize = fig_size)
        axs[0].imshow(map_data, cmap='gray', origin='lower', alpha=0.5)
        axs[0].set_title('occupancy',fontsize = fontsize)
        self.visualize_quad_tree(axs[1])
        axs[1].imshow(map_data, cmap='gray', origin='lower', alpha=0.5)
        axs[1].set_title('quadtree', fontsize = fontsize)
        if if_save:
            plt.savefig("split.png")
            plt.close()
        if if_show:
            plt.show()

    def draw_polygon_on_map(self, map_data, vertices):
        """
        Draw a polygon defined by 'vertices' on a 2D occupancy map.

        Parameters:
        map_data -- 2D occupancy grid (numpy array) where 0 represents free space and 1 represents obstacles.
        vertices -- List of clockwise-ordered vertices defining the polygon, each vertex as a tuple (x, y).
        """
        fig, ax = plt.subplots()
        ax.imshow(map_data, cmap='gray', origin='lower', extent=[0, map_data.shape[1], 0, map_data.shape[0]])

        # Create a Polygon from the vertices
        vertices = [(x, y) for y, x in vertices]
        polygon = Polygon(vertices, closed=True, edgecolor='r', facecolor='none', linewidth=1)
        ax.add_patch(polygon)

        # Set plot limits and labels
        ax.set_xlim(0, map_data.shape[1])
        ax.set_ylim(0, map_data.shape[0])
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        plt.savefig("collision")
        plt.close()
        # plt.show()

class PathPlanning:
    def __init__(self, occupy_map: QuadTreeNode, origin_map, agent_radius=10, last_scope = 100, goal_sampling_rate=0.2, extend_length = 5, max_iter=2000, consider_range = 30):
        self.occupy_map = occupy_map
        self.origin_map = origin_map
        self.distance_map = distance_transform_edt(origin_map)
        self.agent_radius = agent_radius
        self.last_scope = last_scope
        self.nodes = []
        self.goal_sampling_rate = goal_sampling_rate
        self.extend_length = extend_length
        self.max_iter = max_iter
        self.consider_range = consider_range

    def get_random_node(self, goal):
        if np.random.rand() < self.goal_sampling_rate:
            # With a certain probability, choose the goal node
            return goal
        else:
            probability_distribution = self.distance_map ** 2
            normalized_distribution = probability_distribution / np.sum(probability_distribution)
            flat_index = np.random.choice(len(probability_distribution.ravel()), p=normalized_distribution.ravel())
            rows, cols = self.distance_map.shape
            row_index = flat_index // cols
            col_index = flat_index % cols
            return Node(col_index, row_index)

    def nearest_node(self, nodes, random_node):
        return min(nodes, key=lambda node: np.hypot(node.x - random_node.x, node.y - random_node.y))

    def steer(self, from_node, to_node):
        if from_node == to_node:
            return from_node
        dist = np.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
        if dist > self.extend_length:
            theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
            new_node = Node(from_node.x + self.extend_length * np.cos(theta), from_node.y + self.extend_length * np.sin(theta))
            new_node.cost = from_node.cost + self.extend_length
            new_node.parent = from_node
            return new_node
        else:
            to_node.cost = from_node.cost + dist
            to_node.parent = from_node
            return to_node

    def collision_free(self, from_node, to_node, mode = 'navigation'):
        """
        Check if the path from from_node to to_node collides with any obstacles in the environment.
        
        Parameters:
        from_node -- The starting Node object with coordinates (x, y)
        to_node -- The ending Node object with coordinates (x, y)
        
        Returns:
        True if no collision occurs, otherwise False
        """
        if from_node == to_node or (from_node.x==to_node.x and from_node.y==to_node.y):
            if mode == 'rrt':
                return False
            if mode == 'navigation':
                return True
        # Calculate direction vector from from_node to to_node
        direction_vector = np.array([to_node.y - from_node.y, to_node.x - from_node.x])
        
        # Normalize the direction vector
        norm_direction_vector = direction_vector / np.linalg.norm(direction_vector)
        
        # Extend to_node by agent_radius in the direction of to_node
        extended_to_node = np.array([to_node.y, to_node.x]) + norm_direction_vector * self.agent_radius
        
        # Calculate perpendicular (normal) vector to the direction
        norm_vector = np.array([-norm_direction_vector[1], norm_direction_vector[0]])
        
        # Calculate the four vertices of the parallelogram, ordered clockwise
        vertices = [
            (from_node.y + norm_vector[0] * self.agent_radius, from_node.x + norm_vector[1] * self.agent_radius),
            (extended_to_node[0] + norm_vector[0] * self.agent_radius, extended_to_node[1] + norm_vector[1] * self.agent_radius),
            (extended_to_node[0] - norm_vector[0] * self.agent_radius, extended_to_node[1] - norm_vector[1] * self.agent_radius),
            (from_node.y - norm_vector[0] * self.agent_radius, from_node.x - norm_vector[1] * self.agent_radius),
            (from_node.y + norm_vector[0] * self.agent_radius, from_node.x + norm_vector[1] * self.agent_radius)
        ]

        # Step 2: Use quadtree for coarse collision detection
        # self.occupy_map.draw_polygon_on_map(self.origin_map, vertices)
        collision_free, bounding_box = self.occupy_map.query(vertices)
        # print(collision_free)
        if collision_free:
            # Quadtree query indicates no collision will occur
            return True
        return False

    def find_near_nodes(self, new_node):
        return [node for node in self.nodes if np.hypot(node.x - new_node.x, node.y - new_node.y) < self.consider_range]

    def choose_parent(self, near_nodes, new_node):
        """
        Chooses the best parent for new_node from near_nodes based on cost and collision-free path.
        
        Parameters:
        near_nodes -- List of nearby nodes to consider as potential parents
        new_node -- The new node to which a parent is being assigned
        
        Returns:
        The node chosen as the best parent. If no suitable parent is found, returns None.
        """
        if new_node.x == 104 and new_node.y ==532 and new_node.parent==new_node:
            print('')
        if not near_nodes:
            return new_node.parent
        
        costs = []
        for node in near_nodes:
            if node.x == 104 and node.y ==532 and node==new_node:
                print('')
            if self.collision_free(node, new_node, mode='rrt'):
                # Calculate the cost to reach new_node from the current near node
                costs.append(node.cost + np.hypot(node.x - new_node.x, node.y - new_node.y))
            else:
                costs.append(float('inf'))
        
        min_cost = min(costs)
        
        # If all paths from near_nodes to new_node have collisions (cost is infinity)
        if min_cost == float('inf'):
            return None
        
        min_index = costs.index(min_cost)
        new_node.cost = min_cost
        best_parent_node = near_nodes[min_index]
        
        new_node.parent = best_parent_node
        if best_parent_node.x == 104 and best_parent_node.y ==532 and best_parent_node == new_node:
            print('')
        return best_parent_node

    def reconnect(self, near_nodes, new_node):
        """
        Reconnects nearby nodes to new_node if it provides a cheaper path.
        
        Parameters:
        near_nodes -- List of nearby nodes to consider for reconnection
        new_node -- The new node that may become a new parent to some near_nodes
        
        This method updates the parent of near_nodes if a path through new_node is cheaper.
        """
        for node in near_nodes:
            if self.collision_free(new_node, node, mode='rrt') and new_node.cost + np.hypot(new_node.x - node.x, new_node.y - node.y) < node.cost:
                node.parent = new_node
                node.cost = new_node.cost + np.hypot(new_node.x - node.x, new_node.y - node.y)

    def rrt_star(self, start, goal):
        self.nodes = [start]
        for _ in range(self.max_iter):
            random_node = self.get_random_node(goal)
            nearest = self.nearest_node(self.nodes, random_node)
            new_node = self.steer(nearest, random_node)
            if self.collision_free(nearest, new_node, mode='rrt'):
                near_nodes = self.find_near_nodes(new_node)
                parent_node = self.choose_parent(near_nodes, new_node)
                if parent_node:
                    new_node.parent = parent_node
                    new_node.cost = parent_node.cost + np.hypot(new_node.x - parent_node.x, new_node.y - parent_node.y)
                    self.nodes.append(new_node)
                    self.reconnect(near_nodes, new_node)

                if np.hypot(new_node.x - goal.x, new_node.y - goal.y) < self.last_scope-1:
                    final_node = self.steer(new_node, goal)  # Attempt to connect directly to goal
                    if self.collision_free(new_node, final_node, mode='rrt'):  # Ensure path to goal is collision-free
                        return final_node, 0  # Path found to goal
                    else:
                        return new_node, 1  # # Path found to goal but goal is occupied
        # nodes = [node for node in self.nodes if self.nodes==self.nodes.parent]
        return min(self.nodes, key=lambda node: np.hypot(node.x - goal.x, node.y - goal.y)), 2  # Path not found to goal but return the nearest
    
    def visualize(self):
        """
        Visualizes the current state of the RRT* exploration.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(self.origin_map, cmap='gray_r', origin='lower', extent=[0, self.origin_map.shape[1], 0, self.origin_map.shape[0]])
        
        for node in self.nodes:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], 'r-', alpha=0.4)
        
        plt.plot(self.nodes[0].x, self.nodes[0].y, "xr")  # Start
        plt.grid(True)
        plt.axis("equal")
        plt.show()  # Pause to update the figure

    def plot_path(self, node, start, goal, name = "path"):
        """
        Plots the path from start to goal by backtracking from the goal node to the start node.
        
        Parameters:
        node -- The final node reached in the path (should be close or equal to the goal node)
        start -- The starting Node
        goal -- The goal Node
        """
        plt.ioff()
        fig, ax = plt.subplots()
        # Plot obstacles from the occupancy map
        ax.imshow(self.origin_map, cmap='gray_r', origin='lower', extent=[0, self.origin_map.shape[1], 0, self.origin_map.shape[0]])
        
        # Plot path
        path = []
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([start.x, start.y])  # Add start point to the path
        path.reverse()  # Reverse to start from the starting point
        px, py = zip(*path)  # Separate into x and y coordinates
        plt.plot(px, py, "-r")  # Plot path in red
        
        # Plot start and goal
        plt.plot(start.x, start.y, "xr", label="start")  # Start in red 'x'
        plt.plot(goal.x, goal.y, "xg", label="end")  # Goal in green 'x'
        
        ax.set_aspect('equal', adjustable='box')
        ax.axis("off")
        plt.legend()
        plt.savefig(name)
        plt.close()
        # plt.show()

if __name__ == "__main__":
    import yaml
    with open('llm_agent/agent_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    occupancy_map = np.load('map.npy')
    quad_tree_root = QuadTreeNode(0, 0, map_data = 1-(occupancy_map==0),**config['map_config']['quadtree_config'])
    quad_tree_root.plot_quad_tree(occupancy_map,fig_size=(14,8),if_save=True)
    path_planner = PathPlanning(quad_tree_root, 1-(occupancy_map==0), **config['planner_config']) # Navigation method
    start, goal = Node(90,195), Node(190,210)
    node, node_type= path_planner.rrt_star(start, goal)
    path_planner.plot_path(node, start, goal)