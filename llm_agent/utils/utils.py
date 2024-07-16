import io
import os
import spacy
import json
import yaml
import pickle
import json
from json import JSONEncoder
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

######################## evaluate ########################
class Evaluator:
    def __init__(self, agent_type):
        self.total_episodes = 0
        self.path_length = []
        self.navigation_error = []
        self.success = []
        self.spl = []
        self.type = agent_type
        if self.type == 'agent':
            self.dialogue_turn = []
            self.candidates_reduced = []

    def update(self, result_dict, task_info):
        self.total_episodes+=1
        self.path_length.append(result_dict['path_length'])
        navigation_error = np.linalg.norm(np.array(result_dict['path'][-1]['position'])-task_info['position'])
        self.navigation_error.append(navigation_error)
        result_dict['success'] = result_dict['success_view'] and (navigation_error<=3)
        self.success.append(result_dict['success'])
        self.spl.append(result_dict['success']*task_info['shortest_path_length']/max(result_dict['path_length'], task_info['shortest_path_length']))
        assert len(self.path_length) == self.total_episodes
        assert len(self.navigation_error) == self.total_episodes
        assert len(self.success) == self.total_episodes
        assert len(self.spl) == self.total_episodes
        if self.type == 'Dialogue':
            self.dialogue_turn.append(result_dict['dialogue_turn'])
            self.candidates_reduced.append(np.average(result_dict['candidates_reduced']) if len(result_dict['candidates_reduced']) > 0 else 0)
            assert len(self.dialogue_turn) == self.total_episodes
            assert len(self.candidates_reduced) == self.total_episodes
            return {
            'Success': self.success[-1],
            'Path Length': self.path_length[-1],
            'Navigation Error': self.navigation_error[-1],
            'Success weighted by Path Length (SPL)': self.spl[-1],
            'Dialogue Turn': self.dialogue_turn[-1],
            'Average Candidates Reduced this Turn': self.candidates_reduced[-1]
            }
        return {
            'Success': self.success[-1],
            'Path Length': self.path_length[-1],
            'Navigation Error': self.navigation_error[-1],
            'Success weighted by Path Length (SPL)': self.spl[-1]
            }

    def calculate_metrics(self):
        avg_path_length = np.average(self.path_length) if len(self.path_length) > 0 else 0
        avg_nav_error = np.average(self.navigation_error) if len(self.navigation_error) > 0 else 0
        sr = np.average(self.success) if len(self.success) > 0 else 0
        spl = np.average(self.spl) if len(self.spl) > 0 else 0
        if self.type == 'Dialogue':
            avg_dialogue_turns = np.average(self.dialogue_turn) if len(self.dialogue_turn) > 0 else 0
            avg_candidates_reduction = np.average(self.candidates_reduced) if len(self.candidates_reduced) > 0 else 0
            return {
            'Success Rate (SR)': sr,
            'Average Path Length': avg_path_length,
            'Average Navigation Error': avg_nav_error,
            'Success weighted by Path Length (SPL)': spl,
            'Average Dialogue Turns': avg_dialogue_turns,
            'Average Candidates Reduced per Turn': avg_candidates_reduction
        }
        
        return {
            'Success Rate (SR)': sr,
            'Average Path Length': avg_path_length,
            'Average Navigation Error': avg_nav_error,
            'Success weighted by Path Length (SPL)': spl
        }

    def report(self, save_path = None):
        metrics = self.calculate_metrics()
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as file:
                json.dump(metrics, file, ensure_ascii=False, indent=4)

    def calculate_path_length(self, history_positions):
        """
        Calculate the path length in 2D space, considering only x and y coordinates.

        Parameters:
        history_positions (list of lists): Each sublist contains x, y, z coordinates.

        Returns:
        float: Total path length.
        """
        if len(history_positions) < 2:
            return 0.0  # If there are fewer than 2 positions, the path length is 0

        # Convert history positions to a numpy array for easier manipulation
        positions = np.array(history_positions)
        
        # Extract only x and y coordinates
        xy_positions = positions[:, :2]

        # Calculate the Euclidean distances between consecutive points
        distances = np.sqrt(np.sum(np.diff(xy_positions, axis=0) ** 2, axis=1))

        # Sum up all the distances to get the total path length
        total_length = np.sum(distances)
        return total_length

def get_pixel(occupancy, coords):
    xs = occupancy[1:, 0]
    ys = occupancy[0, 1:]
    length = len(xs)
    sorter = [length-1-i for i in range(length)]
    x_idx = np.searchsorted(xs, coords[:,1], side='left', sorter=sorter)
    x_idx = length - x_idx - 1
    y_idx = np.searchsorted(ys, coords[:,0], side='left')
    return x_idx, y_idx

def calculate_shortest(grid, start, end):    
    scale = min(round(abs(grid[0,2]-grid[0,1]),2), round(abs(grid[2,0]-grid[1,0]) ,2))
    xs, ys = get_pixel(grid, np.array([start,end]))
    grid = grid[1:,1:]==1
    start = (ys[0], xs[0])
    end = (ys[1], xs[1])
    flag = True
    i = 1
    grid[max(0, end[0]-5*i):min(grid.shape[0], end[0]+5*i), max(0, end[1]-5*i):min(grid.shape[1], end[1]+5*i)] = 1
    while flag:
        grid = 1 - grid
        def neighbors(pos):
            x, y = pos
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
                    yield nx, ny

        heap = [(0, start)]
        visited = set()
        distance = {start: 0}
        prev = {start: None}

        while heap:
            dist, current = heapq.heappop(heap)
            if current == end:
                break

            if current in visited:
                continue

            visited.add(current)

            for neighbor in neighbors(current):
                new_dist = dist + 1
                if neighbor not in distance or new_dist < distance[neighbor]:
                    distance[neighbor] = new_dist
                    prev[neighbor] = current
                    heapq.heappush(heap, (new_dist, neighbor))

        path = []
        try:
            while end is not None:
                path.append(end)
                end = prev[end]
            flag = False
        except:
            grid[max(0, end[0]-np.power(5,i)):min(grid.shape[0], end[0]+np.power(5,i)), max(0, end[1]-np.power(5,i)):min(grid.shape[1], end[1]+np.power(5,i))] = 1
            i+=1
    return (len(path)-1)*scale

######################## visualization ########################
def generate_unique_color(tag, hue_step=0.41):
    """Generate a unique color for each tag based on its hash value."""
    h = hash(tag) * hue_step % 1
    return hsv_to_rgb((h, 0.75, 0.75))

def visualize_panoptic_segmentation_num(image, panoptic_seg_id, result, model): 
    """
    Visualize the panoptic segmentation results of an image.
    Parameters: 
    - image: PIL.Image, the original image.
    - panoptic_seg_id: numpy.ndarray, the ID map of panoptic segmentation.
    - result: Output result from the DETR model which includes segment information.
    - model: The loaded DETR model which contains label configuration.
    """ 
    # Generate a color image of the same size as the segmentation map to represent the segmentation results
    panoptic_seg_rgb = np.zeros((panoptic_seg_id.shape[0], panoptic_seg_id.shape[1], 3), dtype=np.uint8) 
    legend_elements = []    
    seen_labels = dict()
    
    # Iterate over all unique segmentation IDs and assign a random color to each segmented area
    unique_ids = np.unique(panoptic_seg_id) 
    for category_id in unique_ids: 
        if result.get(category_id) is None:
            continue
        label = model.config.id2label[result[category_id]]
        print(label)
        if category_id == 0:  # ID 0 is for the background
            continue 
        if label not in seen_labels:
            color = generate_unique_color(category_id)
            seen_labels[label] = color
            # Create a legend element for the label
            legend_elements.append(patches.Patch(facecolor=color, edgecolor='r', label=label))
            # Color the corresponding areas in the RGB segmentation image
            panoptic_seg_rgb[panoptic_seg_id == category_id] = (color * 255).astype(np.uint8)
        else:
            # Reuse the color if the label has already been seen
            panoptic_seg_rgb[panoptic_seg_id == category_id] = (seen_labels[label] * 255).astype(np.uint8)
    
    # Create a canvas to display the original image and segmentation result
    fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 
    axs[0].imshow(image) 
    axs[0].set_title('RGB') 
    axs[0].axis('off') 
    axs[1].imshow(panoptic_seg_rgb)
    axs[1].set_title('Segmentation') 
    axs[1].axis('off') 
    # Position the legend outside of the plot
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('seg_num') 
    plt.close()

def visualize_panoptic_segmentation_loc(image, panoptic_seg_id, model, name = 'seg_loc'): 
    """
    Visualize the panoptic segmentation results of an image.
    Parameters: 
    - image: PIL.Image, the original image.
    - panoptic_seg_id: numpy.ndarray, the ID map of panoptic segmentation.
    - result: Output result from the DETR model which includes segment information.
    - model: The loaded DETR model which contains label configuration.
    """ 
    # Generate a color image of the same size as the segmentation map to represent the segmentation results
    panoptic_seg_rgb = np.zeros((panoptic_seg_id.shape[0], panoptic_seg_id.shape[1], 3), dtype=np.uint8) 
    legend_elements = []    
    seen_labels = dict()
    
    # Iterate over all unique segmentation IDs and assign a random color to each segmented area
    unique_ids = np.unique(panoptic_seg_id) 
    for category_id in unique_ids: 
        label = model.config.id2label[category_id]
        print(label)
        if category_id == 0:  # ID 0 is for the background
            continue 
        if label not in seen_labels:
            color = generate_unique_color(category_id)
            seen_labels[label] = color
            # Create a legend element for the label
            legend_elements.append(patches.Patch(facecolor=color, edgecolor='r', label=label))
            # Color the corresponding areas in the RGB segmentation image
            panoptic_seg_rgb[panoptic_seg_id == category_id] = (color * 255).astype(np.uint8)
        else:
            # Reuse the color if the label has already been seen
            panoptic_seg_rgb[panoptic_seg_id == category_id] = (seen_labels[label] * 255).astype(np.uint8)
    
    # Create a canvas to display the original image and segmentation result
    fig, axs = plt.subplots(1, 1, figsize=(9, 6)) 
    axs.imshow(panoptic_seg_rgb)
    axs.set_title('Segmentation') 
    axs.axis('off') 
    # Position the legend outside of the plot
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(name) 
    plt.close()

def plot_point_clouds(original_cloud, compressed_cloud, title1='Original Point Cloud', title2='Compressed Point Cloud'):
        """
        Plot a comparison of original and compressed point clouds in 3D.
        
        Args:
        - original_cloud (numpy array): The original point cloud data of shape (n, 3).
        - compressed_cloud (numpy array): The compressed point cloud data.
        - title1 (str): Title for the original point cloud plot.
        - title2 (str): Title for the compressed point cloud plot.
        """
        fig = plt.figure(figsize=(14, 7))

        # Original point cloud
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(original_cloud[:, 0], original_cloud[:, 1], original_cloud[:, 2], color='blue', alpha=0.5)
        ax1.set_title(title1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Compressed point cloud
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(compressed_cloud[:, 0], compressed_cloud[:, 1], compressed_cloud[:, 2], color='red', alpha=0.5)
        ax2.set_title(title2)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        plt.tight_layout()
        plt.show()
        plt.close()

def annotate_original_map(rgb_image, instance_masks):

    # Assuming original_image is a numpy array representing your image
    # Assuming instance_masks is a numpy array with the same dimensions as original_image
    # where each unique number represents a different instance (0 is typically for background)
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize = (8, 8), dpi = 64)

    # Show the original image
    ax.imshow(rgb_image)

    # Overlay each unique instance from the instance masks
    for instance_label in np.unique(instance_masks):
        if instance_label == -1:  # Skip the background
            continue

        # Create a mask for the current instance
        mask = instance_masks == instance_label
        centroid = (np.mean(np.where(mask)[0]), np.mean(np.where(mask)[1]))

        color = tuple(rgb_to_hsv(np.array(rgb_image.getpixel((centroid[1],centroid[0])))/255))

        # Create an RGBA image for the current instance
        colored_mask = np.zeros((*mask.shape, 4), dtype=np.float32)
        colored_mask[mask] = np.array([*color, 0.1])  # Add semi-transparency

        # Overlay the colored mask on the original image
        ax.imshow(colored_mask)

        # Calculate the centroid and place the label
        ax.text(centroid[1], centroid[0], str(instance_label), color='white', ha='center', va='center', fontsize=20, fontweight = 'bold')
    ax.axis('off')  # Turn off the axis
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    # image = Image.open(buf)
    # buf.close()
    return buf

######################## others ########################
_original_default = JSONEncoder.default
def new_default(self, obj):
    if isinstance(obj, np.ndarray):
        return list(obj)
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return _original_default(self, obj)
JSONEncoder.default = new_default

def load_file(file_path: str):
    """Load data from a file based on its file extension.

    Supported file types include:
    - .json: Load a JSON file
    - .yaml or .yml: Load a YAML file
    - .pkl: Load a Pickle file
    - .npy: Load a NumPy file

    Args:
    file_path (str): The path to the file to be loaded.

    Returns:
    object: The contents of the file.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Get the file extension
    _, ext = os.path.splitext(file_path)

    try:
        if ext == ".json":
            with open(file_path, "r") as file:
                return json.load(file)
        elif ext in [".yaml", ".yml"]:
            with open(file_path, "r") as file:
                return yaml.safe_load(file)
        elif ext == ".pkl":
            with open(file_path, "rb") as file:
                return pickle.load(file)
        elif ext == ".npy":
            return np.load(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    except Exception as e:
        print(f"Failed to load the file: {e}")
        raise

def find_semantic_matches(df1:pd.DataFrame, df2:pd.DataFrame):
    """
    Find semantic matches between two dataframes using natural language processing.

    This function compares each 'Name' in df2 against all 'Name' entries in df1 to find the closest semantic match using spaCy's similarity measurement. The function returns a DataFrame containing the original name from df2, its closest match in df1, and the corresponding ID from df1.

    Args:
    df1 (pd.DataFrame): The first DataFrame, containing columns 'Name' and 'ID'. It acts as the reference dataset.
    df2 (pd.DataFrame): The second DataFrame, containing the column 'Name'. This is the dataset whose entries are to be matched against df1.

    Returns:
    pd.DataFrame: A new DataFrame with columns ['Original Name', 'Closest Match', 'ID'] where:
        - 'Original Name' is the name from df2.
        - 'Closest Match' is the name from df1 that has the highest similarity to the 'Original Name'.
        - 'ID' is the ID of the 'Closest Match' from df1.
    """
    results = pd.DataFrame(columns=['Original Name', 'Closest Match', 'ID'])
    nlp = spacy.load('en_core_web_md')
    for name in df2['Name']:
        doc1 = nlp(name)
        max_similarity = -1
        match_id = None
        closest_match = None
        for _, row in df1.iterrows():
            doc2 = nlp(row['Name'])
            similarity = doc1.similarity(doc2)
            if similarity > max_similarity:
                max_similarity = similarity
                match_id = row['ID']
                closest_match = row['Name']
        results = pd.concat([results, pd.DataFrame({'Original Name': name, 'Closest Match': closest_match, 'ID': match_id}, index=[0])], ignore_index=True)
    return results