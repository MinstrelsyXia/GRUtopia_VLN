'''
Author: w61
Date: 2024.7.25
NOTE: This method does not work!
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import omni
from omni.isaac.occupancy_map.bindings import _occupancy_map
from pxr import UsdGeom, Usd, Sdf, Gf, UsdPhysics, UsdUtils

from grutopia.core.util.log import log

class IsaacOccupancyMap:
    def __init__(self, args):
        self.args = args
        # get current scene bounding box
        self.scene_min_bounds, self.scene_max_bounds = self.get_scene_bbox()
        self.threhold = 10 # threshold for the occupancy map
    
    def generate_occupancy_map(self, origin_pos, save_img=False):
        ''' Generate 2D occupancy map from Isaac Sim
        '''
        physx = omni.physx.acquire_physx_interface()
        stage_id = omni.usd.get_context().get_stage_id()

        generator = _occupancy_map.Generator(physx, stage_id)
        # 0.05m cell size, output buffer will have 4 for occupied cells, 5 for unoccupied, and 6 for cells that cannot be seen
        # this assumes your usd stage units are in m, and not cm
        generator.update_settings(.05, 4, 5, 6)
        # Set location to map from and the min and max bounds to map to
        min_bounds = (max(self.scene_min_bounds[0], origin_pos[0]-self.threhold), 
                      max(self.scene_min_bounds[1], origin_pos[1]-self.threhold),
                      origin_pos[2]-1.05)
        max_bounds = (min(self.scene_max_bounds[0], origin_pos[0]+self.threhold),
                      min(self.scene_max_bounds[1], origin_pos[1]+self.threhold),
                      origin_pos[2]-1.05)
        origin_pos = (origin_pos[0], origin_pos[1], origin_pos[2]-1.05)
        # min_bounds = (self.scene_min_bounds[0], self.scene_min_bounds[1], origin_pos[2])
        # max_bounds = (self.scene_max_bounds[0], self.scene_max_bounds[1], origin_pos[2])
        generator.set_transform(origin_pos, min_bounds, max_bounds)
        generator.generate2d()
        if save_img:
            self.draw_map(generator.get_buffer(), generator.get_dimensions(), self.args.log_image_dir, save=True)
        
        # Get locations of the occupied cells in the stage
        # points = generator.get_occupied_positions()
        # # Get computed 2d occupancy buffer
        # buffer = generator.get_buffer()
        # # Get dimensions for 2d buffer
        # dims = generator.get_dimensions()
        # print(1)

    def get_scene_bbox(self, scene_prim_path="/World/env_0/scene"):
        # Get the bounding box information of the model
        stage = omni.usd.get_context().get_stage()
        model_prim = stage.GetPrimAtPath(scene_prim_path) # NOTE: how to achieve the current scene when there are different envs
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(model_prim)

        # Get the minimum and maximum values of the bounding box
        min_point = bbox.GetRange().GetMin()
        max_point = bbox.GetRange().GetMax()
        return min_point, max_point

    def draw_map(self, buffer, dims, log_image_dir='', save=False):
        plt.clf()
        occupancy_map = np.array(buffer).reshape(dims[1], dims[0])

        # Create a custom colormap
        cmap = ListedColormap(['black', 'white', 'gray'])
        # bounds = [4,5,6]
        norm = plt.Normalize(vmin=4, vmax=6)

        # Plot the occupancy map
        plt.imshow(occupancy_map, cmap=cmap, norm=norm)
        plt.colorbar(ticks=[4, 5, 6], label='Occupancy State')
        plt.title('Occupancy Map Visualization')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        # plt.xticks(np.arange(occupancy_map.shape[1]))
        # plt.yticks(np.arange(occupancy_map.shape[0]))
        # plt.grid(which='both', color='grey', linestyle='-', linewidth=0.5)
        # plt.show()
        if save:
            save_file = os.path.join(log_image_dir, 'isaacsim_occupancy_map.png')
            plt.savefig(save_file)
            log.info(f"Isaac Occupancy map saved at {save_file}")