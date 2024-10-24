from pathlib import Path
import hydra
from omegaconf import DictConfig
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
from vlmaps.vlmaps.map.vlmap import VLMap
from vlmaps.vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.vlmaps.utils.visualize_utils import (
    pool_3d_label_to_2d,
    pool_3d_rgb_to_2d,
    visualize_rgb_map_3d,
    visualize_masked_map_2d,
    visualize_heatmap_2d,
    visualize_heatmap_3d,
    visualize_masked_map_3d,
    get_heatmap_from_mask_2d,
    get_heatmap_from_mask_3d,
)




@hydra.main(
    version_base=None,
    config_path="../config_my",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    print(config.map_config)
    # data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    # print(data_dirs[config.scene_id])
    # vlmap = VLMap(config.map_config, data_dir=data_dirs[config.scene_id])
    vlmap_save_path =  '/ssd/xiaxinyuan/code/w61-grutopia/logs/5LpN3gDmAk7_1/'
    vlmap = VLMap(config.map_config, data_dir= vlmap_save_path)
    vlmap.load_map(vlmap_save_path)

    # cat = input("What is your interested category in this scene?")
    cat = "bed"

    vlmap._init_clip()
    print("considering categories: ")
    print(mp3dcat[1:-1])

    # init categories with cat and other:
    mask = vlmap.index_map(cat, with_init_cat=False)
    mask_2d = pool_3d_label_to_2d(mask, vlmap.grid_pos, config.params.gs)
    rgb_2d = pool_3d_rgb_to_2d(vlmap.grid_rgb, vlmap.grid_pos, config.params.gs)

    print(vlmap.map_save_path)

    save_path = Path(vlmap_save_path) /'vlmap_cam'/ f"{cat}_masked_2d_binary.jpg"
    visualize_masked_map_2d(rgb_2d, mask_2d, save_path = save_path)

    heatmap = get_heatmap_from_mask_2d(mask_2d, cell_size=config.params.cs, decay_rate=config.decay_rate)
    save_path = Path(vlmap_save_path)/'vlmap_cam'/ f"{cat}_heatmap_2d_binary.jpg"
    visualize_heatmap_2d(rgb_2d, heatmap,save_path=save_path)



    vlmap.init_categories(mp3dcat[1:-1])
    mask = vlmap.index_map(cat, with_init_cat=True)
    
    mask_2d = pool_3d_label_to_2d(mask, vlmap.grid_pos, config.params.gs)
    rgb_2d = pool_3d_rgb_to_2d(vlmap.grid_rgb, vlmap.grid_pos, config.params.gs)
    save_path = Path(vlmap_save_path) /'vlmap_cam'/ f"{cat}_masked_2d.jpg"
    visualize_masked_map_2d(rgb_2d, mask_2d,save_path=save_path)

    heatmap = get_heatmap_from_mask_2d(mask_2d, cell_size=config.params.cs, decay_rate=config.decay_rate)
    save_path = Path(vlmap_save_path) /'vlmap_cam'/ f"{cat}_heatmap_2d.jpg"
    visualize_heatmap_2d(rgb_2d, heatmap,save_path=save_path)



    # if config.init_categories:
    #     vlmap.init_categories(mp3dcat[1:-1])
    #     mask = vlmap.index_map(cat, with_init_cat=True)
    # else:
    #     mask = vlmap.index_map(cat, with_init_cat=False)

    # # init categories wit mp3d40:
    # if config.init_categories:
    #     vlmap.init_categories(mp3dcat[1:-1])
    #     mask = vlmap.index_map(cat, with_init_cat=True)
    # else:
    #     mask = vlmap.index_map(cat, with_init_cat=False)

    # if config.index_2d:
    #     mask_2d = pool_3d_label_to_2d(mask, vlmap.grid_pos, config.params.gs)
    #     rgb_2d = pool_3d_rgb_to_2d(vlmap.grid_rgb, vlmap.grid_pos, config.params.gs)
    #     save_path = Path(vlmap_save_path) /'vlmap_cam'/ f"{cat}_masked_2d.jpg"
    #     visualize_masked_map_2d(rgb_2d, mask_2d)

    #     heatmap = get_heatmap_from_mask_2d(mask_2d, cell_size=config.params.cs, decay_rate=config.decay_rate)
    #     save_path = Path(vlmap_save_path) /'vlmap_cam'/ f"{cat}_heatmap_2d.jpg"
    #     visualize_heatmap_2d(rgb_2d, heatmap,save_path=save_path)
    # else:
    #     visualize_masked_map_3d(vlmap.grid_pos, mask, vlmap.grid_rgb)
    #     heatmap = get_heatmap_from_mask_3d(
    #         vlmap.grid_pos, mask, cell_size=config.params.cs, decay_rate=config.decay_rate
    #     )
    #     visualize_heatmap_3d(vlmap.grid_pos, heatmap, vlmap.grid_rgb)
    



if __name__ == "__main__":
    main()
