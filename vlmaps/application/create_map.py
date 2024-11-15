from pathlib import Path
import hydra
from omegaconf import DictConfig
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from vlmaps.vlmaps.map.vlmap import VLMap


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_creation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    vlmap = VLMap(config.map_config)
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()]) # 字典序排列

    vlmap.create_map(data_dirs[config.scene_id])


if __name__ == "__main__":
    main()
