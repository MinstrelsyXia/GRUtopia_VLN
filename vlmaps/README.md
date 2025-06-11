# VLMap-Plus:

## Quick Start
1. Install the environment
```shell
pip install -r vlmaps/requirements.txt
```
2. run file:

```shell
# start the main program: 
bash vlmaps/docker/slurm_run_single_episode.sh
# shutdown all the programs:
bash vlmaps/docker/shutdown_multiple_pids.sh
```

which will run `isaac_robot_docker.py` and record your logs at 

3. File Configuration:
The default bash command will run `vlmap_dataset_cfg` at `vlmaps/config_my/vlmap_dataset_cfg.yaml`. Built upon the main `vln` task, the config 

4. Dataset preparation:
- You should put the `Matterport3D` and `VLN` folder at the same level of your workspace, or config it at `base_data_dir` and `mp3d_data_dir` at `vln/configs/vln_cfg_vlmap.yaml`.

```
Matterport3D/
├── convert_mp3d_obj_to_usd.py* 
├── copy_fixed_to_scans.py 
├── data/ 
    ├── v1/
        ├── scans/
            ├── zsNo4HB9uLZ/
├── download_mp.py* 
├── process_mp.py* 
├── scans/ 
    ├── zsNo4HB9uLZ/
        ├── matterport_mesh/
└── .vscode/
VLNCE/
├── R2R_VLNCE_v1-3_corrected/
    ├── gather_data/
    ├── train/
    ├── val_seen/
        ├── val_seen.json.gz
    ├── val_unseen/
```

To create the target dataset for evaluation, see `vlmaps/docker/form_dataset.py`
1. Execute `python vlmaps/docker/form_dataset.py --action load` 