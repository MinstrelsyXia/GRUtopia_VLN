# VLMap-Plus:

## Quick Start
1. Install the git repository and the environment:
```shell
# pip install -r vlmaps/requirements.txt
git clone https://github.com/MinstrelsyXia/GRUtopia_VLN.git
git checkout xxy_vlmap_final
# create a conda environment and download the required packages for grutopia
source setup_test_env.sh
# isaacsim package route: /cpfs/user/xiaxinyuan/isaac-sim-4.2.0
# download packages for vln:
pip install -r requirements_vln.txt
```


2. Dataset preparation:
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
GRUtopia_VLN(your working space)/
├── assets/
    ├── policy/
    ├── robots/
├── demo/
├── docs/
├── grutopia/
├── grutopia_extension/
```

You should put assets under your working space, i.e. `GRUtopia_VLN/assets/`. It is currently under `/cpfs/user/xiaxinyuan/assets/`. Try to create a symbolic link to the assets folder in your working space:
```shell
# under your working space
ln -s /cpfs/user/xiaxinyuan/assets/ .

```



3. Checkpoint preparation:


For VLMap, you should place the checkpoints at the following path:
```shell
# under your working space
cd GRUtopia_VLN
# lseg checkpoint for VLMap
ln -s /cpfs/user/xiaxinyuan/checkpoints/demo_e200.ckpt /cpfs/user/xiaxinyuan/code/aliyun/GRUtopia_VLN/vlmaps/vlmaps/lseg/checkpoints/

# predownloaded huggingface checkpoints for VLMap
mkdir -p ~/.cache/torch/hub/checkpoints/
ln -s /cpfs/user/xiaxinyuan/checkpoints/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz ~/.cache/torch/hub/checkpoints/

mkdir -p ~/.cache//huggingface/hub/
ln -s /cpfs/user/xiaxinyuan/checkpoints/models--timm--vit_large_patch16_384.augreg_in21k_ft_in1k/ ~/.cache/huggingface/hub/

mkdir -p ~/.cache/clip/
ln -s /cpfs/user/xiaxinyuan/checkpoints/ViT-B-32.pt  ~/.cache/clip/


# api key, use soft link to avoid unintended exposure
ln -s /cpfs/user/xiaxinyuan/code/api_key/ .
```

4. run the file to check the environment:
You can config your debug kernel at `.vscode/launch.json`
```shell
{
    "version": "0.2.0",
    "configurations": [ 
        {
            "name": "Python: run isaac_robot_docker",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/vlmaps/application_my/isaac_robot_docker.py",
            "console": "integratedTerminal",
            // "console": "externalTerminal",
            "env": {
                "RESOURCE_NAME": "IsaacSim",
            },
            "python": "/root/miniconda3/envs/grutopia/bin/python",
            "justMyCode": false,
            "args": [
                "--config-name", "vlmap_dataset_cfg",
                "episode_file=vlmaps/docker/debug/0.txt",
                "last_scan_file=vlmaps/docker/debug/last_scan_0.txt"
            ],
        },
    ]
}
```


4. File Configuration:
The default bash command will run `vlmap_dataset_cfg` at `vlmaps/config_my/vlmap_dataset_cfg.yaml`. Built upon the main `vln` task, the config loads the other configs as follows:

- `data_paths`: You should create a yaml file under `vlmaps/config_my/data_paths/`, i.e. `vlmaps/config_my/data_paths/vlmap_dataset.yaml`
  - `vlmaps_data_dir`: The directory where the pixel embeddings of vlmap is stored.
  - `test_file_save_dir`: The directory where the observations, obstacle map and other data for debug is saved
- `map_config`: Set params for the semantic map, usually no need to change.
  - `map_type`: Set to `IsaacSimMap` for the VLMap task. The main code is in `vlmaps/vlmaps/map/map.py` and `vlmaps/application_my/build_dynamic_map.py`
- `params`: Set params for the obstacle map for local navigation, usually no need to change.
- `vln_config`: Inherit from the main `vln` task
   - `headless`: Set to `True` so that the simulation will not open a GUI window.
   - `sim_cfg_file`: Choose the config file for the simulation, i.e. `vln/configs/sim_cfg_policy_h1_eval.yaml`
   - `vln_cfg_file`: Choose the config file for the VLN task, i.e. `vln/configs/vln_cfg_vlmap.yaml`
       - `base_data_dir`: Set to the directory where `R2R_VLNCE_v1-3` dataset stored, i.e. `../VLN/VLNCE/R2R_VLNCE_v1-3_corrected`.
       - `mp3d_data_dir`: Set to the directory where `Matterport3D` dataset stored, i.e. `../Matterport3D/data/v1/scans/`



6. run file:

```shell
# start the main program: 
bash vlmaps/docker/slurm_run_single_episode.sh
# shutdown all the programs:
bash vlmaps/docker/shutdown_multiple_pids.sh
```

which will run `isaac_robot_docker.py` and record your logs at 
To create the target dataset for evaluation, see `vlmaps/docker/form_dataset.py`
1. Execute `python vlmaps/docker/form_dataset.py --action load`


## real-2-sim pipeline:
