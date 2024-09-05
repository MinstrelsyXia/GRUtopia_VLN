#!/bin/bash
flags_sample_episodes="
    --vln_cfg_file vln/configs/vln_extract_data_multi_process.yaml
    --sim_cfg_file vln/configs/sample_episodes_sim_cfg.yaml
    --headless
    --save_path_planning
"

python vln/main_sample_episode.py $flags_sample_episodes