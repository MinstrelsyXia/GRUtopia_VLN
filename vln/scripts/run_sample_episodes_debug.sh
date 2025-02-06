#!/bin/bash
flags_sample_episodes="
    --vln_cfg_file vln/configs/vln_extract_data_multi_process_debug.yaml
    --sim_cfg_file vln/configs/sample_episodes_sim_cfg_debug.yaml
    --headless
    --save_path_planning
    --split val_seen
    --scan dhjEzFoUFzH
    --path_id 5815
"

python vln/main_sample_episode.py $flags_sample_episodes