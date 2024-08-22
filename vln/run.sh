#!/bin/bash
flags="
    --headless
    --test_verbose
    --save_obs
    --path_id 5593
    --windows_head
"

flags_sample_episodes="
    --test_verbose
    --headless
    --save_obs
    --vln_cfg_file vln/configs/vln_extract_data.yaml
    --sim_cfg_file vln/configs/sample_episodes_sim_cfg.yaml
    --windows_head
    --windows_head_type save
"

python vln/main.py $flags_sample_episodes