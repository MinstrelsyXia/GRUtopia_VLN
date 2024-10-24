#!/bin/bash
flags="
    --headless
    --test_verbose
    --save_obs
    --path_id 5593
    --windows_head
"

flags_gui="
    --test_verbose
    --windows_head
    --path_id 1726
    --split val_seen
"

flags_sample_episodes="
    --headless
    --vln_cfg_file vln/configs/vln_extract_data.yaml
    --sim_cfg_file vln/configs/sample_episodes_sim_cfg.yaml
"

flags_sample_episodes_script_single_scan="
    --headless
    --vln_cfg_file vln/configs/vln_extract_data_script.yaml
    --sim_cfg_file vln/configs/sample_episodes_sim_cfg.yaml
    --split train
    --scan JeFG25nYj2p
    --test_verbose
    --windows_head
    --windows_head_type save
"

python vln/main.py $flags_gui
