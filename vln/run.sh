#!/bin/bash
flags="
    --headless
    --test_verbose
    --save_obs
    --path_id 5593
    --windows_head
"
python vln/main.py $flags