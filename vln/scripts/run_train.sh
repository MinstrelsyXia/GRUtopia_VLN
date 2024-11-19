#!/bin/bash
export MAGNUM_LOG=quiet

flags_cma="
  --exp-config vlnce_baselines/config/r2r_baselines/cma.yaml
  --run-type eval
"

flags_cma_raw_train="
  --exp-config vlnce_baselines/config/r2r_baselines/dp/dp_raw_train.yaml
  --run-type train
"

if [ "$1" == "--train" ]; then
  flags="
    --exp-config vln/configs/train/cma_dp_w61.yaml
    --run-type train
  "
elif [ "$1" == "--train_crossGRU" ]; then
  flags="
    --exp-config vlnce_baselines/config/r2r_baselines/dp/cma_dp_train_crossGRU.yaml
    --run-type train
  "

elif [ "$1" == "--train_aug" ]; then
  flags="
    --exp-config vlnce_baselines/config/r2r_baselines/dp/cma_dp_w61_aug.yaml
    --run-type train
  "
  
elif [ "$1" == "--debug" ]; then
  flags="
    --exp-config vlnce_baselines/config/r2r_baselines/dp/dp_debug.yaml
    --run-type train
  "

elif [ "$1" == "--eval" ]; then
  flags="
    --exp-config vlnce_baselines/config/r2r_baselines/dp/cma_dp_eval.yaml
    --run-type eval
  "
elif [ "$1" == "--collect_dataset" ]; then
  flags="
    --exp-config vlnce_baselines/config/r2r_baselines/dp/dp_collect_data.yaml
    --run-type collect_dataset
  "
fi
python vln/run_policy.py $flags