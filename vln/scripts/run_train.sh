#!/bin/bash
export MAGNUM_LOG=quiet
export PYTHONPATH=$PYTHONPATH:/ailab/user/wangliuyi/code/w61_grutopia

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
    --exp-config vln/configs/train/cma_dp_train.yaml
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

elif [ "$1" == "--collect_dataset" ]; then
  flags="
    --exp-config vlnce_baselines/config/r2r_baselines/dp/dp_collect_data.yaml
    --run-type collect_dataset
  "
elif [ "$1" == "--eval" ]; then
  flags="
    --run-type eval
    --exp-config vln/configs/train/cma_dp_w61.yaml
    --headless
    --test_verbose
  "
elif [ "$1" == "--train_noRNN" ]; then
  flags="
    --exp-config vln/configs/train/cma_dp_train_noRNN.yaml
    --run-type train
  "
elif [ "$1" == "--preprocess_features" ]; then
  flags="
    --exp-config vln/configs/train/cma_dp_train_noRNN.yaml
    --run-type preprocess_features
  "
fi
python vln/run_policy.py $flags