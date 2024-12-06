#!/bin/bash
#SBATCH --job-name=train_depth         # 作业名称
#SBATCH --output=logs/%x_%j.out          # 标准输出文件路径 (%j 会被替换为作业ID)
#SBATCH --error=logs/%x_%j.err           # 标准错误文件路径
#SBATCH --gres=gpu:4                  # GPU请求
#SBATCH --cpus-per-task=24             # 每个任务的CPU核心数
#SBATCH --partition=smartbot              # 使用GPU分区

# 创建日志目录
mkdir -p logs

source activate grutopia_train

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