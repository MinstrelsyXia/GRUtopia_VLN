#!/bin/bash
#SBATCH --job-name=dp_descreteDataset_Train         # 作业名称
#SBATCH --output=logs/%j_%x.out          # 标准输出文件路径
#SBATCH --error=logs/%j_%x.err           # 标准错误文件路径
#SBATCH --gres=gpu:4                # GPU请求
#SBATCH --cpus-per-task=24           # 每个任务的CPU核心数
#SBATCH --partition=smartbot              # 使用GPU分区
#SBATCH --mem=128G                     # 总内存分配

# 创建日志目录
mkdir -p logs

module load anaconda/2024.02
module unload tensorboard
source activate grutopia_train

export MAGNUM_LOG=quiet
export PYTHONPATH=$PYTHONPATH:/ailab/user/wangliuyi/code/w61_grutopia

# tar -xf /ailab/user/wangliuyi/code/w61_grutopia/data/sample_episodes/20241207_sample_episodes_processed/sample_data.lmdb.tar -C /dev/shm

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
elif [ "$1" == "--train_cma" ]; then
  flags="
    --exp-config vln/configs/train/cma_InstrLongCLIP_train.yaml
    --run-type train
    --train_quiet
  "
elif [ "$1" == "--train_cma_clip" ]; then
  flags="
    --exp-config vln/configs/train/cma_clip_train.yaml
    --run-type train
    --train_quiet
  "
fi
python vln/run_policy.py $flags