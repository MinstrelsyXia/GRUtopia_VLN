#!/bin/bash
#SBATCH --job-name=train_depth         # 作业名称
#SBATCH --output=logs/%x_%j.out          # 标准输出文件路径 (%j 会被替换为作业ID)
#SBATCH --error=logs/%x_%j.err           # 标准错误文件路径
#SBATCH --gres=gpu:6                  # GPU请求
#SBATCH --cpus-per-task=48            # 每个任务的CPU核心数
#SBATCH --partition=smartbot              # 使用GPU分区
#SBATCH --mem=256G                     # 总内存分配

# 创建日志目录
mkdir -p logs

source activate grutopia_train

export MAGNUM_LOG=quiet
export PYTHONPATH=$PYTHONPATH:/ailab/user/wangliuyi/code/w61_grutopia

tar -xf /ailab/user/wangliuyi/code/w61_grutopia/data/sample_episodes/20241207_sample_episodes_processed/sample_data.lmdb.tar -C /dev/shm


if [ "$1" == "--train" ]; then
  if [ "$2" == "ddp" ]; then
    # DDP training using torch.run
    # export CUDA_VISIBLE_DEVICES=4,5,6,7

    python -m torch.distributed.run --nproc_per_node=4 \
            --nnodes=1 \
            --node_rank=0 \
            --master_addr=localhost \
            --master_port=29500 \
            vln/run_policy.py \
            --exp-config vln/configs/train/cma_dp_train_noRNN.yaml \
            --run-type train
  else
    # Traditional DataParallel training
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
    python vln/run_policy.py \
           --exp-config vln/configs/train/cma_dp_train.yaml \
           --run-type train
  fi
fi 