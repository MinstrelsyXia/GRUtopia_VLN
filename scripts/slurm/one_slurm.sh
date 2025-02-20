#!/bin/bash
#SBATCH -J 20250220_h1_dp_ckpt30_eval
#SBATCH -o slurm_logs/%j_%x.out 
#SBATCH -e slurm_logs/%j_%x.err 
#SBATCH -p gpu_4090  
#SBATCH -N 1 ###使用1个节点
#SBATCH -n 6 ###总共申请6个CPU核心
#SBATCH --gres=gpu:1 ###每个节点使用1张

module load anaconda/2024.02
source activate grutopia

export MAGNUM_LOG=quiet
export PYTHONPATH=$PYTHONPATH:/ailab/user/wangliuyi/.conda/envs/grutopia/bin/python

SAMPLE_CONFIG_FILE="vln/configs/v2/sample.json"
EVAL_CONFIG_FILE="vln/configs/v2/eval.json"

if [ "$1" == "sample" ]; then
    CONFIG_FILE=$SAMPLE_CONFIG_FILE
    echo "Using sample config file: $CONFIG_FILE"
elif [ "$1" == "eval" ]; then
    CONFIG_FILE=$EVAL_CONFIG_FILE
    echo "Using eval config file: $CONFIG_FILE"
else
    echo "Invalid argument"
    exit 1
fi

python vln/launch.py --rank $2 --cfg_file "$CONFIG_FILE"
