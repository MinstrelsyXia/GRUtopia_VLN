#!/bin/bash
#SBATCH -J 20250212_slurm_test
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

CONFIG_FILE="vln/configs/v2/sample.json"


python vln/launch.py --rank $1 --cfg_file "$CONFIG_FILE"
