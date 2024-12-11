#!/bin/bash
export MAGNUM_LOG=quiet
export PYTHONPATH=$PYTHONPATH:/ailab/user/wangliuyi/code/w61_grutopia

if [ "$1" == "--train" ]; then
  if [ "$2" == "ddp" ]; then
    # DDP training using torch.run
    export CUDA_VISIBLE_DEVICES=4,5,6,7

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