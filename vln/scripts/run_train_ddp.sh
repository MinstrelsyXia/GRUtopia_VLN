#!/bin/bash
export MAGNUM_LOG=quiet
export PYTHONPATH=$PYTHONPATH:/ailab/user/wangliuyi/code/w61_grutopia

if [ "$1" == "--train" ]; then
  if [ "$2" == "ddp" ]; then
    # DDP training using torch.run
    export CUDA_VISIBLE_DEVICES=4,5,6
    export NCCL_SOCKET_IFNAME=lo
    # export NCCL_PORT_RANGE=29501-29510
    export NCCL_PORT_RANGE=29511-29515

    python -m torch.distributed.run --nproc_per_node=3 \
            --nnodes=1 \
            --node_rank=0 \
            --master_addr=localhost \
            --master_port=29510 \
            vln/run_policy.py \
            --exp-config vln/configs/train/cma_dp_train_noRNN.yaml \
            --run-type train \
            # NAME 20241214-vlnce-rp-noRNN-DPDec12-txtLayer12-txtQformer-stackRGB16-bs128-lr2e-4_continue \
            # TORCH_GPU_IDS [0,1,2,3] \
            # DDP.use_dp False \
            # IL.batch_size 40
  else
    # Traditional DataParallel training
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
    python vln/run_policy.py \
           --exp-config vln/configs/train/cma_dp_train.yaml \
           --run-type train
  fi
fi 