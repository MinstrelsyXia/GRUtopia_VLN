#!/bin/bash
export MAGNUM_LOG=quiet
export PYTHONPATH=$PYTHONPATH:/ailab/user/wangliuyi/code/w61_grutopia

max_attempts=90  # 最大重试次数
attempt=1

while [ $attempt -le $max_attempts ]; do
    if [ "$1" == "--val_seen" ]; then
        flags="
            --exp-config vln/configs/train/cma_dp_eval_val_seen.yaml
            --run-type eval
            --headless
            --test_verbose
        "
    elif [ "$1" == "--val_unseen" ]; then
        flags="
            --exp-config vln/configs/train/cma_dp_eval_val_unseen.yaml
            --run-type eval
            --headless
            --test_verbose
        "
    elif [ "$1" == "--train" ]; then
        flags="
            --exp-config vln/configs/train/cma_dp_eval_train.yaml
            --run-type eval
            --headless
            --test_verbose
        "
    fi

    echo "尝试运行第 $attempt 次..."
    
    if python vln/run_policy.py $flags; then
        echo "程序成功完成"
        exit 0
    else
        echo "程序出错,退出码: $?"
        if [ $attempt -lt $max_attempts ]; then
            echo "等待 5 秒后重试..."
            sleep 5
        fi
        attempt=$((attempt + 1))
    fi
done

echo "达到最大重试次数 ($max_attempts),程序退出"
exit 1