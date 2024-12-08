#!/bin/bash
export MAGNUM_LOG=quiet
export PYTHONPATH=$PYTHONPATH:/ailab/user/wangliuyi/code/w61_grutopia

max_attempts=90  # 最大重试次数
attempt=1

while [ $attempt -le $max_attempts ]; do
    if [ "$1" == "--eval" ]; then
        flags="
            --exp-config vln/configs/train/cma_dp_w61.yaml
            --run-type eval
            --headless
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