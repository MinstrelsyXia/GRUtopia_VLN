export PYTHONPATH="/ssd/xiaxinyuan/code/w61-grutopia/thirdparty/landmark_isaacsim_interaction/scgs_renderer/:/ssd/xiaxinyuan/code/w61-grutopia/thirdparty/landmark_isaacsim_interaction/interaction_emulator/:${PYTHONPATH}"

# 定义设备映射并严格限制
declare -A device_map
device_map[0]="4,5"    # rank 0 只能使用 GPU 2,3
device_map[1]="5,6"    # rank 1 只能使用 GPU 4,5
device_map[2]="6, 7"    # rank 2 只能使用 GPU 6,7

# 配置文件路径
CONFIG_FILE="vln/configs/v2/sixth_floor.json"

# 启动多个进程
for idx in 2; do
    # 每次启动前清理环境变量
    unset CUDA_VISIBLE_DEVICES
    
    # 设置进程专属的 GPU
    export CUDA_VISIBLE_DEVICES=${device_map[$idx]}
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    # 启动程序并记录 PID
    python vln/src/task/sample_one_scan.py \
        --rank $idx \
        --scan "sixth_floor" \
        --cfg_file "$CONFIG_FILE" > logs_rank_${idx}.log 2>&1 &
    
    pid=$!
    echo "Started process $idx (PID: $pid) on GPU ${device_map[$idx]}"
    
    # 等待进程确实启动
    sleep 5
done

# 验证进程和 GPU 分配
echo "Checking GPU assignments:"
nvidia-smi

# 等待所有后台进程完成
wait

echo "All processes completed"