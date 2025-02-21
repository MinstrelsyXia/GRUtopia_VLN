salloc -N 1 -n 6 --gres=gpu:1 -p gpu_4090

# 显示完整name
squeue -u wangliuyi -o "%.18i %.12P %.25j %.8u %.2t %.10M %.6D %R"