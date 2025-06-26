#!/usr/bin/env bash
# filepath: setup_conda_path.sh
set -eo &> /dev/null

# 定义默认的 Isaac Sim 路径
DEFAULT_ISAAC_SIM_PATH="/cpfs/shared/simulation/xiaxinyuan/isaac-sim-4.2.0"

# 检查默认路径是否存在
if [[ -d "$DEFAULT_ISAAC_SIM_PATH" ]]; then
  echo "找到 Isaac Sim 安装在 [4m$DEFAULT_ISAAC_SIM_PATH[0m."
  read -p "如果要使用不同的路径，请输入包含 isaac-sim.sh 的路径（按回车跳过）>>> " ISAAC_SIM_PATH
  ISAAC_SIM_PATH=${ISAAC_SIM_PATH:-$DEFAULT_ISAAC_SIM_PATH}
# 检查常规的安装位置
elif [[ -d ~/.local/share/ov/pkg ]] && [[ $(ls ~/.local/share/ov/pkg | grep 'isaac[-_]sim') ]]; then
  FOUND_ISAAC_SIM_PATH=$(ls -d ~/.local/share/ov/pkg/* | grep 'isaac[-_]sim' | tail -n 1)
  echo "在 [4m$FOUND_ISAAC_SIM_PATH[0m 找到了 Isaac Sim。默认将使用此路径。"
  read -p "如果要使用不同的路径，请输入包含 isaac-sim.sh 的路径（按回车跳过）>>> " ISAAC_SIM_PATH
  ISAAC_SIM_PATH=${ISAAC_SIM_PATH:-$FOUND_ISAAC_SIM_PATH}
# 如果找不到，请用户输入
else
  echo "在默认位置未找到 Isaac Sim。"
  echo "如果您尚未安装 Isaac Sim，请先安装它。"
  read -p "如果已经在自定义位置安装，请输入包含 isaac-sim.sh 的路径 >>> " ISAAC_SIM_PATH
fi

# 验证路径有效性
while [[ ! -f "${ISAAC_SIM_PATH}/isaac-sim.sh" ]]; do
  read -p "在 [4m$ISAAC_SIM_PATH[0m 中未找到 isaac-sim.sh！请确保输入了正确的路径 >>> " ISAAC_SIM_PATH
done
echo -e "\n使用位于 [4m$ISAAC_SIM_PATH[0m 的 Isaac Sim\n"

# 将路径导出为环境变量，以便其他脚本使用
export ISAAC_SIM_PATH

# 设置 ISAAC_PATH 环境变量（Isaac Sim 内部使用）
export ISAAC_PATH=$ISAAC_SIM_PATH

# 添加到 PATH
if [[ ":$PATH:" != *":$ISAAC_SIM_PATH:"* ]]; then
  export PATH=$ISAAC_SIM_PATH:$PATH
fi

# 设置 Python 路径
if [[ -z "$PYTHONPATH" ]]; then
  export PYTHONPATH=$ISAAC_SIM_PATH
else
  export PYTHONPATH=$ISAAC_SIM_PATH:$PYTHONPATH
fi

# 生成 conda 环境配置的相关路径设置
if [ ! -z "$CONDA_PREFIX" ]; then
  # 确保 conda 环境激活/停用目录存在
  mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
  mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d

  # 创建激活脚本
  CONDA_ACT_FILE="${CONDA_PREFIX}/etc/conda/activate.d/isaac_sim_env_vars.sh"
  echo '#!/bin/sh' > ${CONDA_ACT_FILE}
  echo "export LD_LIBRARY_PATH_OLD=\$LD_LIBRARY_PATH" >> ${CONDA_ACT_FILE}
  echo "export PYTHONPATH_OLD=\$PYTHONPATH" >> ${CONDA_ACT_FILE}
  echo "export ISAAC_SIM_PATH=$ISAAC_SIM_PATH" >> ${CONDA_ACT_FILE}
  echo "source ${ISAAC_SIM_PATH}/setup_conda_env.sh" >> ${CONDA_ACT_FILE}
  chmod +x ${CONDA_ACT_FILE}

  # 创建停用脚本
  CONDA_DEACT_FILE="${CONDA_PREFIX}/etc/conda/deactivate.d/isaac_sim_env_vars.sh"
  echo '#!/bin/sh' > ${CONDA_DEACT_FILE}
  echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH_OLD" >> ${CONDA_DEACT_FILE}
  echo "export PYTHONPATH=\$PYTHONPATH_OLD" >> ${CONDA_DEACT_FILE}
  echo "unset ISAAC_SIM_PATH" >> ${CONDA_DEACT_FILE}
  echo "unset ISAAC_PATH" >> ${CONDA_DEACT_FILE}
  echo "unset CARB_APP_PATH" >> ${CONDA_DEACT_FILE}
  echo "unset LD_LIBRARY_PATH_OLD" >> ${CONDA_DEACT_FILE}
  echo "unset PYTHONPATH_OLD" >> ${CONDA_DEACT_FILE}
  chmod +x ${CONDA_DEACT_FILE}

  echo "Isaac Sim 路径设置已添加到 conda 环境配置中。"
else
  echo "未检测到活动的 conda 环境。"
  echo "如果要在 conda 环境中使用这些设置，请先激活环境，然后再运行此脚本。"
  echo "当前设置仅对当前 shell 会话有效。"
fi

echo -e "\nIsaac Sim 路径已设置为：[4m$ISAAC_SIM_PATH[0m\n"
echo "您可以通过运行以下命令来使用此路径："
echo "  source setup_conda_path.sh"