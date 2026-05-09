#!/bin/bash
# Linux/Mac启动脚本

echo "========================================"
echo "墙面裂缝检测系统"
echo "========================================"
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未检测到Python，请先安装Python 3.8+"
    exit 1
fi

echo "Python已安装"
echo ""

# 检查依赖
echo "检查依赖..."
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: PyTorch未安装"
    read -p "是否现在安装依赖? (y/n) " install_deps
    if [ "$install_deps" = "y" ] || [ "$install_deps" = "Y" ]; then
        echo "安装依赖..."
        pip3 install -r requirements.txt
    fi
fi

echo ""
echo "启动主程序..."
python3 main.py
