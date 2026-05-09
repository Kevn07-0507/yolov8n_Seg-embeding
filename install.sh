#!/bin/bash
# 自动安装脚本 - Linux/Mac

echo "========================================"
echo "墙面裂缝检测系统 - 自动安装"
echo "========================================"
echo ""

# 检查Python版本
echo "检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "错误: pip3未安装"
    exit 1
fi

echo "pip3已安装"
echo ""

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境? (y/n) " create_venv
if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "虚拟环境已激活"
fi

echo ""
echo "安装依赖..."
pip3 install -r requirements.txt

echo ""
echo "检查CUDA..."
python3 -c "import torch; print('CUDA可用:', torch.cuda.is_available())"

echo ""
echo "========================================"
echo "安装完成！"
echo "========================================"
echo ""
echo "下一步:"
echo "1. 运行快速测试: python3 quick_start.py"
echo "2. 启动主菜单: python3 main.py"
echo ""
