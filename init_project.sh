#!/bin/bash
# 项目初始化脚本

echo "========================================"
echo "墙面裂缝检测系统 - 项目初始化"
echo "========================================"
echo ""

# 创建必要的目录
echo "创建项目目录..."
mkdir -p logs
mkdir -p results
mkdir -p exports
mkdir -p batch_results
mkdir -p comparison_results
mkdir -p analysis_results
mkdir -p visualization_results
mkdir -p augmentation_preview
mkdir -p benchmark_results

echo "✓ 目录创建完成"
echo ""

# 检查数据集
echo "检查数据集..."
if [ -d "crack-seg" ]; then
    echo "✓ 数据集目录存在"
else
    echo "⚠ 数据集目录不存在"
    echo "  请确保crack-seg目录存在并包含数据"
fi

echo ""
echo "========================================"
echo "初始化完成！"
echo "========================================"
echo ""
echo "下一步:"
echo "1. 安装依赖: bash install.sh"
echo "2. 测试系统: python3 test_system.py"
echo "3. 启动程序: python3 main.py"
echo ""
