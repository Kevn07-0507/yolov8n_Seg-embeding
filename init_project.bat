@echo off
REM 项目初始化脚本

echo ========================================
echo 墙面裂缝检测系统 - 项目初始化
echo ========================================
echo.

REM 创建必要的目录
echo 创建项目目录...
if not exist logs mkdir logs
if not exist results mkdir results
if not exist exports mkdir exports
if not exist batch_results mkdir batch_results
if not exist comparison_results mkdir comparison_results
if not exist analysis_results mkdir analysis_results
if not exist visualization_results mkdir visualization_results
if not exist augmentation_preview mkdir augmentation_preview
if not exist benchmark_results mkdir benchmark_results

echo 目录创建完成
echo.

REM 检查数据集
echo 检查数据集...
if exist crack-seg (
    echo 数据集目录存在
) else (
    echo 数据集目录不存在
    echo 请确保crack-seg目录存在并包含数据
)

echo.
echo ========================================
echo 初始化完成！
echo ========================================
echo.
echo 下一步:
echo 1. 安装依赖: install.bat
echo 2. 测试系统: python test_system.py
echo 3. 启动程序: python main.py
echo.

pause
