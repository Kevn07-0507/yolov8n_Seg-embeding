@echo off
REM 自动安装脚本 - Windows

echo ========================================
echo 墙面裂缝检测系统 - 自动安装
echo ========================================
echo.

REM 检查Python版本
echo 检查Python版本...
python --version
if errorlevel 1 (
    echo 错误: Python未安装
    pause
    exit /b 1
)

echo.
echo 检查pip...
pip --version
if errorlevel 1 (
    echo 错误: pip未安装
    pause
    exit /b 1
)

echo.
set /p create_venv="是否创建虚拟环境? (Y/N): "
if /i "%create_venv%"=="Y" (
    echo 创建虚拟环境...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo 虚拟环境已激活
)

echo.
echo 安装依赖...
pip install -r requirements.txt

echo.
echo 检查CUDA...
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 下一步:
echo 1. 运行快速测试: python quick_start.py
echo 2. 启动主菜单: python main.py
echo.

pause
