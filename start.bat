@echo off
REM Windows启动脚本

echo ========================================
echo 墙面裂缝检测系统
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未检测到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo Python已安装
echo.

REM 检查依赖
echo 检查依赖...
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo 警告: PyTorch未安装
    echo 是否现在安装依赖? (Y/N)
    set /p install_deps=
    if /i "%install_deps%"=="Y" (
        echo 安装依赖...
        pip install -r requirements.txt
    )
)

echo.
echo 启动主程序...
python main.py

pause
