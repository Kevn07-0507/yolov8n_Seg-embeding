@echo off
REM Windows启动脚本

echo ========================================
echo 墙面裂缝检测系统
echo ========================================
echo.

REM 优先使用 venv Python (含 CUDA)
set VENV_PYTHON=E:\torch\.venv\Scripts\python.exe
if exist "%VENV_PYTHON%" (
    echo 使用 venv Python: %VENV_PYTHON%
    set PYTHON=%VENV_PYTHON%
) else (
    REM 回退到系统 Python
    python --version >nul 2>&1
    if errorlevel 1 (
        echo 错误: 未检测到Python，请先安装Python 3.8+
        pause
        exit /b 1
    )
    echo 使用系统 Python
    set PYTHON=python
)

echo.

REM 检查 PyTorch
%PYTHON% -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo 警告: PyTorch未安装
    echo 是否现在安装依赖? (Y/N)
    set /p install_deps=
    if /i "%install_deps%"=="Y" (
        echo 安装依赖...
        %PYTHON% -m pip install -r requirements.txt
    )
)

echo.
echo 启动主程序...
%PYTHON% main.py

pause
