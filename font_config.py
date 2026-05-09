"""
matplotlib中文字体配置模块
统一处理所有可视化中的中文显示问题
"""
import matplotlib.pyplot as plt
import matplotlib
import platform

def setup_chinese_font():
    """设置matplotlib中文字体"""
    system = platform.system()

    if system == 'Windows':
        # Windows系统优先使用SimHei、Microsoft YaHei
        font_list = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    elif system == 'Darwin':
        # macOS系统使用PingFang SC
        font_list = ['PingFang SC', 'Arial Unicode MS', 'sans-serif']
    else:
        # Linux系统
        font_list = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']

    plt.rcParams['font.sans-serif'] = font_list
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    return True

# 自动执行配置
setup_chinese_font()
