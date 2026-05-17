import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm
import os

def set_ch_font():
    cn_fonts = [f for f in fm.fontManager.ttflist if 'Microsoft YaHei' in f.name or 'SimHei' in f.name]
    if cn_fonts:
        plt.rcParams['font.family'] = cn_fonts[0].name
    else:
        for fname in ['Microsoft YaHei', 'SimHei', 'STXihei', 'KaiTi']:
            try:
                plt.rcParams['font.family'] = fname
                break
            except: pass
    plt.rcParams['axes.unicode_minus'] = False

def draw_optimized_architecture():
    set_ch_font()
    
    # 创建画布
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    colors = {
        'data': '#4A90D9',
        'train': '#50B86C',
        'infer': '#E8913A',
        'app': '#9B59B6',
        'deploy': '#E74C3C',
    }

    # 绘制父框的函数 (标题进一步上移)
    def draw_parent_box(ax, x, y, w, h, text, color):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='#2C3E50', linewidth=1.5)
        ax.add_patch(box)
        # 标题紧贴顶部 (y + h - 0.15)
        ax.text(x + w/2, y + h - 0.15, text.replace('\n', ' '), ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

    # 绘制子框的函数
    def draw_sub_box(ax, x, y, w, h, text, color='#ECF0F1', fontsize=9):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor='#BDC3C7', linewidth=1)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color='#2C3E50')

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=2))

    # --- 标题 ---
    ax.text(8, 8.2, '墙面裂缝智能巡检系统 — 总体架构', ha='center', va='center',
            fontsize=18, fontweight='bold', color='#2C3E50')

    # --- 第一行模块：高度增加到 2.2，起始 y 稍微下调 ---
    y_base = 4.6
    h_main = 2.2

    # 1. 数据层
    draw_parent_box(ax, 0.3, y_base, 2.8, h_main, '数据层 (Data Layer)', colors['data'])
    draw_sub_box(ax, 0.5, 5.7, 2.4, 0.45, 'Train 3717 | Valid 112 | Test 200')
    draw_sub_box(ax, 0.5, 5.0, 1.15, 0.5, 'crack-seg\n数据集')
    draw_sub_box(ax, 1.75, 5.0, 1.15, 0.5, '数据增强\n预处理')

    # 2. 训练层
    draw_parent_box(ax, 3.4, y_base, 2.8, h_main, '训练层 (Training Layer)', colors['train'])
    draw_sub_box(ax, 3.6, 5.7, 2.4, 0.45, 'mAP / Loss 曲线监控')
    draw_sub_box(ax, 3.6, 5.0, 1.15, 0.5, 'YOLOv8-seg\n模型训练')
    draw_sub_box(ax, 4.85, 5.0, 1.15, 0.5, '100 Epochs\n验证与早停')

    # 3. 推理层
    draw_parent_box(ax, 6.5, y_base, 2.8, h_main, '推理层 (Inference Layer)', colors['infer'])
    draw_sub_box(ax, 6.7, 5.7, 2.4, 0.45, 'NMS + 置信度过滤')
    draw_sub_box(ax, 6.7, 5.0, 1.15, 0.5, '单张/批量\n图片检测')
    draw_sub_box(ax, 7.95, 5.0, 1.15, 0.5, '实时视频\n摄像头检测')

    # 4. 应用层
    draw_parent_box(ax, 9.6, y_base, 2.8, h_main, '应用层 (Application Layer)', colors['app'])
    draw_sub_box(ax, 9.8, 5.7, 2.4, 0.45, '可视化 + 报告生成')
    draw_sub_box(ax, 9.8, 5.0, 1.15, 0.5, 'CLI 命令行\nmain.py')
    draw_sub_box(ax, 11.05, 5.0, 1.15, 0.5, 'Web 界面\nGradio')

    # 5. 部署层 (重点调整，防止与标题重叠)
    draw_parent_box(ax, 12.7, y_base, 3.0, h_main, '部署层 (Deploy Layer)', colors['deploy'])
    # Docker 框下移至 5.9，与标题 (位于 6.8) 拉开距离
    draw_sub_box(ax, 12.9, 5.9, 2.6, 0.45, 'Docker 容器 / 边缘设备 (Jetson/RPI)')
    # 其他小框同步下移
    draw_sub_box(ax, 12.9, 5.3, 1.25, 0.45, 'ONNX / TensorRT')
    draw_sub_box(ax, 14.25, 5.3, 1.25, 0.45, 'OpenVINO / TFLite')
    draw_sub_box(ax, 12.9, 4.8, 2.6, 0.4, '模型量化 (FP16/INT8)', fontsize=8)

    # 层间箭头
    arrow_y = y_base + h_main/2
    for x1, x2 in [(3.1, 3.4), (6.2, 6.5), (9.3, 9.6), (12.4, 12.7)]:
        draw_arrow(ax, x1, arrow_y, x2, arrow_y)

    # --- 第二行：技术流程 ---
    flow_steps = [
        ('输入图像', 1.5, colors['data']),
        ('预处理\n640×640', 4.5, colors['data']),
        ('Backbone\nCSPDarknet', 7.5, colors['train']),
        ('Neck\nPAN-FPN', 10.5, colors['train']),
        ('Head\nSegmentation', 13.5, colors['infer']),
    ]
    for i, (text, x, c) in enumerate(flow_steps):
        box = FancyBboxPatch((x-1.0, 2.8), 2.0, 0.8, boxstyle="round,pad=0.1",
                             facecolor=c, edgecolor='#2C3E50', linewidth=1)
        ax.add_patch(box)
        ax.text(x, 3.2, text, ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        if i < len(flow_steps) - 1:
            draw_arrow(ax, x + 1.1, 3.2, x + 1.9, 3.2)

    # 底部说明
    ax.text(8, 2.0, '输出结果: 边界框(BBox) + 分割掩码(Mask) + 裂缝面积占比 + 严重程度评级',
            ha='center', va='center', fontsize=11, color='#34495E',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDEBD0', edgecolor='#E8913A'))

    # 保存
    plt.tight_layout()
    plt.savefig("final_architecture.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("修改后的总体框图已生成：final_architecture.png")

if __name__ == "__main__":
    draw_optimized_architecture()
