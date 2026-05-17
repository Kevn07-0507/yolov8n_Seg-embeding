"""
YOLOv8n-seg 训练与推理流程框图 — PPT 专用
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import matplotlib.font_manager as fm
import numpy as np
import os

# 中文字体
for fname in ['Microsoft YaHei', 'SimHei']:
    hits = [f for f in fm.fontManager.ttflist if fname in f.name]
    if hits:
        plt.rcParams['font.family'] = hits[0].name
        break
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = "报告"
os.makedirs(OUT_DIR, exist_ok=True)

# ═══ 颜色方案 ═══
C = {
    'bg':       '#F8F9FA',
    'data':     '#4A90D9',   # 蓝 - 数据
    'aug':      '#5DADE2',   # 浅蓝 - 增强
    'backbone': '#2ECC71',   # 绿 - Backbone
    'neck':     '#27AE60',   # 深绿 - Neck
    'head':     '#E67E22',   # 橙 - Head
    'loss':     '#E74C3C',   # 红 - Loss
    'post':     '#9B59B6',   # 紫 - 后处理
    'output':   '#1ABC9C',   # 青 - 输出
    'arrow':    '#7F8C8D',
    'text':     '#2C3E50',
    'white':    '#FFFFFF',
    'border':   '#34495E',
}


def draw_rounded_box(ax, x, y, w, h, facecolor, edgecolor='#34495E', lw=1.5, pad=0.1):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=lw)
    ax.add_patch(box)
    return box


def draw_text(ax, x, y, w, h, text, color='white', fontsize=9, bold=True):
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold' if bold else 'normal', color=color)


def draw_arrow_right(ax, x1, y1, x2, y2, color='#7F8C8D', lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


def draw_arrow_down(ax, x1, y1, x2, y2, color='#7F8C8D', lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


# ═══════════════════════════════════════════
# 图1：YOLOv8n-seg 训练流程
# ═══════════════════════════════════════════
def draw_training_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_facecolor(C['bg'])

    # ── 标题 ──
    ax.text(7.5, 7.7, 'YOLOv8n-seg 模型训练流程', ha='center', fontsize=18, fontweight='bold', color=C['text'])

    # ── 第一行：数据流 ──
    y1 = 5.6
    # 数据集
    draw_rounded_box(ax, 0.4, y1, 2.0, 1.2, C['data'])
    draw_text(ax, 0.4, y1, 2.0, 0.7, 'crack-seg 数据集', fontsize=10)
    draw_text(ax, 0.4, y1+0.2, 2.0, 0.65, 'Train: 3717\nValid: 112  Test: 200', color='#E8F0FE', fontsize=7, bold=False)
    draw_arrow_right(ax, 2.4, y1+0.6, 3.1, y1+0.6)

    # 数据增强
    draw_rounded_box(ax, 3.2, y1, 2.2, 1.2, C['aug'])
    draw_text(ax, 3.2, y1, 2.2, 0.5, '数据增强', fontsize=10)
    augs = 'Mosaic | HSV | Flip | Scale | Translate'
    draw_text(ax, 3.2, y1+0.1, 2.2, 0.5, augs, color=C['white'], fontsize=6, bold=False)

    # 模型输入
    draw_arrow_right(ax, 5.4, y1+0.6, 6.1, y1+0.6)
    draw_rounded_box(ax, 6.2, y1, 1.8, 1.2, C['data'])
    draw_text(ax, 6.2, y1, 1.8, 0.6, '输入张量', fontsize=10)
    draw_text(ax, 6.2, y1+0.15, 1.8, 0.55, 'Batch × 3 × 640 × 640', color='#E8F0FE', fontsize=7, bold=False)

    # ── 第二行：模型主干 ──
    y2 = 3.5
    # Backbone
    backbone_w = 3.0
    draw_rounded_box(ax, 0.5, y2, backbone_w, 1.8, C['backbone'])
    draw_text(ax, 0.5, y2, backbone_w, 0.55, 'Backbone: CSPDarknet', fontsize=10)
    layers = 'Conv → C2f → Conv → C2f → Conv → C2f\n→ Conv → SPPF → PSA'
    draw_text(ax, 0.5, y2-0.1, backbone_w, 1.1, layers, color=C['white'], fontsize=7, bold=False)

    draw_arrow_right(ax, 3.5, y2+0.9, 4.1, y2+0.9)

    # Neck
    neck_w = 3.0
    draw_rounded_box(ax, 4.2, y2, neck_w, 1.8, C['neck'])
    draw_text(ax, 4.2, y2, neck_w, 0.55, 'Neck: PAN-FPN', fontsize=10)
    neck_layers = '自上而下 FPN + 自下而上 PAN\n多尺度特征融合 (P3/P4/P5)'
    draw_text(ax, 4.2, y2-0.1, neck_w, 1.1, neck_layers, color=C['white'], fontsize=7, bold=False)

    draw_arrow_right(ax, 7.2, y2+0.9, 7.8, y2+0.9)

    # Head
    head_w = 3.5
    draw_rounded_box(ax, 7.9, y2, head_w, 1.8, C['head'])
    draw_text(ax, 7.9, y2, head_w, 0.55, 'Head: Segment Head', fontsize=10)
    head_layers = 'Proto-mask 分支 (32个原型掩码)\n+ BBox 回归 + Cls 分类'
    draw_text(ax, 7.9, y2-0.1, head_w, 1.1, head_layers, color=C['white'], fontsize=7, bold=False)

    # 箭头连接 Backbone→Neck→Head
    draw_arrow_right(ax, 11.4, y2+0.9, 12.0, y2+0.9)

    # Loss
    loss_w = 2.5
    draw_rounded_box(ax, 12.1, y2, loss_w, 1.8, C['loss'])
    draw_text(ax, 12.1, y2, loss_w, 0.55, 'Loss 计算与反向传播', fontsize=10)
    loss_text = 'Seg Loss + BBox Loss + Cls Loss\n→ 梯度反向传播 → 参数更新'
    draw_text(ax, 12.1, y2-0.1, loss_w, 1.1, loss_text, color=C['white'], fontsize=7, bold=False)

    # ── 第三行：输出 ──
    y3 = 1.8
    draw_rounded_box(ax, 4.5, y3, 6.0, 1.0, C['output'])
    draw_text(ax, 4.5, y3, 6.0, 1.0, '训练输出: best.pt + last.pt + 训练曲线(loss/mAP) + 验证指标', fontsize=10)

    # 循环箭头 Loss → 数据增强
    ax.annotate('', xy=(0.4, y3+0.5), xytext=(13.35, y2+0.9),
                arrowprops=dict(arrowstyle='->', color=C['loss'], lw=2,
                                connectionstyle="arc3,rad=0.5", linestyle='dashed'))
    ax.text(10.0, 4.8, 'Epoch 循环 (100轮)', fontsize=9, color=C['loss'], fontweight='bold',
            rotation=0, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', edgecolor=C['loss'], alpha=0.9))

    # 竖向箭头：数据层→模型层、模型层→输出层
    draw_arrow_down(ax, 7.3, y1, 7.3, y2+1.8, color=C['arrow'], lw=1.5)
    draw_arrow_down(ax, 7.5, y2, 7.5, y3+1.0, color=C['arrow'], lw=1.5)

    # ── 参数标注 ──
    params_y = 0.8
    params_text = '关键参数: epochs=100 | imgsz=640 | batch=16 | lr0=0.01 | AMP混合精度 | patience=20(早停)'
    ax.text(7.5, params_y, params_text, ha='center', fontsize=9, color='#7F8C8D',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=C['white'], edgecolor='#BDC3C7'))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "ppt_yolo_train.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


# ═══════════════════════════════════════════
# 图2：YOLOv8n-seg 推理流程（单张图片）
# ═══════════════════════════════════════════
def draw_inference_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis('off')
    ax.set_facecolor(C['bg'])

    ax.text(7, 5.2, 'YOLOv8n-seg 推理流程 (单张图片)', ha='center', fontsize=17, fontweight='bold', color=C['text'])

    # ── 流程步骤 ──
    y_main = 2.5
    y_sub = 1.0
    y_label = 3.6

    steps = [
        ('Input', '输入图像\n.jpg/.png', '图像采集', C['data']),
        ('Preprocess', 'Resize 640×640\nNormalize\nBGR→RGB', '预处理', C['aug']),
        ('Backbone', 'CSPDarknet\n特征提取\n5个Stage', '骨干网络', C['backbone']),
        ('Neck', 'PAN-FPN\n多尺度融合\nP3/P4/P5', '特征融合', C['neck']),
        ('Head', 'Segment Head\nBBox + Mask\n+ Cls 分支', '检测头', C['head']),
        ('Postprocess', 'NMS 非极大抑制\n置信度过滤\nMask 后处理', '后处理', C['post']),
        ('Output', '可视化结果\n裂缝面积 & 严重度\n标注图', '输出', C['output']),
    ]

    box_w = 1.55
    gap = 0.3
    for i, (title, desc, label, color) in enumerate(steps):
        x = 0.3 + i * (box_w + gap)
        # 主框
        draw_rounded_box(ax, x, y_main, box_w, 1.2, color)
        draw_text(ax, x, y_main, box_w, 1.2, title + '\n\n', fontsize=8)

        # 描述子框
        draw_rounded_box(ax, x, y_sub, box_w, 0.85, C['white'], edgecolor=color, lw=1.2)
        draw_text(ax, x, y_sub, box_w, 0.85, desc, color=C['text'], fontsize=6.5, bold=False)

        # 标签
        ax.text(x + box_w/2, y_label, label, ha='center', fontsize=8, color=color, fontweight='bold')

        # 箭头
        if i < len(steps) - 1:
            draw_arrow_right(ax, x + box_w + 0.02, y_main + 0.6,
                             x + box_w + gap - 0.02, y_main + 0.6, color=C['arrow'])

    # ── 底部说明 ──
    y_bot = 0.25
    info = '模型规格: YOLOv8n-seg (nano) | 参数量: 3.4M | 模型大小: 6.7MB | 推理速度: ~10ms/image (GPU) | FPS: ~100'
    ax.text(7, y_bot, info, ha='center', fontsize=9, color='#7F8C8D',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=C['white'], edgecolor='#BDC3C7'))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "ppt_yolo_inference.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


# ═══════════════════════════════════════════
# 图3：YOLOv8n 网络结构简图
# ═══════════════════════════════════════════
def draw_network_structure():
    fig, ax = plt.subplots(1, 1, figsize=(14, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    ax.set_facecolor(C['bg'])

    ax.text(7, 7.2, 'YOLOv8n-seg 网络结构简图', ha='center', fontsize=17, fontweight='bold', color=C['text'])

    # ── Input ──
    y = 6.0
    draw_rounded_box(ax, 0.3, y, 2.0, 0.8, C['data'])
    draw_text(ax, 0.3, y, 2.0, 0.8, 'Input\n640×640×3', fontsize=9)

    # ── Backbone 各 Stage ──
    backbone_stages = [
        ('Stage 0', 'Conv(k3,s2)\n320×320×16', 2.6, C['backbone']),
        ('Stage 1', 'Conv(k3,s2)\n160×160×32', 5.0, C['backbone']),
        ('Stage 2', 'C2f(n=3)\n80×80×64', 7.4, C['backbone']),
        ('Stage 3', 'C2f(n=3)\n40×40×128', 9.8, C['backbone']),
        ('Stage 4', 'C2f(n=1) + SPPF\n20×20×256', 12.2, C['backbone']),
    ]
    stage_positions = []
    for title, desc, bx, color in backbone_stages:
        draw_rounded_box(ax, bx, y-0.7, 2.1, 0.65, color)
        draw_text(ax, bx, y-0.7, 2.1, 0.65, title + '  ' + desc, fontsize=6.5, bold=True)
        stage_positions.append((bx + 1.05, y-0.7))

    # Backbone label
    ax.text(7, y+0.1, 'Backbone', ha='center', fontsize=11, fontweight='bold', color=C['backbone'])

    # ── Neck: FPN + PAN ──
    neck_y = 3.5
    # FPN (top-down) - 虚线
    for i, (title, _, bx, _) in enumerate(backbone_stages[2:], 2):
        ny = neck_y + (4-i) * 0.75
        # draw box
        if i == 2:
            label = 'P3 (80×80)'
        elif i == 3:
            label = 'P4 (40×40)'
        else:
            label = 'P5 (20×20)'
        draw_rounded_box(ax, bx, ny, 2.1, 0.55, C['neck'])
        draw_text(ax, bx, ny, 2.1, 0.55, label, fontsize=7)

    # FPN label
    ax.text(13.5, 4.8, 'FPN\n(top-down)', ha='center', fontsize=8, color=C['neck'], fontweight='bold')

    # PAN 上采样箭头 (简化)
    for i in range(2, 5):
        ax.annotate('', xy=(backbone_stages[i][2] + 0.3, neck_y + (4-i)*0.75 + 0.55),
                     xytext=(backbone_stages[i-1][2] + 0.3, neck_y + (5-i)*0.75),
                     arrowprops=dict(arrowstyle='->', color=C['neck'], lw=1.2, linestyle='dashed'))

    # Neck label
    ax.text(7, 5.3, 'Neck (PAN-FPN)', ha='center', fontsize=11, fontweight='bold', color=C['neck'])

    # ── Head ──
    head_y = 1.8
    head_items = [
        ('BBox Head', 2.6, C['head']),
        ('Cls Head', 7.0, C['head']),
        ('Mask Proto', 11.5, C['head']),
    ]
    for title, hx, color in head_items:
        draw_rounded_box(ax, hx, head_y, 2.0, 0.8, color)
        draw_text(ax, hx, head_y, 2.0, 0.8, title + '\nConv+Conv2d', fontsize=7.5)

    ax.text(7, 2.8, 'Head (Segment)', ha='center', fontsize=11, fontweight='bold', color=C['head'])

    # ── Output ──
    out_y = 0.5
    outputs = [
        ('BBoxes\n[x1,y1,x2,y2]', 0.8, C['output']),
        ('Classes\n[crack: 1]', 4.2, C['output']),
        ('Masks\n160×160×32 → H×W', 7.6, C['output']),
        ('NMS + Filter\n→ Final Result', 11.0, C['output']),
    ]
    for title, ox, _ in outputs:
        draw_rounded_box(ax, ox, out_y, 2.8, 0.7, C['output'])
        draw_text(ax, ox, out_y, 2.8, 0.7, title, fontsize=7)

    # ── 连接线 (简化) ──
    draw_arrow_down(ax, 7.2, 6.0, 7.2, 5.6, C['arrow'])

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "ppt_yolo_network.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


if __name__ == '__main__':
    print("Generating PPT diagrams...")
    p1 = draw_training_pipeline()
    p2 = draw_inference_pipeline()
    p3 = draw_network_structure()
    print("Done.")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")
