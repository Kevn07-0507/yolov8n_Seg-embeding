"""
详细模型评估脚本 — 输出逐 IoU 步进的 AP 数据 + 绘图
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from ultralytics import YOLO
from pathlib import Path

# 中文字体
for fname in ['Microsoft YaHei', 'SimHei']:
    hits = [f for f in fm.fontManager.ttflist if fname in f.name]
    if hits:
        plt.rcParams['font.family'] = hits[0].name
        break
plt.rcParams['axes.unicode_minus'] = False


def _plot_ap_curves(rows, save_dir):
    """绘制 Box AP 和 Mask AP 的 IoU 曲线"""
    ious = [float(r['iou']) for r in rows]
    box_ap = [r.get('box_ap', 0) for r in rows]
    mask_ap = [r.get('mask_ap', 0) for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F8F9FA')

    ax.plot(ious, box_ap, 'o-', color='#3498DB', linewidth=2.5, markersize=8, label='Box AP')
    ax.plot(ious, mask_ap, 's-', color='#E74C3C', linewidth=2.5, markersize=8, label='Mask AP (Seg)')

    for i, (bx, mx) in enumerate(zip(box_ap, mask_ap)):
        ax.annotate(f'{bx:.3f}', (ious[i], bx), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=8, color='#3498DB')
        ax.annotate(f'{mx:.3f}', (ious[i], mx), textcoords="offset points",
                    xytext=(0, -16), ha='center', fontsize=8, color='#E74C3C')

    ax.set_xlabel('IoU 阈值', fontsize=13)
    ax.set_ylabel('Average Precision (AP)', fontsize=13)
    ax.set_title('AP vs IoU 曲线', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(1.0, max(box_ap + mask_ap) * 1.15))
    ax.set_xlim(0.47, 0.98)

    # mAP 标注
    box_mAP = np.mean(box_ap)
    mask_mAP = np.mean(mask_ap)
    ax.text(0.96, 0.06, f'Box mAP50-95: {box_mAP:.4f}\nMask mAP50-95: {mask_mAP:.4f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECF0F1', edgecolor='#BDC3C7'))

    plt.tight_layout()
    plot_path = str(save_dir / 'per_iou_ap_curve.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"AP 曲线已保存: {plot_path}")
    return plot_path


def detailed_val(model_path, data_yaml='crack-seg/data.yaml', workers=1):
    """运行详细验证并导出逐 IoU 的 AP 数据"""
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    print("正在详细评估...")
    metrics = model.val(data=data_yaml, workers=workers, plots=False, save_json=False)

    save_dir = Path(model.trainer.save_dir) if model.trainer and model.trainer.save_dir else Path('runs/segment/detailed_val')
    output = save_dir / 'per_iou_ap.csv'

    iou_thresholds = np.round(np.linspace(0.50, 0.95, 10), 2)

    rows = []
    cls_idx = 0

    for i, iou in enumerate(iou_thresholds):
        row = {'iou': f'{iou:.2f}'}

        if hasattr(metrics, 'box') and hasattr(metrics.box, 'all_ap'):
            ap_array = metrics.box.all_ap
            if ap_array.ndim == 3:
                row['box_ap'] = round(float(ap_array[cls_idx, i, :].mean()), 6)
            elif ap_array.ndim == 2:
                row['box_ap'] = round(float(ap_array[cls_idx, i]), 6)
            else:
                row['box_ap'] = round(float(ap_array[i]), 6)

        if hasattr(metrics, 'seg') and hasattr(metrics.seg, 'all_ap'):
            ap_array = metrics.seg.all_ap
            if ap_array.ndim == 3:
                row['mask_ap'] = round(float(ap_array[cls_idx, i, :].mean()), 6)
            elif ap_array.ndim == 2:
                row['mask_ap'] = round(float(ap_array[cls_idx, i]), 6)
            else:
                row['mask_ap'] = round(float(ap_array[i]), 6)

        rows.append(row)

    # ── CSV ──
    os.makedirs(str(save_dir), exist_ok=True)
    with open(str(output), 'w', encoding='utf-8-sig') as f:
        f.write('iou_threshold,box_ap,mask_ap\n')
        for row in rows:
            f.write(f"{row['iou']},{row.get('box_ap', '')},{row.get('mask_ap', '')}\n")
    print(f"CSV 已导出: {output}")

    # ── 表格 ──
    print(f"\n{'IoU':>6}  {'Box AP':>8}  {'Mask AP':>8}")
    print("-" * 28)
    for row in rows:
        bap = row.get('box_ap', '')
        map_ = row.get('mask_ap', '')
        print(f"{row['iou']:>6}  {bap:>8}  {map_:>8}")

    box_mean = np.mean([r.get('box_ap', 0) for r in rows])
    mask_mean = np.mean([r.get('mask_ap', 0) for r in rows])
    print(f"\nmAP50:  {rows[0].get('box_ap', 0):.4f} (Box)  {rows[0].get('mask_ap', 0):.4f} (Mask)")
    print(f"mAP50-95 (10-step mean):  {box_mean:.4f} (Box)  {mask_mean:.4f} (Mask)")

    # ── 绘图 ──
    _plot_ap_curves(rows, save_dir)

    return str(output)


if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else input("请输入模型路径: ").strip()
    data_yaml = sys.argv[2] if len(sys.argv) > 2 else 'crack-seg/data.yaml'

    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        sys.exit(1)

    detailed_val(model_path, data_yaml)
