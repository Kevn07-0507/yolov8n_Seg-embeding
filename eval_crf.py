"""
CRF 后处理效果评估 — 验证集全量对比
"""
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from postprocess import detect_with_postprocess

for fname in ['Microsoft YaHei', 'SimHei']:
    hits = [f for f in fm.fontManager.ttflist if fname in f.name]
    if hits:
        plt.rcParams['font.family'] = hits[0].name
        break
plt.rcParams['axes.unicode_minus'] = False


def compute_iou(pred_mask, gt_mask):
    """计算单张掩码的 IoU"""
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)
    inter = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    return inter / union if union > 0 else 0.0


def evaluate_crf_vs_baseline(model_path, data_yaml='crack-seg/data.yaml',
                              output_dir='crf_evaluation', num_samples=50):
    """
    对比 baseline (原始推理) vs CRF 增强推理

    评估维度:
      - 每张图的掩码 IoU
      - 裂缝数量差异
      - 面积占比差异
      - 可视化对比图
    """
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    # 读取验证集图片
    val_images_dir = Path('crack-seg/valid/images')
    val_labels_dir = Path('crack-seg/valid/labels')
    val_images = sorted(list(val_images_dir.glob('*.jpg')))[:num_samples]

    print(f"评估 {len(val_images)} 张验证集图片...")

    results = []

    for img_path in tqdm(val_images, desc="CRF vs Baseline"):
        label_path = val_labels_dir / (img_path.stem + '.txt')
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # ── Baseline 推理 ──
        baseline_r = model.predict(source=str(img_path), conf=0.25, iou=0.5, verbose=False)[0]
        baseline_n = len(baseline_r.boxes) if baseline_r.boxes is not None else 0
        baseline_area = 0
        if baseline_r.masks is not None:
            for m in baseline_r.masks.data:
                mask = m.cpu().numpy()
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask = cv2.resize(mask, (w, h))
                baseline_area += (mask > 0.5).sum()
        baseline_area_pct = baseline_area / (h * w) * 100 if h * w > 0 else 0

        # ── CRF 增强推理 ──
        crf_r, refined, _ = detect_with_postprocess(model, str(img_path),
                                                    use_tta=False, use_crf=True)
        crf_r = crf_r[0]
        crf_n = len(crf_r.boxes) if crf_r.boxes is not None else 0
        crf_area = 0
        for m in refined:
            crf_area += (m > 127).sum()
        crf_area_pct = crf_area / (h * w) * 100 if h * w > 0 else 0

        # ── GT 统计 ──
        gt_n = 0
        gt_area = 0
        gt_mask_combined = np.zeros((h, w), dtype=np.uint8)
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    pts = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                    pts[:, 0] *= w
                    pts[:, 1] *= h
                    pts = pts.astype(np.int32)
                    cv2.fillPoly(gt_mask_combined, [pts], 255)
                    gt_n += 1
        gt_area = (gt_mask_combined > 127).sum()
        gt_area_pct = gt_area / (h * w) * 100 if h * w > 0 else 0

        # ── IoU 对比 ──
        # baseline 掩码合成
        baseline_mask = np.zeros((h, w), dtype=np.uint8)
        if baseline_r.masks is not None:
            for m in baseline_r.masks.data:
                mk = m.cpu().numpy()
                if mk.shape[0] != h or mk.shape[1] != w:
                    mk = cv2.resize(mk, (w, h))
                baseline_mask |= ((mk > 0.5) * 255).astype(np.uint8)

        # CRF 掩码合成
        crf_mask = np.zeros((h, w), dtype=np.uint8)
        for m in refined:
            crf_mask |= m

        iou_base = compute_iou(baseline_mask, gt_mask_combined) if gt_area > 0 else None
        iou_crf = compute_iou(crf_mask, gt_mask_combined) if gt_area > 0 else None

        results.append({
            'image': img_path.name,
            'gt_cracks': gt_n,
            'baseline_cracks': baseline_n,
            'crf_cracks': crf_n,
            'gt_area%': round(gt_area_pct, 3),
            'baseline_area%': round(baseline_area_pct, 3),
            'crf_area%': round(crf_area_pct, 3),
            'iou_baseline': round(iou_base, 4) if iou_base is not None else None,
            'iou_crf': round(iou_crf, 4) if iou_crf is not None else None,
        })

    # ═══ 汇总统计 ═══
    _plot_and_save(results, output_dir, num_samples)


def _plot_and_save(results, output_dir, num_samples):
    """生成图表并保存汇总 CSV"""
    iou_base = [r['iou_baseline'] for r in results if r['iou_baseline'] is not None]
    iou_crf = [r['iou_crf'] for r in results if r['iou_crf'] is not None]

    # ── 保存 CSV ──
    csv_path = os.path.join(output_dir, 'crf_vs_baseline.csv')
    with open(csv_path, 'w', encoding='utf-8-sig') as f:
        keys = ['image', 'gt_cracks', 'baseline_cracks', 'crf_cracks',
                'gt_area%', 'baseline_area%', 'crf_area%', 'iou_baseline', 'iou_crf']
        f.write(','.join(keys) + '\n')
        for r in results:
            f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
    print(f"CSV 已保存: {csv_path}")

    # ── 摘要 ──
    print("\n" + "=" * 55)
    print("CRF 后处理评估摘要")
    print("=" * 55)
    print(f"  评估图片: {num_samples}")
    print(f"  平均 IoU (Baseline): {np.mean(iou_base):.4f}" if iou_base else "  N/A")
    print(f"  平均 IoU (CRF):      {np.mean(iou_crf):.4f}" if iou_crf else "  N/A")
    if iou_base and iou_crf:
        delta = np.mean(iou_crf) - np.mean(iou_base)
        print(f"  IoU 提升:             {delta:+.4f}")
    area_base = [r['baseline_area%'] for r in results]
    area_crf = [r['crf_area%'] for r in results]
    area_gt = [r['gt_area%'] for r in results]
    print(f"  平均面积% Baseline: {np.mean(area_base):.3f}")
    print(f"  平均面积% CRF:      {np.mean(area_crf):.3f}")
    print(f"  平均面积% GT:       {np.mean(area_gt):.3f}")

    # ── 绘图 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CRF vs Baseline 验证集对比', fontsize=16, fontweight='bold')

    # 1. IoU 对比
    ax = axes[0, 0]
    if iou_base and iou_crf:
        x = np.arange(len(iou_base))
        ax.plot(x, sorted(iou_base), 's-', label=f'Baseline (avg={np.mean(iou_base):.3f})', color='#3498DB', markersize=3)
        ax.plot(x, sorted(iou_crf), 'o-', label=f'CRF (avg={np.mean(iou_crf):.3f})', color='#E74C3C', markersize=3)
        ax.set_xlabel('图片 (按IoU排序)')
        ax.set_ylabel('IoU')
        ax.set_title('掩码 IoU 对比 (vs GT)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. IoU 直方图
    ax = axes[0, 1]
    if iou_base and iou_crf:
        bins = np.linspace(0, 1, 21)
        ax.hist(iou_base, bins, alpha=0.5, label='Baseline', color='#3498DB')
        ax.hist(iou_crf, bins, alpha=0.5, label='CRF', color='#E74C3C')
        ax.set_xlabel('IoU')
        ax.set_ylabel('图片数')
        ax.set_title('IoU 分布对比')
        ax.legend()

    # 3. 裂缝数量对比
    ax = axes[1, 0]
    gt_n = [r['gt_cracks'] for r in results]
    base_n = [r['baseline_cracks'] for r in results]
    crf_n = [r['crf_cracks'] for r in results]
    ax.plot(gt_n, gt_n, '--', color='gray', alpha=0.5, label='GT=Pred (完美)')
    ax.scatter(gt_n, base_n, alpha=0.6, s=20, label='Baseline', color='#3498DB')
    ax.scatter(gt_n, crf_n, alpha=0.6, s=20, label='CRF', color='#E74C3C')
    ax.set_xlabel('GT 裂缝数')
    ax.set_ylabel('预测裂缝数')
    ax.set_title('裂缝数量对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 面积对比
    ax = axes[1, 1]
    ax.scatter(area_gt, area_base, alpha=0.6, s=20, label='Baseline', color='#3498DB')
    ax.scatter(area_gt, area_crf, alpha=0.6, s=20, label='CRF', color='#E74C3C')
    max_a = max(area_gt + area_base + area_crf) * 1.1
    ax.plot([0, max_a], [0, max_a], '--', color='gray', alpha=0.5, label='GT=Pred')
    ax.set_xlabel('GT 裂缝面积%')
    ax.set_ylabel('预测裂缝面积%')
    ax.set_title('裂缝面积占比对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'crf_vs_baseline.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"对比图已保存: {plot_path}")


if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else input("模型路径: ").strip()
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        sys.exit(1)
    n = input("评估图片数 (默认 50, 输入 all 为全部 200): ").strip()
    if n.lower() == 'all':
        n = 200
    else:
        n = int(n) if n else 50
    evaluate_crf_vs_baseline(model_path, num_samples=n)
