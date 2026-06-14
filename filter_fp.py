"""
误检过滤器 — 图像纹理 + 几何混合
窗框=直线均匀  图案=规律重复  裂缝=杂乱不规则
"""
import cv2
import numpy as np
from ultralytics import YOLO


def _crack_score(img_patch):
    """
    分析原图 ROI 区域的"裂缝特征分数"
    分数越高越像裂缝，越低越像窗框/图案
    """
    gray = cv2.cvtColor(img_patch, cv2.COLOR_RGB2GRAY) if img_patch.ndim == 3 else img_patch

    # ── 1. 边缘密度 ──
    edges = cv2.Canny(gray, 50, 150)
    edge_pct = np.count_nonzero(edges) / gray.size

    # ── 2. 边缘方向熵 (裂缝方向杂, 窗框/图案方向集中) ──
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    strong = mag > 20
    if np.sum(strong) > 10:
        dirs = np.arctan2(gy[strong], gx[strong]) * 180 / np.pi
        hist, _ = np.histogram(dirs, bins=12, range=(-90, 90))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist)) / np.log(12)  # 归一化熵 0~1
    else:
        entropy = 0

    # ── 3. 局部方差 (裂缝纹理粗糙, 窗框光滑) ──
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    local_var = cv2.Laplacian(blur, cv2.CV_64F).var() / 1000.0

    # ── 4. 宽高比 ──
    h, w = gray.shape[:2]
    aspect = max(h, w) / (min(h, w) + 1)

    # ── 综合评分 ──
    score = 0.0
    score += min(edge_pct * 10, 1.0) * 0.30    # 边缘密度 (裂缝高)
    score += entropy * 0.40                      # 方向熵 (裂缝高)
    score += min(local_var, 1.0) * 0.25         # 纹理方差 (裂缝高)
    score -= min(aspect / 20, 0.3)              # 细长惩罚 (窗框)

    return score


def filter_detections(results, conf=0.25, iou=0.7, score_thresh=0.25, verbose=True):
    """
    混和过滤: 图像纹理 + 几何
    """
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        keep = []
        original_img = r.orig_img

        for i, (box, conf_val) in enumerate(zip(r.boxes.xyxy, r.boxes.conf)):
            x1, y1, x2, y2 = [int(v) for v in box]
            bw, bh = x2 - x1, y2 - y1

            # 裁剪 ROI，分析原图纹理
            x1c = max(0, x1); y1c = max(0, y1)
            x2c = min(original_img.shape[1], x2); y2c = min(original_img.shape[0], y2)
            roi = original_img[y1c:y2c, x1c:x2c]

            if roi.size == 0:
                keep.append(i)
                continue

            score = _crack_score(roi)
            conf_val = conf_val.item()

            # 高置信度直接保留 (模型很确定是裂缝)
            # 低置信度 + 低纹理分数 → 剔除
            if conf_val >= 0.5 or score >= score_thresh:
                keep.append(i)
            else:
                if verbose:
                    print(f"  [Filtered] conf={conf_val:.2f} score={score:.2f}: 纹理不似裂缝")

        if keep:
            r.boxes = r.boxes[keep]
            if r.masks is not None:
                r.masks.data = r.masks.data[keep]
        else:
            r.boxes = None
            r.masks = None

    return results


class FilteredDetector:
    def __init__(self, model_path='runs/segment/crack_seg4/weights/best.pt'):
        self.model = YOLO(model_path)

    def detect(self, image_path, conf=0.25, iou=0.7, score_thresh=0.25):
        raw = self.model.predict(source=image_path, conf=conf, iou=iou, verbose=False)
        filtered = filter_detections(raw, conf=conf, iou=iou, verbose=True, score_thresh=score_thresh)
        if filtered[0].boxes is not None:
            annotated = filtered[0].plot()
        else:
            annotated = raw[0].orig_img.copy()
        return raw, filtered, annotated


if __name__ == '__main__':
    import os, matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    mp = input("模型路径 (默认 best.pt): ").strip() or 'runs/segment/crack_seg4/weights/best.pt'
    ip = input("图片路径: ").strip()
    if not os.path.exists(ip):
        print("文件不存在"); exit()

    detector = FilteredDetector(mp)
    raw, filtered, annotated = detector.detect(ip)

    fr = filtered[0]
    n_raw = len(raw[0].boxes) if raw[0].boxes else 0
    n_f = len(fr.boxes) if fr.boxes else 0
    print(f"\n原始: {n_raw} | 过滤后: {n_f} (剔除 {n_raw - n_f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(raw[0].plot()[..., ::-1])
    axes[0].set_title(f'原始 ({n_raw})'); axes[0].axis('off')
    axes[1].imshow(annotated[..., ::-1])
    axes[1].set_title(f'过滤后 ({n_f})'); axes[1].axis('off')
    plt.tight_layout(); plt.show()
