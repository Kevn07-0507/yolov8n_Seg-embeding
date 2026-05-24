"""
推理后处理模块 — TTA + CRF 掩码精炼
"""
import numpy as np
import cv2

# ─── TTA 推理 ───
def predict_tta(model, image_path, conf=0.25, iou=0.7):
    """
    YOLOv8 内置 TTA 推理（augment=True）
    model.predict() 自动：flip + scale + 多结果融合
    """
    return model.predict(source=image_path, conf=conf, iou=iou, augment=True, verbose=False)


# ─── CRF 掩码精炼 (pydensecrf 版) ───
def _crf_densecrf(image, mask, sdims=3, schan=10, compat=10, iters=5):
    """
    使用 pydensecrf 对单张掩码做条件随机场精炼
    若 pydensecrf 不可用则回退到 opencv 模式
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        return _crf_opencv(image, mask)

    h, w = image.shape[:2]
    mask = (mask > 127).astype(np.float32)
    softmax = np.stack([1 - mask, mask], axis=2)
    softmax = softmax.transpose(2, 0, 1).reshape(2, -1)
    softmax = np.ascontiguousarray(softmax)

    img = np.ascontiguousarray(image)
    d = dcrf.DenseCRF2D(w, h, 2)
    unary = unary_from_softmax(softmax)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=sdims, compat=compat)
    d.addPairwiseBilateral(sxy=schan, srgb=13, rgbim=img, compat=compat)
    Q = d.inference(iters)
    refined = np.argmax(Q, axis=0).reshape(h, w).astype(np.uint8) * 255
    return refined


# ─── GrabCut 掩码精炼 (Windows 兼容版) ───
def _crf_opencv(image, mask, iters=3):
    """
    使用 GrabCut 对单张掩码做边缘精炼
    用预测掩码作为 Seeds，GrabCut 迭代将掩码边界吸附到原图裂缝边缘

    Args:
        image: 原始 RGB 图像 (H, W, 3)
        mask:  预测掩码灰度图 (H, W), 0-255
        iters: GrabCut 迭代次数 (1~5, 越大越贴合但越慢)
    """
    h, w = image.shape[:2]
    mask_bin = (mask > 127).astype(np.uint8)

    # ── 构建 GrabCut seeds ──
    # GC_BGD=0, GC_FGD=1, GC_PR_BGD=2, GC_PR_FGD=3
    # BG:   确定性背景 (远离裂缝)
    # PR_BG: 可能背景 (掩码膨胀 3px 减原掩码，给算法一些探索空间)
    # PR_FG: 可能前景 (掩码本身)
    # FG:   确定性前景 (掩码腐蚀 2px，核心裂缝区域)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    fg_definite = cv2.erode(mask_bin, kernel3, iterations=1)     # 确定性前景: 掩码腐蚀
    fg_probable = mask_bin                                        # 可能前景: 全部掩码
    bg_dilated = cv2.dilate(fg_probable, kernel5, iterations=2)   # 外面一圈

    gc_mask = np.zeros((h, w), dtype=np.uint8)
    gc_mask[fg_definite == 1] = cv2.GC_FGD       # = 1
    gc_mask[fg_probable == 1] = cv2.GC_PR_FGD    # = 3 (definite 区域保持 1, 不覆盖)
    # fg_probable 覆盖了 fg_definite, 但填 GC_PR_FGD 前先给了 GC_FGD
    # 再写入 GC_FGD 恢复核心区域
    gc_mask[fg_definite == 1] = cv2.GC_FGD
    gc_mask[(bg_dilated == 1) & (fg_probable == 0)] = cv2.GC_PR_BGD  # = 2 膨胀环(非裂缝区)

    # GrabCut 需要 CV_8UC3
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_MASK)

    # 提取前景: GC_FGD (1) + GC_PR_FGD (3)
    result = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # 后处理: 去噪 + 闭运算连接
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel3, iterations=1)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel3, iterations=1)

    return result


def refine_masks(result, crf_sdims=3, crf_schan=10, crf_compat=10, crf_iters=5):
    """
    对一个 YOLO predict result 中的所有掩码做 CRF 精炼
    返回 (refined_masks, 带精炼掩码的标注图)
    """
    if result.masks is None or result.masks.data is None:
        return [], result.plot()

    original_img = result.orig_img.copy()
    h, w = original_img.shape[:2]
    refined_list = []

    # 先画 box
    annotated = original_img.copy()
    if result.boxes is not None:
        for box in result.boxes.xyxy.cpu().numpy().astype(int):
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # 对每个掩码做 CRF 精炼后叠加
    for i, mask_tensor in enumerate(result.masks.data):
        mask = mask_tensor.cpu().numpy()
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h))
        mask = (mask * 255).astype(np.uint8)

        refined = _crf_densecrf(original_img, mask,
                                sdims=crf_sdims, schan=crf_schan,
                                compat=crf_compat, iters=crf_iters)
        refined_list.append(refined)

        # 半透明叠加
        overlay = annotated.copy()
        overlay[refined > 127] = (0, 0, 255)
        cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

    return refined_list, annotated


def detect_with_postprocess(model, image_path, conf=0.25, iou=0.7,
                            use_tta=True, use_crf=True,
                            crf_sdims=3, crf_schan=10, crf_compat=10, crf_iters=5):
    """
    完整推理管线: 可选 TTA + 可选 CRF

    Args:
        model: YOLO 模型
        image_path: 图片路径
        conf, iou: 置信度/IOU 阈值
        use_tta: 是否使用 TTA
        use_crf: 是否使用 CRF 精炼
        crf_*: CRF 参数

    Returns:
        (results, refined_masks, annotated_img)
    """
    if use_tta:
        results = predict_tta(model, image_path, conf, iou)
    else:
        results = model.predict(source=image_path, conf=conf, iou=iou, verbose=False)

    r = results[0]
    refined_masks = []
    annotated = r.plot()

    if use_crf and r.masks is not None and len(r.masks) > 0:
        refined_masks, annotated = refine_masks(r, crf_sdims, crf_schan, crf_compat, crf_iters)

    return results, refined_masks, annotated
