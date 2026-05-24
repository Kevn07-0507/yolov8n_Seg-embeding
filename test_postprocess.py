"""
TTA + CRF 推理测试
"""
import os
import sys
import cv2
from ultralytics import YOLO
from postprocess import detect_with_postprocess


def main():
    model_path = input("模型路径 (默认 runs/segment/crack_seg/weights/best.pt): ").strip()
    if not model_path:
        model_path = "runs/segment/crack_seg/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        return

    image_path = input("图片路径: ").strip()
    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        return

    model = YOLO(model_path)

    print("\n推理中...")
    # note: Ultralytics 8.3.x seg 不支持 augment，TTA 已跳过
    results, refined, annotated = detect_with_postprocess(
        model, image_path, use_tta=False, use_crf=True
    )

    r = results[0]
    num = len(r.boxes) if r.boxes else 0
    print(f"检测到 {num} 条裂缝")

    # 保存
    save_dir = "postprocess_results"
    os.makedirs(save_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(f"{save_dir}/{basename}_tta_crf.jpg", annotated)
    print(f"结果保存至: {save_dir}/{basename}_tta_crf.jpg")


if __name__ == "__main__":
    main()
