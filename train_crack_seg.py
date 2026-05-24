"""
墙面裂缝分割模型训练脚本
YOLOv8n-seg 笔记本训练
"""
from ultralytics import YOLO
import torch
import sys
import os

_VENV_PYTHON = r'E:\torch\.venv\Scripts\python.exe'


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    if device == 'cpu':
        print("\n" + "!" * 60)
        print("  警告: 未检测到 CUDA，将使用 CPU 训练（速度极慢）")
        print(f"  当前 Python: {sys.executable}")
        print("!" * 60 + "\n")

    model = YOLO('yolov8n-seg.pt')

    if device == 'cuda':
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    print("\n" + "=" * 60)
    print("YOLOv8n-seg 训练")
    print("  epochs=200  batch=4  imgsz=640")
    print("=" * 60)

    results = model.train(
        data='crack-seg/data.yaml',
        epochs=200,
        imgsz=640,
        batch=4,
        device=device,
        workers=2,
        project='runs/segment',
        name='crack_seg',
        patience=20,
        amp=True,
        save=True,
        plots=True,
        val=True,
        optimizer='auto',
        lr0=0.01,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        mask_ratio=4,
    )

    print("\n训练完成!")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    print(f"最后模型保存在: {results.save_dir}/weights/last.pt")


if __name__ == '__main__':
    main()
