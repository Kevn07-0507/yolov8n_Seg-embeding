"""
墙面裂缝分割模型训练脚本
使用YOLOv8进行裂缝检测和分割
两阶段训练 + 差分学习率 + 高分辨率掩码
"""
from ultralytics import YOLO
import torch
import sys
from pathlib import Path

_VENV_PYTHON = r'E:\torch\.venv\Scripts\python.exe'


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    if device == 'cpu':
        print("\n" + "!" * 60)
        print("  警告: 未检测到 CUDA，将使用 CPU 训练（速度极慢）")
        print(f"  当前 Python: {sys.executable}")
        print(f"  建议使用 venv Python 运行本项目")
        print("!" * 60 + "\n")
        if sys.executable != _VENV_PYTHON:
            print(f"  尝试: {_VENV_PYTHON} train_crack_seg.py\n")

    model = YOLO('yolov8n-seg.pt')

    if device == 'cuda':
        torch.cuda.empty_cache()
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # ═══ 共享训练参数 ═══
    common = dict(
        data='crack-seg/data.yaml',
        imgsz=640,
        batch=4,
        device=device,
        workers=2,
        project='runs/segment',
        patience=20,
        amp=True,
        save=True,
        plots=True,
        val=True,
        # 数据增强
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
        # 分割精度
        mask_ratio=8,
        retina_masks=True,
    )

    # ═══ Stage 1: 冻结 Backbone，训练 Head ═══
    print("\n" + "=" * 60)
    print("Stage 1/2: 冻结 Backbone (freeze=10)，训练 Neck+Head")
    print("  学习率: lr0=0.01 (高)  |  30 epochs")
    print("=" * 60)
    model.train(
        epochs=30,
        lr0=0.01,
        freeze=10,
        name='crack_seg_ft_stage1',
        **common,
    )

    # ═══ Stage 2: 重新加载 Stage1 权重，解冻全部微调 ═══
    stage1_best = str(Path(model.trainer.save_dir) / 'weights' / 'best.pt')
    print(f"\nStage 1 完成，加载权重: {stage1_best}")
    model = YOLO(stage1_best)

    print("\n" + "=" * 60)
    print("Stage 2/2: 解冻全部 (freeze=0)，全模型微调")
    print("  学习率: lr0=0.001 (低)  |  70 epochs")
    print("=" * 60)
    results = model.train(
        epochs=70,
        lr0=0.001,
        freeze=0,
        name='crack_seg_ft_stage2',
        **common,
    )

    print("\n训练完成!")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    print(f"最后模型保存在: {results.save_dir}/weights/last.pt")


if __name__ == '__main__':
    main()
