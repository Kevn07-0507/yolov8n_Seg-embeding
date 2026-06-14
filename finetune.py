"""
负样本微调脚本 — 加载 best.pt，在新增标注上微调
将 lizi/ + labels/ → 合并到 crack-seg/train/ → 低 lr 微调 20 epoch
"""
import shutil
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

# ═══ 配置 ═══
IMG_DIR   = Path('lizi')           # 新图片目录
LBL_DIR   = Path('labels')         # 新标注目录
BEST_PT   = Path('runs/segment/crack_seg4/weights/best.pt')
TRAIN_IMG = Path('crack-seg/train/images')
TRAIN_LBL = Path('crack-seg/train/labels')
DATA_YAML = Path('crack-seg/data.yaml')
EPOCHS    = 10
BATCH     = 2
LR        = 0.001
WORKERS   = 1
IMGSZ     = 640


def merge_data():
    """将新标注的图片和标签复制到训练集目录"""
    images = sorted(IMG_DIR.glob('*'))
    labels = sorted(LBL_DIR.glob('*.txt'))

    copied_img = 0
    copied_lbl = 0

    for img_path in images:
        if not img_path.suffix.lower().replace('jpeg', '.jpg') in ('.jpg', '.png', '.bmp'):
            continue
        lbl_path = LBL_DIR / (img_path.stem + '.txt')
        if not lbl_path.exists():
            print(f"  跳过 (无标注): {img_path.name}")
            continue

        # 复制图片
        dst_img = TRAIN_IMG / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
            copied_img += 1

        # 复制标签
        dst_lbl = TRAIN_LBL / lbl_path.name
        if not dst_lbl.exists():
            shutil.copy2(lbl_path, dst_lbl)
            copied_lbl += 1

    print(f"新增: {copied_img} 张图片, {copied_lbl} 个标签")
    return copied_img


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    print(f"模型: {BEST_PT}")

    if not BEST_PT.exists():
        print(f"模型不存在: {BEST_PT}")
        sys.exit(1)

    if not IMG_DIR.exists():
        print(f"图片目录不存在: {IMG_DIR}")
        sys.exit(1)

    # 1. 合并数据
    print("\n=== 1. 合并数据 ===")
    n = merge_data()
    if n == 0:
        print("数据已存在，直接进入训练")

    # 2. 加载模型
    print(f"\n=== 2. 微调训练 ({EPOCHS} epochs, lr={LR}) ===")
    model = YOLO(str(BEST_PT))

    if device == 'cuda':
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        workers=WORKERS,
        project='runs/segment',
        name='crack_seg_finetune',
        patience=10,
        amp=True,
        save=True,
        plots=True,
        val=True,
        optimizer='auto',
        lr0=LR,
        # 数据增强保持
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

    print(f"\n微调完成!")
    print(f"最佳模型: {results.save_dir}/weights/best.pt")


if __name__ == '__main__':
    main()
