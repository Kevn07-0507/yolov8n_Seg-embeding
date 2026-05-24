"""
可视化脚本
可视化训练结果、预测结果和模型性能
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import font_config  # 解决中文显示异常问题


def _export_data_table(df, results_dir):
    """导出训练全量数据表格（所有 available 列）"""
    output_path = os.path.join(results_dir, 'training_data_table.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"数据表格已导出: {output_path}")
    print(f"  共 {len(df)} 行 × {len(df.columns)} 列")


def visualize_training_results(results_dir='runs/segment/crack_seg'):
    """可视化训练结果"""
    print(f"可视化训练结果: {results_dir}")

    if not os.path.exists(results_dir):
        print(f"训练结果目录不存在: {results_dir}")
        return

    # 读取results.csv
    csv_path = os.path.join(results_dir, 'results.csv')
    if not os.path.exists(csv_path):
        print(f"未找到训练结果文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # 去除列名空格

    # ═══ 导出数据表格 ═══
    _export_data_table(df, results_dir)

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('训练过程可视化', fontsize=16)

    epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))

    # 1. 训练和验证损失
    ax = axes[0, 0]
    if 'train/box_loss' in df.columns:
        ax.plot(epochs, df['train/box_loss'], label='Box Loss', linewidth=2)
    if 'train/seg_loss' in df.columns:
        ax.plot(epochs, df['train/seg_loss'], label='Seg Loss', linewidth=2)
    if 'train/cls_loss' in df.columns:
        ax.plot(epochs, df['train/cls_loss'], label='Cls Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('训练损失')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 验证损失
    ax = axes[0, 1]
    if 'val/box_loss' in df.columns:
        ax.plot(epochs, df['val/box_loss'], label='Box Loss', linewidth=2)
    if 'val/seg_loss' in df.columns:
        ax.plot(epochs, df['val/seg_loss'], label='Seg Loss', linewidth=2)
    if 'val/cls_loss' in df.columns:
        ax.plot(epochs, df['val/cls_loss'], label='Cls Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('验证损失')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Precision和Recall
    ax = axes[0, 2]
    if 'metrics/precision(B)' in df.columns:
        ax.plot(epochs, df['metrics/precision(B)'], label='Precision', linewidth=2)
    if 'metrics/recall(B)' in df.columns:
        ax.plot(epochs, df['metrics/recall(B)'], label='Recall', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # 4. mAP50
    ax = axes[1, 0]
    if 'metrics/mAP50(B)' in df.columns:
        ax.plot(epochs, df['metrics/mAP50(B)'], label='Box mAP50', linewidth=2, color='#1f77b4')
    if 'metrics/mAP50(M)' in df.columns:
        ax.plot(epochs, df['metrics/mAP50(M)'], label='Mask mAP50', linewidth=2, color='#ff7f0e')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP50')
    ax.set_title('mAP@0.5')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # 5. mAP50-95
    ax = axes[1, 1]
    if 'metrics/mAP50-95(B)' in df.columns:
        ax.plot(epochs, df['metrics/mAP50-95(B)'], label='Box mAP50-95', linewidth=2, color='#1f77b4')
    if 'metrics/mAP50-95(M)' in df.columns:
        ax.plot(epochs, df['metrics/mAP50-95(M)'], label='Mask mAP50-95', linewidth=2, color='#ff7f0e')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP50-95')
    ax.set_title('mAP@0.5:0.95')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # 6. 学习率
    ax = axes[1, 2]
    if 'lr/pg0' in df.columns:
        ax.plot(epochs, df['lr/pg0'], label='LR pg0', linewidth=2)
    if 'lr/pg1' in df.columns:
        ax.plot(epochs, df['lr/pg1'], label='LR pg1', linewidth=2)
    if 'lr/pg2' in df.columns:
        ax.plot(epochs, df['lr/pg2'], label='LR pg2', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('学习率变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(results_dir, 'training_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"训练可视化已保存: {output_path}")

    plt.show()

def visualize_predictions(image_dir, save_dir='visualization_results'):
    """可视化预测结果"""
    print(f"\n可视化预测结果: {image_dir}")

    if not os.path.exists(image_dir):
        print(f"预测结果目录不存在: {image_dir}")
        return

    # 查找预测结果图片
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    if not image_files:
        print("未找到预测结果图片")
        return

    # 创建输出目录
    os.makedirs(save_dir, exist_ok=True)

    # 显示前9张预测结果
    num_images = min(9, len(image_files))
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('裂缝检测预测结果', fontsize=16)

    for idx in range(num_images):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        img_path = os.path.join(image_dir, image_files[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.imshow(img)
        ax.axis('off')
        ax.set_title(image_files[idx], fontsize=8)

    # 隐藏多余的子图
    for idx in range(num_images, 9):
        row = idx // 3
        col = idx % 3
        axes[row, col].axis('off')

    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(save_dir, 'predictions_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"预测结果可视化已保存: {output_path}")

    plt.show()

def compare_predictions(original_dir, prediction_dir, num_samples=5, save_dir='visualization_results'):
    """对比原图和预测结果"""
    print(f"\n对比原图和预测结果")

    if not os.path.exists(original_dir) or not os.path.exists(prediction_dir):
        print("目录不存在")
        return

    original_files = set([f for f in os.listdir(original_dir) if f.endswith(('.jpg', '.png'))])
    prediction_files = set([f for f in os.listdir(prediction_dir) if f.endswith(('.jpg', '.png'))])

    # 找到共同的文件
    common_files = list(original_files & prediction_files)

    if not common_files:
        print("未找到匹配的图片")
        return

    num_samples = min(num_samples, len(common_files))

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
    fig.suptitle('原图 vs 预测结果对比', fontsize=16)

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_samples):
        filename = common_files[idx]

        # 原图
        original_path = os.path.join(original_dir, filename)
        original_img = cv2.imread(original_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        axes[idx, 0].imshow(original_img)
        axes[idx, 0].axis('off')
        axes[idx, 0].set_title(f'原图: {filename}', fontsize=10)

        # 预测结果
        prediction_path = os.path.join(prediction_dir, filename)
        prediction_img = cv2.imread(prediction_path)
        prediction_img = cv2.cvtColor(prediction_img, cv2.COLOR_BGR2RGB)

        axes[idx, 1].imshow(prediction_img)
        axes[idx, 1].axis('off')
        axes[idx, 1].set_title(f'预测结果: {filename}', fontsize=10)

    plt.tight_layout()

    # 保存图表
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, 'comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"对比结果已保存: {output_path}")

    plt.show()

def main():
    print("墙面裂缝检测可视化工具")
    print("="*60)

    print("\n请选择可视化类型:")
    print("1. 训练过程可视化")
    print("2. 预测结果可视化")
    print("3. 原图与预测结果对比")

    choice = input("\n请输入选项 (1-3): ").strip()

    if choice == '1':
        # 训练过程可视化
        results_dir = input("请输入训练结果目录 (默认: runs/segment/crack_seg): ").strip()
        if not results_dir:
            results_dir = 'runs/segment/crack_seg'
        visualize_training_results(results_dir)

    elif choice == '2':
        # 预测结果可视化
        image_dir = input("请输入预测结果图片目录: ").strip()
        if os.path.exists(image_dir):
            visualize_predictions(image_dir)
        else:
            print(f"目录不存在: {image_dir}")

    elif choice == '3':
        # 对比可视化
        original_dir = input("请输入原图目录: ").strip()
        prediction_dir = input("请输入预测结果目录: ").strip()

        if os.path.exists(original_dir) and os.path.exists(prediction_dir):
            num_samples = input("对比图片数量 (默认: 5): ").strip()
            num_samples = int(num_samples) if num_samples else 5
            compare_predictions(original_dir, prediction_dir, num_samples)
        else:
            print("目录不存在")

    else:
        print("无效的选项!")

if __name__ == '__main__':
    main()
