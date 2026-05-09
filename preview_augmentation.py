"""
数据增强预览脚本
可视化不同数据增强方法的效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
import os
import font_config  # 解决中文显示异常问题

class AugmentationPreview:
    def __init__(self):
        """初始化数据增强预览器"""
        self.augmentations = self.get_augmentations()

    def get_augmentations(self):
        """定义各种数据增强方法"""
        return {
            '原图': None,
            '水平翻转': A.HorizontalFlip(p=1.0),
            '垂直翻转': A.VerticalFlip(p=1.0),
            '旋转': A.Rotate(limit=30, p=1.0),
            '亮度对比度': A.RandomBrightnessContrast(p=1.0),
            '模糊': A.Blur(blur_limit=7, p=1.0),
            '高斯噪声': A.GaussNoise(p=1.0),
            '色调饱和度': A.HueSaturationValue(p=1.0),
            '随机裁剪': A.RandomCrop(height=400, width=400, p=1.0),
            '弹性变换': A.ElasticTransform(p=1.0),
            '网格扭曲': A.GridDistortion(p=1.0),
            '光学扭曲': A.OpticalDistortion(p=1.0),
        }

    def preview_single_image(self, image_path, save_path=None):
        """
        预览单张图片的所有增强效果

        Args:
            image_path: 图片路径
            save_path: 保存路径
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 创建子图
        num_augs = len(self.augmentations)
        cols = 4
        rows = (num_augs + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        fig.suptitle(f'数据增强预览 - {os.path.basename(image_path)}', fontsize=16)

        axes = axes.flatten() if num_augs > 1 else [axes]

        # 应用每种增强
        for idx, (name, aug) in enumerate(self.augmentations.items()):
            if aug is None:
                aug_image = image
            else:
                try:
                    augmented = aug(image=image)
                    aug_image = augmented['image']
                except Exception as e:
                    print(f"增强 {name} 失败: {e}")
                    aug_image = image

            axes[idx].imshow(aug_image)
            axes[idx].set_title(name, fontsize=12)
            axes[idx].axis('off')

        # 隐藏多余的子图
        for idx in range(num_augs, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        # 保存或显示
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"预览已保存: {save_path}")
        else:
            plt.show()

    def preview_batch(self, image_dir, num_samples=3, save_dir='augmentation_preview'):
        """
        批量预览多张图片的增强效果

        Args:
            image_dir: 图片目录
            num_samples: 预览图片数量
            save_dir: 保存目录
        """
        # 获取图片文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))

        if not image_files:
            print(f"在 {image_dir} 中未找到图片")
            return

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 预览前N张图片
        for idx, image_path in enumerate(image_files[:num_samples]):
            save_path = os.path.join(save_dir, f'preview_{idx+1}.png')
            print(f"\n预览图片 {idx+1}/{num_samples}: {image_path.name}")
            self.preview_single_image(str(image_path), save_path)

        print(f"\n所有预览已保存到: {save_dir}")

    def compare_augmentations(self, image_path, aug_names, save_path=None):
        """
        对比特定的几种增强方法

        Args:
            image_path: 图片路径
            aug_names: 增强方法名称列表
            save_path: 保存路径
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 创建子图
        num_augs = len(aug_names)
        fig, axes = plt.subplots(1, num_augs, figsize=(5*num_augs, 5))
        fig.suptitle('数据增强对比', fontsize=16)

        if num_augs == 1:
            axes = [axes]

        # 应用选定的增强
        for idx, name in enumerate(aug_names):
            if name not in self.augmentations:
                print(f"未知的增强方法: {name}")
                continue

            aug = self.augmentations[name]
            if aug is None:
                aug_image = image
            else:
                augmented = aug(image=image)
                aug_image = augmented['image']

            axes[idx].imshow(aug_image)
            axes[idx].set_title(name, fontsize=14)
            axes[idx].axis('off')

        plt.tight_layout()

        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"对比图已保存: {save_path}")
        else:
            plt.show()

def main():
    print("数据增强预览工具")
    print("="*60)

    previewer = AugmentationPreview()

    print("\n可用的增强方法:")
    for idx, name in enumerate(previewer.augmentations.keys(), 1):
        print(f"  {idx}. {name}")

    print("\n请选择操作:")
    print("1. 预览单张图片的所有增强")
    print("2. 批量预览多张图片")
    print("3. 对比特定增强方法")

    choice = input("\n请输入选项 (1-3): ").strip()

    if choice == '1':
        image_path = input("请输入图片路径: ").strip()
        if os.path.exists(image_path):
            save = input("是否保存预览? (y/n): ").strip().lower()
            save_path = 'augmentation_preview/single_preview.png' if save == 'y' else None
            previewer.preview_single_image(image_path, save_path)
        else:
            print(f"图片不存在: {image_path}")

    elif choice == '2':
        image_dir = input("请输入图片目录: ").strip()
        if os.path.exists(image_dir):
            num_samples = input("预览图片数量 (默认: 3): ").strip()
            num_samples = int(num_samples) if num_samples else 3
            previewer.preview_batch(image_dir, num_samples)
        else:
            print(f"目录不存在: {image_dir}")

    elif choice == '3':
        image_path = input("请输入图片路径: ").strip()
        if os.path.exists(image_path):
            print("\n输入要对比的增强方法（用逗号分隔）:")
            aug_names = input().strip().split(',')
            aug_names = [name.strip() for name in aug_names]
            save = input("是否保存对比图? (y/n): ").strip().lower()
            save_path = 'augmentation_preview/comparison.png' if save == 'y' else None
            previewer.compare_augmentations(image_path, aug_names, save_path)
        else:
            print(f"图片不存在: {image_path}")

    else:
        print("无效的选项!")

if __name__ == '__main__':
    # 检查albumentations是否安装
    try:
        import albumentations
    except ImportError:
        print("错误: 需要安装albumentations库")
        print("运行: pip install albumentations")
        exit(1)

    main()
