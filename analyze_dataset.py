"""
数据集分析脚本
分析墙面裂缝数据集的统计信息
"""
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import font_config  # 解决中文显示异常问题


def analyze_images(image_dir):
    """分析图片尺寸和格式"""
    print(f"\n分析目录: {image_dir}")

    if not os.path.exists(image_dir):
        print(f"目录不存在: {image_dir}")
        return None

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        print("没有找到图片文件")
        return None

    widths = []
    heights = []
    formats = defaultdict(int)

    print(f"正在分析 {len(image_files)} 张图片...")

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)

        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)

            ext = os.path.splitext(img_file)[1].lower()
            formats[ext] += 1

    stats = {
        'count': len(image_files),
        'width_mean': np.mean(widths),
        'width_std': np.std(widths),
        'width_min': np.min(widths),
        'width_max': np.max(widths),
        'height_mean': np.mean(heights),
        'height_std': np.std(heights),
        'height_min': np.min(heights),
        'height_max': np.max(heights),
        'formats': dict(formats),
        'widths': widths,
        'heights': heights,
    }

    return stats

def analyze_labels(label_dir):
    """分析标签文件"""
    print(f"\n分析标签目录: {label_dir}")

    if not os.path.exists(label_dir):
        print(f"目录不存在: {label_dir}")
        return None

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    if not label_files:
        print("没有找到标签文件")
        return None

    total_objects = 0
    objects_per_image = []
    polygon_points = []

    print(f"正在分析 {len(label_files)} 个标签文件...")

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        num_objects = len(lines)
        total_objects += num_objects
        objects_per_image.append(num_objects)

        for line in lines:
            parts = line.strip().split()
            if len(parts) > 1:
                # 第一个是类别ID，后面是坐标点
                num_points = (len(parts) - 1) // 2
                polygon_points.append(num_points)

    stats = {
        'label_count': len(label_files),
        'total_objects': total_objects,
        'avg_objects_per_image': np.mean(objects_per_image),
        'max_objects_per_image': np.max(objects_per_image),
        'min_objects_per_image': np.min(objects_per_image),
        'avg_polygon_points': np.mean(polygon_points) if polygon_points else 0,
        'objects_per_image': objects_per_image,
        'polygon_points': polygon_points,
    }

    return stats

def plot_statistics(train_stats, val_stats, test_stats):
    """绘制统计图表"""
    print("\n生成统计图表...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('墙面裂缝数据集统计分析', fontsize=16)

    # 1. 数据集大小分布
    ax = axes[0, 0]
    datasets = ['训练集', '验证集', '测试集']
    counts = [
        train_stats['count'] if train_stats else 0,
        val_stats['count'] if val_stats else 0,
        test_stats['count'] if test_stats else 0
    ]
    ax.bar(datasets, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('图片数量')
    ax.set_title('数据集大小分布')
    for i, v in enumerate(counts):
        ax.text(i, v + 50, str(v), ha='center', va='bottom')

    # 2. 图片尺寸分布（宽度）
    ax = axes[0, 1]
    if train_stats:
        ax.hist(train_stats['widths'], bins=30, alpha=0.7, label='训练集', color='#1f77b4')
    if val_stats:
        ax.hist(val_stats['widths'], bins=30, alpha=0.7, label='验证集', color='#ff7f0e')
    if test_stats:
        ax.hist(test_stats['widths'], bins=30, alpha=0.7, label='测试集', color='#2ca02c')
    ax.set_xlabel('宽度 (像素)')
    ax.set_ylabel('数量')
    ax.set_title('图片宽度分布')
    ax.legend()

    # 3. 图片尺寸分布（高度）
    ax = axes[0, 2]
    if train_stats:
        ax.hist(train_stats['heights'], bins=30, alpha=0.7, label='训练集', color='#1f77b4')
    if val_stats:
        ax.hist(val_stats['heights'], bins=30, alpha=0.7, label='验证集', color='#ff7f0e')
    if test_stats:
        ax.hist(test_stats['heights'], bins=30, alpha=0.7, label='测试集', color='#2ca02c')
    ax.set_xlabel('高度 (像素)')
    ax.set_ylabel('数量')
    ax.set_title('图片高度分布')
    ax.legend()

    # 4. 标签统计
    ax = axes[1, 0]
    if train_stats and 'total_objects' in train_stats:
        datasets = ['训练集', '验证集', '测试集']
        total_objs = [
            train_stats.get('total_objects', 0),
            val_stats.get('total_objects', 0) if val_stats else 0,
            test_stats.get('total_objects', 0) if test_stats else 0
        ]
        ax.bar(datasets, total_objs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('裂缝对象数量')
        ax.set_title('裂缝对象总数')
        for i, v in enumerate(total_objs):
            ax.text(i, v + 10, str(v), ha='center', va='bottom')

    # 5. 每张图片的对象数量分布
    ax = axes[1, 1]
    if train_stats and 'objects_per_image' in train_stats:
        ax.hist(train_stats['objects_per_image'], bins=20, alpha=0.7, label='训练集', color='#1f77b4')
    if val_stats and 'objects_per_image' in val_stats:
        ax.hist(val_stats['objects_per_image'], bins=20, alpha=0.7, label='验证集', color='#ff7f0e')
    if test_stats and 'objects_per_image' in test_stats:
        ax.hist(test_stats['objects_per_image'], bins=20, alpha=0.7, label='测试集', color='#2ca02c')
    ax.set_xlabel('每张图片的裂缝数量')
    ax.set_ylabel('图片数量')
    ax.set_title('每张图片的裂缝数量分布')
    ax.legend()

    # 6. 多边形点数分布
    ax = axes[1, 2]
    if train_stats and 'polygon_points' in train_stats:
        ax.hist(train_stats['polygon_points'], bins=30, alpha=0.7, label='训练集', color='#1f77b4')
    if val_stats and 'polygon_points' in val_stats:
        ax.hist(val_stats['polygon_points'], bins=30, alpha=0.7, label='验证集', color='#ff7f0e')
    if test_stats and 'polygon_points' in test_stats:
        ax.hist(test_stats['polygon_points'], bins=30, alpha=0.7, label='测试集', color='#2ca02c')
    ax.set_xlabel('多边形点数')
    ax.set_ylabel('数量')
    ax.set_title('分割多边形点数分布')
    ax.legend()

    plt.tight_layout()

    # 保存图表
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dataset_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"统计图表已保存: {output_path}")

    plt.show()

def print_summary(train_stats, val_stats, test_stats, train_label_stats, val_label_stats, test_label_stats):
    """打印统计摘要"""
    print("\n" + "="*60)
    print("数据集统计摘要")
    print("="*60)

    # 训练集
    if train_stats:
        print(f"\n【训练集】")
        print(f"  图片数量: {train_stats['count']}")
        print(f"  图片尺寸: {train_stats['width_mean']:.0f}x{train_stats['height_mean']:.0f} (平均)")
        print(f"  尺寸范围: {train_stats['width_min']}x{train_stats['height_min']} ~ {train_stats['width_max']}x{train_stats['height_max']}")
        if train_label_stats:
            print(f"  裂缝对象总数: {train_label_stats['total_objects']}")
            print(f"  平均每张图片裂缝数: {train_label_stats['avg_objects_per_image']:.2f}")
            print(f"  平均多边形点数: {train_label_stats['avg_polygon_points']:.1f}")

    # 验证集
    if val_stats:
        print(f"\n【验证集】")
        print(f"  图片数量: {val_stats['count']}")
        print(f"  图片尺寸: {val_stats['width_mean']:.0f}x{val_stats['height_mean']:.0f} (平均)")
        if val_label_stats:
            print(f"  裂缝对象总数: {val_label_stats['total_objects']}")
            print(f"  平均每张图片裂缝数: {val_label_stats['avg_objects_per_image']:.2f}")

    # 测试集
    if test_stats:
        print(f"\n【测试集】")
        print(f"  图片数量: {test_stats['count']}")
        print(f"  图片尺寸: {test_stats['width_mean']:.0f}x{test_stats['height_mean']:.0f} (平均)")
        if test_label_stats:
            print(f"  裂缝对象总数: {test_label_stats['total_objects']}")
            print(f"  平均每张图片裂缝数: {test_label_stats['avg_objects_per_image']:.2f}")

    print("\n" + "="*60)

def main():
    print("墙面裂缝数据集分析工具")
    print("="*60)

    # 分析训练集
    train_img_stats = analyze_images('crack-seg/train/images')
    train_label_stats = analyze_labels('crack-seg/train/labels')

    # 分析验证集
    val_img_stats = analyze_images('crack-seg/valid/images')
    val_label_stats = analyze_labels('crack-seg/valid/labels')

    # 分析测试集
    test_img_stats = analyze_images('crack-seg/test/images')
    test_label_stats = analyze_labels('crack-seg/test/labels')

    # 打印摘要
    print_summary(train_img_stats, val_img_stats, test_img_stats,
                  train_label_stats, val_label_stats, test_label_stats)

    # 绘制统计图表
    # 合并图片和标签统计
    if train_img_stats and train_label_stats:
        train_img_stats.update(train_label_stats)
    if val_img_stats and val_label_stats:
        val_img_stats.update(val_label_stats)
    if test_img_stats and test_label_stats:
        test_img_stats.update(test_label_stats)

    plot_statistics(train_img_stats, val_img_stats, test_img_stats)

if __name__ == '__main__':
    main()
