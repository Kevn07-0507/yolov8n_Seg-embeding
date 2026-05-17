"""
工具函数模块
提供常用的辅助函数
"""
import os
import cv2
import numpy as np
from pathlib import Path
import json
import yaml

def load_yaml(file_path):
    """
    加载YAML文件

    Args:
        file_path: YAML文件路径

    Returns:
        字典对象
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"加载YAML文件失败: {e}")
        return None

def save_yaml(data, file_path):
    """
    保存为YAML文件

    Args:
        data: 数据字典
        file_path: 保存路径

    Returns:
        是否成功
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        print(f"保存YAML文件失败: {e}")
        return False

def load_json(file_path):
    """
    加载JSON文件

    Args:
        file_path: JSON文件路径

    Returns:
        字典对象
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        return None

def save_json(data, file_path, indent=2):
    """
    保存为JSON文件

    Args:
        data: 数据字典
        file_path: 保存路径
        indent: 缩进空格数

    Returns:
        是否成功
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"保存JSON文件失败: {e}")
        return False

def get_image_files(directory, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
    """
    获取目录中的所有图片文件

    Args:
        directory: 目录路径
        extensions: 图片扩展名列表

    Returns:
        图片文件路径列表
    """
    image_files = set()
    for ext in extensions:
        image_files.update(Path(directory).rglob(f'*{ext}'))
        image_files.update(Path(directory).rglob(f'*{ext.upper()}'))
    return [str(f) for f in image_files]

def resize_image(image, target_size, keep_ratio=True):
    """
    调整图片大小

    Args:
        image: 输入图片
        target_size: 目标大小 (width, height)
        keep_ratio: 是否保持宽高比

    Returns:
        调整后的图片
    """
    if keep_ratio:
        h, w = image.shape[:2]
        target_w, target_h = target_size

        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 调整大小
        resized = cv2.resize(image, (new_w, new_h))

        # 创建目标大小的画布
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # 计算居中位置
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        # 将调整后的图片放到画布上
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas
    else:
        return cv2.resize(image, target_size)

def draw_bbox(image, bbox, label=None, color=(0, 255, 0), thickness=2):
    """
    在图片上绘制边界框

    Args:
        image: 输入图片
        bbox: 边界框 (x1, y1, x2, y2)
        label: 标签文本
        color: 颜色 (B, G, R)
        thickness: 线条粗细

    Returns:
        绘制后的图片
    """
    x1, y1, x2, y2 = map(int, bbox)

    # 绘制矩形
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # 绘制标签
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        # 获取文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # 绘制背景矩形
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )

        # 绘制文本
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )

    return image

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU

    Args:
        box1: 边界框1 (x1, y1, x2, y2)
        box2: 边界框2 (x1, y1, x2, y2)

    Returns:
        IoU值
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def create_directory(directory):
    """
    创建目录（如果不存在）

    Args:
        directory: 目录路径

    Returns:
        是否成功
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"创建目录失败: {e}")
        return False

def get_file_size(file_path, unit='MB'):
    """
    获取文件大小

    Args:
        file_path: 文件路径
        unit: 单位 ('B', 'KB', 'MB', 'GB')

    Returns:
        文件大小
    """
    if not os.path.exists(file_path):
        return 0

    size_bytes = os.path.getsize(file_path)

    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3
    }

    return size_bytes / units.get(unit, 1)

def format_time(seconds):
    """
    格式化时间

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def calculate_metrics(tp, fp, fn):
    """
    计算精确率、召回率和F1分数

    Args:
        tp: True Positives
        fp: False Positives
        fn: False Negatives

    Returns:
        (precision, recall, f1)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def merge_images(images, grid_size=None, padding=10):
    """
    将多张图片合并为网格

    Args:
        images: 图片列表
        grid_size: 网格大小 (rows, cols)，None则自动计算
        padding: 图片间距

    Returns:
        合并后的图片
    """
    if not images:
        return None

    num_images = len(images)

    # 自动计算网格大小
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    else:
        rows, cols = grid_size

    # 获取单张图片大小
    h, w = images[0].shape[:2]

    # 创建画布
    canvas_h = rows * h + (rows + 1) * padding
    canvas_w = cols * w + (cols + 1) * padding
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # 放置图片
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        y = row * h + (row + 1) * padding
        x = col * w + (col + 1) * padding

        canvas[y:y+h, x:x+w] = img

    return canvas

if __name__ == '__main__':
    # 测试工具函数
    print("工具函数模块测试")

    # 测试时间格式化
    print(f"3661秒 = {format_time(3661)}")

    # 测试指标计算
    precision, recall, f1 = calculate_metrics(80, 10, 20)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
