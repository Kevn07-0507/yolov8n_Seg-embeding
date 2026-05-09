"""
快速开始脚本 - 墙面裂缝检测
自动完成环境检查、模型下载和简单测试
"""
import os
import sys

def check_environment():
    """检查环境和依赖"""
    print("=== 检查环境 ===")

    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch未安装")
        return False

    try:
        import ultralytics
        print(f"✓ Ultralytics版本: {ultralytics.__version__}")
    except ImportError:
        print("✗ Ultralytics未安装")
        return False

    try:
        import cv2
        print(f"✓ OpenCV版本: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV未安装")
        return False

    return True

def check_dataset():
    """检查数据集"""
    print("\n=== 检查数据集 ===")

    if not os.path.exists('crack-seg'):
        print("✗ 数据集目录不存在: crack-seg")
        return False

    if not os.path.exists('crack-seg/data.yaml'):
        print("✗ 数据集配置文件不存在: crack-seg/data.yaml")
        return False

    train_images = 'crack-seg/train/images'
    val_images = 'crack-seg/valid/images'
    test_images = 'crack-seg/test/images'

    if os.path.exists(train_images):
        train_count = len([f for f in os.listdir(train_images) if f.endswith(('.jpg', '.png'))])
        print(f"✓ 训练集: {train_count} 张图片")
    else:
        print("✗ 训练集目录不存在")
        return False

    if os.path.exists(val_images):
        val_count = len([f for f in os.listdir(val_images) if f.endswith(('.jpg', '.png'))])
        print(f"✓ 验证集: {val_count} 张图片")

    if os.path.exists(test_images):
        test_count = len([f for f in os.listdir(test_images) if f.endswith(('.jpg', '.png'))])
        print(f"✓ 测试集: {test_count} 张图片")

    return True

def download_pretrained_model():
    """下载预训练模型"""
    print("\n=== 下载预训练模型 ===")

    from ultralytics import YOLO

    try:
        print("正在下载 YOLOv8n-seg 预训练模型...")
        model = YOLO('yolov8n-seg.pt')
        print("✓ 模型下载完成")
        return True
    except Exception as e:
        print(f"✗ 模型下载失败: {e}")
        return False

def run_quick_test():
    """运行快速测试"""
    print("\n=== 运行快速测试 ===")

    from ultralytics import YOLO

    # 使用测试集的第一张图片进行测试
    test_images = 'crack-seg/test/images'
    if not os.path.exists(test_images):
        print("✗ 测试集不存在")
        return False

    test_files = [f for f in os.listdir(test_images) if f.endswith(('.jpg', '.png'))]
    if not test_files:
        print("✗ 测试集中没有图片")
        return False

    test_image = os.path.join(test_images, test_files[0])
    print(f"使用测试图片: {test_files[0]}")

    try:
        model = YOLO('yolov8n-seg.pt')
        results = model.predict(
            source=test_image,
            save=True,
            project='runs/quick_test',
            name='test',
            conf=0.25,
        )
        print("✓ 测试完成")
        print(f"  结果保存在: runs/quick_test/test")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def main():
    print("墙面裂缝检测 - 快速开始\n")

    # 1. 检查环境
    if not check_environment():
        print("\n请先安装依赖: pip install -r requirements.txt")
        return

    # 2. 检查数据集
    if not check_dataset():
        print("\n请确保数据集已正确解压到 crack-seg 目录")
        return

    # 3. 下载预训练模型
    if not download_pretrained_model():
        return

    # 4. 运行快速测试
    if not run_quick_test():
        return

    print("\n=== 环境准备完成 ===")
    print("\n接下来你可以:")
    print("1. 训练模型: python train_crack_seg.py")
    print("2. 进行预测: python predict_crack_seg.py")
    print("\n训练建议:")
    print("- 如果有GPU，训练会更快")
    print("- 默认训练100轮，大约需要1-2小时（取决于硬件）")
    print("- 可以在 train_crack_seg.py 中调整参数")

if __name__ == '__main__':
    main()
