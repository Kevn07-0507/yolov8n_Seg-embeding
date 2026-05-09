"""
模型优化和导出脚本
支持模型量化、导出为ONNX、TensorRT等格式
"""
from ultralytics import YOLO
import torch
import os

def export_onnx(model_path, img_size=640):
    """
    导出为ONNX格式

    Args:
        model_path: 模型路径
        img_size: 输入图像大小
    """
    print(f"\n导出ONNX模型...")
    model = YOLO(model_path)

    success = model.export(
        format='onnx',
        imgsz=img_size,
        dynamic=True,  # 动态输入尺寸
        simplify=True,  # 简化模型
    )

    if success:
        onnx_path = model_path.replace('.pt', '.onnx')
        print(f"✓ ONNX模型已保存: {onnx_path}")
        return onnx_path
    else:
        print("✗ ONNX导出失败")
        return None

def export_tensorrt(model_path, img_size=640):
    """
    导出为TensorRT格式（需要CUDA）

    Args:
        model_path: 模型路径
        img_size: 输入图像大小
    """
    if not torch.cuda.is_available():
        print("✗ TensorRT导出需要CUDA支持")
        return None

    print(f"\n导出TensorRT模型...")
    model = YOLO(model_path)

    try:
        success = model.export(
            format='engine',
            imgsz=img_size,
            half=True,  # FP16精度
        )

        if success:
            engine_path = model_path.replace('.pt', '.engine')
            print(f"✓ TensorRT模型已保存: {engine_path}")
            return engine_path
        else:
            print("✗ TensorRT导出失败")
            return None
    except Exception as e:
        print(f"✗ TensorRT导出失败: {e}")
        return None

def export_openvino(model_path, img_size=640):
    """
    导出为OpenVINO格式

    Args:
        model_path: 模型路径
        img_size: 输入图像大小
    """
    print(f"\n导出OpenVINO模型...")
    model = YOLO(model_path)

    try:
        success = model.export(
            format='openvino',
            imgsz=img_size,
        )

        if success:
            print(f"✓ OpenVINO模型已导出")
            return True
        else:
            print("✗ OpenVINO导出失败")
            return False
    except Exception as e:
        print(f"✗ OpenVINO导出失败: {e}")
        return False

def export_coreml(model_path, img_size=640):
    """
    导出为CoreML格式（用于iOS/macOS）

    Args:
        model_path: 模型路径
        img_size: 输入图像大小
    """
    print(f"\n导出CoreML模型...")
    model = YOLO(model_path)

    try:
        success = model.export(
            format='coreml',
            imgsz=img_size,
        )

        if success:
            print(f"✓ CoreML模型已导出")
            return True
        else:
            print("✗ CoreML导出失败")
            return False
    except Exception as e:
        print(f"✗ CoreML导出失败: {e}")
        return False

def export_tflite(model_path, img_size=640):
    """
    导出为TFLite格式（用于移动端）

    Args:
        model_path: 模型路径
        img_size: 输入图像大小
    """
    print(f"\n导出TFLite模型...")
    model = YOLO(model_path)

    try:
        success = model.export(
            format='tflite',
            imgsz=img_size,
        )

        if success:
            print(f"✓ TFLite模型已导出")
            return True
        else:
            print("✗ TFLite导出失败")
            return False
    except Exception as e:
        print(f"✗ TFLite导出失败: {e}")
        return False

def quantize_model(model_path, data_yaml='crack-seg/data.yaml'):
    """
    量化模型（INT8）

    Args:
        model_path: 模型路径
        data_yaml: 数据集配置文件
    """
    print(f"\n量化模型...")
    model = YOLO(model_path)

    try:
        # 导出为INT8量化的ONNX模型
        success = model.export(
            format='onnx',
            int8=True,
            data=data_yaml,
        )

        if success:
            print(f"✓ 量化模型已导出")
            return True
        else:
            print("✗ 模型量化失败")
            return False
    except Exception as e:
        print(f"✗ 模型量化失败: {e}")
        return False

def benchmark_model(model_path, img_size=640):
    """
    测试模型性能

    Args:
        model_path: 模型路径
        img_size: 输入图像大小
    """
    print(f"\n性能测试...")
    model = YOLO(model_path)

    try:
        metrics = model.benchmark(
            imgsz=img_size,
            half=torch.cuda.is_available(),  # 如果有GPU则使用FP16
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        print("✓ 性能测试完成")
        return metrics
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        return None

def compare_model_sizes(model_dir='runs/segment/crack_seg/weights'):
    """
    比较不同格式模型的大小

    Args:
        model_dir: 模型目录
    """
    print(f"\n模型大小比较:")
    print("-" * 60)

    if not os.path.exists(model_dir):
        print(f"目录不存在: {model_dir}")
        return

    formats = {
        '.pt': 'PyTorch',
        '.onnx': 'ONNX',
        '.engine': 'TensorRT',
        '.mlmodel': 'CoreML',
        '.tflite': 'TFLite',
    }

    for ext, name in formats.items():
        files = [f for f in os.listdir(model_dir) if f.endswith(ext)]
        for file in files:
            file_path = os.path.join(model_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"{name:15s} ({file}): {size_mb:.2f} MB")

def main():
    print("模型优化和导出工具")
    print("="*60)

    # 检查模型
    model_path = 'runs/segment/crack_seg/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"未找到模型: {model_path}")
        model_path = input("请输入模型路径: ").strip()
        if not os.path.exists(model_path):
            print("模型不存在!")
            return

    print(f"使用模型: {model_path}")

    print("\n请选择操作:")
    print("1. 导出ONNX格式")
    print("2. 导出TensorRT格式 (需要CUDA)")
    print("3. 导出OpenVINO格式")
    print("4. 导出CoreML格式 (用于iOS/macOS)")
    print("5. 导出TFLite格式 (用于移动端)")
    print("6. 量化模型 (INT8)")
    print("7. 性能测试")
    print("8. 导出所有格式")
    print("9. 比较模型大小")

    choice = input("\n请输入选项 (1-9): ").strip()

    img_size = input("输入图像大小 (默认: 640): ").strip()
    img_size = int(img_size) if img_size else 640

    if choice == '1':
        export_onnx(model_path, img_size)
    elif choice == '2':
        export_tensorrt(model_path, img_size)
    elif choice == '3':
        export_openvino(model_path, img_size)
    elif choice == '4':
        export_coreml(model_path, img_size)
    elif choice == '5':
        export_tflite(model_path, img_size)
    elif choice == '6':
        quantize_model(model_path)
    elif choice == '7':
        benchmark_model(model_path, img_size)
    elif choice == '8':
        print("\n导出所有格式...")
        export_onnx(model_path, img_size)
        export_tensorrt(model_path, img_size)
        export_openvino(model_path, img_size)
        export_coreml(model_path, img_size)
        export_tflite(model_path, img_size)
        print("\n所有格式导出完成!")
    elif choice == '9':
        compare_model_sizes()
    else:
        print("无效的选项!")

    print("\n完成!")

if __name__ == '__main__':
    main()
