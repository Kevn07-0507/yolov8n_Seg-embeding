"""
墙面裂缝分割模型推理脚本
使用训练好的YOLOv8模型进行裂缝检测和分割
"""
from ultralytics import YOLO
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def predict_single_image(model, image_path, save_dir='runs/predict'):
    """
    对单张图片进行预测

    Args:
        model: YOLO模型
        image_path: 图片路径
        save_dir: 结果保存目录
    """
    results = model.predict(
        source=image_path,
        save=True,
        save_txt=True,
        save_conf=True,
        project=save_dir,
        name='crack_detection',
        conf=0.25,  # 置信度阈值
        iou=0.7,    # NMS IOU阈值
    )

    return results

def predict_batch(model, source_dir, save_dir='runs/predict'):
    """
    批量预测图片

    Args:
        model: YOLO模型
        source_dir: 图片目录
        save_dir: 结果保存目录
    """
    results = model.predict(
        source=source_dir,
        save=True,
        save_txt=True,
        save_conf=True,
        project=save_dir,
        name='crack_detection_batch',
        conf=0.25,
        iou=0.7,
    )

    return results

def predict_video(model, video_path, save_dir='runs/predict'):
    """
    对视频进行预测

    Args:
        model: YOLO模型
        video_path: 视频路径
        save_dir: 结果保存目录
    """
    results = model.predict(
        source=video_path,
        save=True,
        project=save_dir,
        name='crack_detection_video',
        conf=0.25,
        iou=0.7,
        stream=True,  # 流式处理视频
    )

    return results

def evaluate_model(model, data_yaml='crack-seg/data.yaml'):
    """
    在验证集上评估模型性能

    Args:
        model: YOLO模型
        data_yaml: 数据集配置文件
    """
    metrics = model.val(
        data=data_yaml,
        split='val',
        save_json=True,
        save_hybrid=True,
    )

    print("\n=== 模型评估结果 ===")
    print(f"mAP50: {metrics.seg.map50:.4f}")
    print(f"mAP50-95: {metrics.seg.map:.4f}")
    print(f"Precision: {metrics.seg.mp:.4f}")
    print(f"Recall: {metrics.seg.mr:.4f}")

    return metrics

def main():
    model_path = 'E:/University_files/embed/runs/segment/crack_seg4/weights/best.pt'

    if not os.path.exists(model_path):
        print(f"未找到训练好的模型: {model_path}")
        model_path = input("请输入模型路径: ").strip().strip('"').strip("'")
        if not os.path.exists(model_path):
            print("模型不存在，退出。")
            return

    model = YOLO(model_path)
    print(f"加载模型: {model_path}")

    # 选择操作模式
    print("\n请选择操作模式:")
    print("1. 单张图片预测")
    print("2. 批量图片预测")
    print("3. 视频预测")
    print("4. 模型评估")
    print("5. 测试集预测")

    choice = input("\n请输入选项 (1-5): ").strip()

    if choice == '1':
        # 单张图片预测
        image_path = input("请输入图片路径: ").strip().strip('"').strip("'")
        image_path = os.path.normpath(image_path)
        print(f"  实际路径: {image_path}")
        if os.path.exists(image_path):
            # Windows路径转正斜杠，避免YOLO内部读取失败
            image_path_yolo = image_path.replace('\\', '/')
            results = predict_single_image(model, image_path_yolo)
            # 获取实际保存路径（YOLO会自动递增目录名）
            actual_save_dir = model.predictor.save_dir
            print(f"\n预测完成! 结果保存在: {actual_save_dir}")
            # 打印检测摘要
            r = results[0]
            print(f"  图像尺寸: {r.orig_shape}")
            n = len(r.boxes) if r.boxes is not None else 0
            print(f"  检测到裂缝数量: {n}")
            if n > 0 and r.boxes.conf is not None:
                confs = r.boxes.conf.tolist()
                print(f"  置信度: {[f'{c:.2f}' for c in confs]}")
            else:
                print("  提示: 未检出，可尝试降低置信度阈值 (当前 conf=0.25)")
            # 显示原图和检测结果
            orig = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            annotated = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(orig)
            axes[0].set_title('原始图像')
            axes[0].axis('off')
            axes[1].imshow(annotated)
            axes[1].set_title(f'检测结果 (裂缝数: {n})')
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()
        else:
            print(f"图片不存在: {image_path}")

    elif choice == '2':
        # 批量预测
        dir_path = input("请输入图片目录路径: ").strip()
        if os.path.exists(dir_path):
            results = predict_batch(model, dir_path)
            actual_save_dir = model.predictor.save_dir
            print(f"\n预测完成! 结果保存在: {actual_save_dir}")
            print(f"  共处理图片: {len(results)} 张")
        else:
            print(f"目录不存在: {dir_path}")

    elif choice == '3':
        # 视频预测
        video_path = input("请输入视频路径: ").strip()
        if os.path.exists(video_path):
            results = predict_video(model, video_path)
            actual_save_dir = model.predictor.save_dir
            print(f"\n预测完成! 结果保存在: {actual_save_dir}")
        else:
            print(f"视频不存在: {video_path}")

    elif choice == '4':
        # 模型评估
        if os.path.exists('crack-seg/data.yaml'):
            metrics = evaluate_model(model)
        else:
            print("未找到数据集配置文件: crack-seg/data.yaml")

    elif choice == '5':
        # 测试集预测
        test_dir = 'crack-seg/test/images'
        if os.path.exists(test_dir):
            print(f"\n正在对测试集进行预测...")
            results = predict_batch(model, test_dir, save_dir='runs/predict')
            actual_save_dir = model.predictor.save_dir
            print(f"\n预测完成! 结果保存在: {actual_save_dir}")
            print(f"  共处理图片: {len(results)} 张")
        else:
            print(f"测试集目录不存在: {test_dir}")

    else:
        print("无效的选项!")

if __name__ == '__main__':
    main()
