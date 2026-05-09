"""
墙面裂缝分割模型训练脚本
使用YOLOv8进行裂缝检测和分割
"""
from ultralytics import YOLO
import torch

def main():
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载预训练的YOLOv8分割模型
    model = YOLO('yolov8n-seg.pt')  # nano版本，速度快

    # 显存优化设置
    if device == 'cuda':
        # 清理显存碎片
        torch.cuda.empty_cache()
        # 建议设置显存分配器 (可选，但在OOM时很有用)
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # 训练模型
    results = model.train(
        data='crack-seg/data.yaml',  # 数据集配置文件
        epochs=100,                   # 训练轮数
        imgsz=640,                    # 输入图像大小
        batch=4,                      # 显存溢出时调小 (原16 -> 现4)
        device=device,                # 使用的设备
        workers=2,                    # 降低线程数减少内存占用 (原4 -> 现2)
        project='runs/segment',       # 项目保存路径
        name='crack_seg',             # 实验名称
        patience=20,                  # 早停耐心值
        amp=True,                     # 使用混合精度训练 (显著减少显存占用)
        save=True,                    # 保存检查点
        plots=True,                   # 保存训练图表
        val=True,                     # 训练时验证

        # 数据增强参数
        hsv_h=0.015,                  # HSV色调增强
        hsv_s=0.7,                    # HSV饱和度增强
        hsv_v=0.4,                    # HSV明度增强
        degrees=0.0,                  # 旋转角度
        translate=0.1,                # 平移
        scale=0.5,                    # 缩放
        shear=0.0,                    # 剪切
        perspective=0.0,              # 透视变换
        flipud=0.0,                   # 上下翻转概率
        fliplr=0.5,                   # 左右翻转概率
        mosaic=1.0,                   # 马赛克增强概率
        mixup=0.0,                    # mixup增强概率
    )

    # 打印训练结果
    print("\n训练完成!")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    print(f"最后模型保存在: {results.save_dir}/weights/last.pt")

if __name__ == '__main__':
    main()
