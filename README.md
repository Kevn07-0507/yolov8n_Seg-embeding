# 🧱 墙面裂缝检测系统

基于 **YOLOv8 实例分割** 的墙面裂缝自动检测与分割系统，提供从数据准备、模型训练到推理部署的全流程解决方案。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/YOLOv8-Seg-00b894.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/License-AGPL--3.0-green.svg" alt="License">
</p>

---

## 📋 项目简介

本项目使用 YOLOv8 实例分割模型对建筑墙面裂缝进行自动检测和像素级分割，适用于建筑结构检测、基础设施巡检等场景。

| 数据集 | 训练集 | 验证集 | 测试集 |
|--------|--------|--------|--------|
| crack-seg | 3717 张 | 112 张 | 200 张 |

## ✨ 主要功能

- **模型训练** — 支持 YOLOv8n/s/m/l/x-seg 多规格模型，可自定义参数
- **图像预测** — 单张 / 批量图片裂缝检测与分割
- **实时检测** — 摄像头或视频文件实时推理，显示 FPS
- **结果可视化** — 训练曲线、预测掩码、原图对比
- **批量报告** — 自动生成 JSON / CSV / HTML 检测报告
- **模型优化** — 导出为 ONNX / TensorRT / OpenVINO / CoreML / TFLite 格式
- **Web 界面** — 基于 Gradio 的图形化操作界面
- **模型对比** — 多模型精度与速度对比评估

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（可选，用于 GPU 加速）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 解压数据集

```bash
# 将 crack-seg.rar 解压到项目根目录
unrar x crack-seg.rar
```

### 开始使用

```bash
# 方式一：交互式主菜单（推荐）
python main.py

# 方式二：直接运行各模块
python quick_start.py      # 环境检查与快速测试
python analyze_dataset.py  # 数据集统计分析
python train_crack_seg.py  # 训练模型
python predict_crack_seg.py # 模型预测
python web_app.py          # 启动 Web 界面
```

## 📁 项目结构

```
embed/
├── crack-seg/                  # 数据集
│   ├── data.yaml               #   数据集配置文件
│   ├── train/                  #   训练集 (images + labels)
│   ├── valid/                  #   验证集
│   └── test/                   #   测试集
├── crack-seg.rar               # 数据集压缩包
├── yolo11n.pt                  # YOLO11 预训练权重
├── yolov8n-seg.pt              # YOLOv8-seg 预训练权重
├── config.yaml                 # 全局配置文件
│
├── main.py                     # 主入口（交互式菜单）
├── train_crack_seg.py          # 模型训练
├── predict_crack_seg.py        # 模型预测
├── batch_process.py            # 批量处理 & 报告生成
├── realtime_detect.py          # 实时检测（摄像头/视频）
├── web_app.py                  # Gradio Web 界面
├── export_model.py             # 模型导出
├── compare_models.py           # 模型对比
├── benchmark.py                # 性能基准测试
├── analyze_dataset.py          # 数据集分析
├── visualize_results.py        # 结果可视化
├── preview_augmentation.py     # 数据增强预览
├── quick_start.py              # 快速启动检查
├── test_system.py              # 系统测试
│
├── config_manager.py           # 配置管理模块
├── utils.py                    # 工具函数
├── logger.py                   # 日志模块
├── font_config.py              # 字体配置
│
├── requirements.txt            # Python 依赖
├── Dockerfile                  # Docker 镜像
├── docker-compose.yml          # Docker Compose
├── start.bat / start.sh        # 一键启动脚本
└── 报告/                       # 项目文档与报告
```

## 📖 使用说明

### 1. 训练模型

```bash
python train_crack_seg.py
```

训练参数可在脚本中调整：

```python
model.train(
    data='crack-seg/data.yaml',
    epochs=100,        # 训练轮数
    imgsz=640,         # 输入尺寸
    batch=16,          # 批次大小（显存不足时可调小）
    device='cuda',     # 设备选择
    patience=20,       # 早停轮数
)
```

训练结果保存于 `runs/segment/crack_seg/`

### 2. 图像预测

```python
from ultralytics import YOLO

model = YOLO('runs/segment/crack_seg/weights/best.pt')
results = model.predict('image.jpg', save=True)
```

### 3. 实时检测

```bash
python realtime_detect.py
```

支持摄像头实时流和视频文件输入。

### 4. 启动 Web 界面

```bash
python web_app.py
```

访问 `http://localhost:7860` 打开图形化操作界面。

### 5. 模型导出

```bash
python export_model.py
```

支持导出为 ONNX、TensorRT、OpenVINO、CoreML、TFLite 等格式。

## ⚙️ 配置参数

所有参数集中在 `config.yaml` 中管理：

- **训练参数** — 模型选择、学习率、数据增强等
- **预测参数** — 置信度阈值、NMS IoU、最大检测数等
- **批量处理** — 输出目录、裂缝严重程度分级阈值
- **实时检测** — 摄像头 ID、FPS 显示
- **Web 界面** — 服务地址与端口

## 📊 模型性能

| 指标 | 数值（参考） |
|------|-------------|
| mAP50 | 0.85 - 0.93 |
| mAP50-95 | 0.65 - 0.77 |
| 推理速度 | 10 - 60 ms/张 |
| FPS | 16 - 100 |

> 具体性能取决于所选模型规格与硬件配置

## 🐳 Docker 部署

```bash
docker-compose up -d
```

## 🔧 常见问题

**显存不足？**  
将 `batch` 调小至 8 或 4，启用 `amp=True` 混合精度训练。

**如何提升精度？**  
换用更大的模型（如 `yolov8m-seg.pt`）、增加训练轮数、增大输入尺寸。

## 📚 参考资料

- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com/)
- [YOLOv8 分割任务](https://docs.ultralytics.com/tasks/segment/)

## 📄 许可证

本项目代码采用 MIT License，YOLOv8 模型遵循 [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 许可证。
