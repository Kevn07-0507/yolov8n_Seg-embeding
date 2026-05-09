# 快速参考指南

## 一键启动

### Windows
```bash
start.bat
```

### Linux/Mac
```bash
bash start.sh
```

### Python
```bash
python main.py
```

## 常用命令

### 训练
```bash
# 基础训练
python train_crack_seg.py

# 自定义参数（修改config.yaml）
python train_crack_seg.py
```

### 预测
```bash
# 单张图片
python predict_crack_seg.py

# 批量处理
python batch_process.py

# 实时检测
python realtime_detect.py

# Web界面
python web_app.py
```

### 分析
```bash
# 数据集分析
python analyze_dataset.py

# 训练结果可视化
python visualize_results.py

# 模型对比
python compare_models.py

# 性能测试
python benchmark.py
```

### 导出
```bash
# 模型导出
python export_model.py
```

## 配置文件

### config.yaml
```yaml
train:
  model: yolov8n-seg.pt
  epochs: 100
  batch: 16
  imgsz: 640

predict:
  conf: 0.25
  iou: 0.7
```

## 目录结构

```
embed/
├── crack-seg/          # 数据集
├── runs/               # 训练结果
├── results/            # 预测结果
├── logs/               # 日志文件
└── *.py               # Python脚本
```

## 核心脚本

| 脚本 | 功能 | 命令 |
|------|------|------|
| main.py | 主菜单 | `python main.py` |
| train_crack_seg.py | 训练 | `python train_crack_seg.py` |
| predict_crack_seg.py | 预测 | `python predict_crack_seg.py` |
| batch_process.py | 批量处理 | `python batch_process.py` |
| web_app.py | Web界面 | `python web_app.py` |
| realtime_detect.py | 实时检测 | `python realtime_detect.py` |

## 快速问题解决

### 显存不足
```yaml
# config.yaml
train:
  batch: 8  # 减小batch
  imgsz: 320  # 减小图片尺寸
```

### 训练太慢
- 使用GPU
- 增加workers
- 使用更小的模型

### 检测精度低
- 增加训练轮数
- 使用更大的模型
- 降低置信度阈值

### 推理速度慢
- 使用更小的模型
- 导出为TensorRT
- 减小输入尺寸

## 模型选择

| 模型 | 大小 | 速度 | 精度 | 适用场景 |
|------|------|------|------|----------|
| yolov8n | 6MB | 最快 | 中等 | 实时应用 |
| yolov8s | 22MB | 快 | 良好 | 平衡 |
| yolov8m | 50MB | 中等 | 很好 | 高精度 |
| yolov8l | 87MB | 慢 | 优秀 | 离线处理 |
| yolov8x | 136MB | 最慢 | 最佳 | 最高精度 |

## 导出格式

| 格式 | 用途 | 命令 |
|------|------|------|
| ONNX | 跨平台 | 选项1 |
| TensorRT | NVIDIA GPU | 选项2 |
| OpenVINO | Intel硬件 | 选项3 |
| CoreML | iOS/macOS | 选项4 |
| TFLite | 移动端 | 选项5 |

## 性能优化

### 训练优化
- 使用GPU
- 增大batch
- 混合精度训练
- 多GPU训练

### 推理优化
- 模型量化
- TensorRT加速
- 批量推理
- 异步处理

## 常见错误

### CUDA out of memory
```bash
# 解决方案
batch: 4  # 减小batch
```

### No module named 'xxx'
```bash
pip install -r requirements.txt
```

### 训练loss为NaN
```yaml
# 降低学习率
train:
  lr0: 0.001
```

## 文档链接

- [README.md](README.md) - 项目说明
- [GUIDE.md](GUIDE.md) - 详细指南
- [FAQ.md](FAQ.md) - 常见问题
- [CHANGELOG.md](CHANGELOG.md) - 更新日志

## 技术支持

- 查看文档
- 搜索FAQ
- 提交Issue
- 联系维护者

---

**提示**: 使用 `python main.py` 启动交互式菜单，更方便！
