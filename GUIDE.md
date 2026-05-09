# 墙面裂缝检测系统 - 使用指南

## 目录

1. [快速开始](#快速开始)
2. [数据准备](#数据准备)
3. [模型训练](#模型训练)
4. [模型推理](#模型推理)
5. [高级功能](#高级功能)
6. [常见问题](#常见问题)
7. [最佳实践](#最佳实践)

## 快速开始

### 第一步：安装依赖

```bash
pip install -r requirements.txt
```

### 第二步：运行快速测试

```bash
python quick_start.py
```

这将检查环境、下载预训练模型并运行测试。

### 第三步：启动主菜单

```bash
python main.py
```

或使用启动脚本：
- Windows: 双击 `start.bat`
- Linux/Mac: `bash start.sh`

## 数据准备

### 数据集结构

确保数据集按以下结构组织：

```
crack-seg/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 数据集分析

运行数据集分析脚本：

```bash
python analyze_dataset.py
```

这将生成：
- 图片尺寸分布统计
- 裂缝数量统计
- 可视化图表

## 模型训练

### 基础训练

使用默认参数训练：

```bash
python train_crack_seg.py
```

### 自定义训练参数

编辑 `config.yaml` 文件：

```yaml
train:
  model: yolov8n-seg.pt  # 可选: n/s/m/l/x
  epochs: 100
  batch: 16
  imgsz: 640
```

### 训练技巧

1. **选择合适的模型大小**
   - `yolov8n-seg.pt`: 最快，适合实时应用
   - `yolov8s-seg.pt`: 平衡速度和精度
   - `yolov8m-seg.pt`: 更高精度
   - `yolov8l-seg.pt`: 高精度
   - `yolov8x-seg.pt`: 最高精度

2. **调整批次大小**
   - GPU显存充足：增大batch（如32）
   - GPU显存不足：减小batch（如8或4）

3. **学习率调整**
   - 默认学习率通常效果良好
   - 如果训练不稳定，降低学习率

4. **数据增强**
   - 已启用常见增强方法
   - 可在配置文件中调整参数

### 监控训练过程

训练过程中会生成：
- `runs/segment/crack_seg/results.csv`: 训练指标
- `runs/segment/crack_seg/weights/best.pt`: 最佳模型
- `runs/segment/crack_seg/weights/last.pt`: 最后模型

可视化训练结果：

```bash
python visualize_results.py
```

## 模型推理

### 单张图片预测

```bash
python predict_crack_seg.py
```

选择选项1，输入图片路径。

### 批量处理

```bash
python batch_process.py
```

输入图片目录，系统会：
- 检测所有图片
- 生成详细报告（JSON/CSV/HTML）
- 保存标注后的图片
- 统计严重程度

### 实时检测

#### 摄像头检测

```bash
python realtime_detect.py
```

选择选项1，输入摄像头ID（通常为0）。

#### 视频文件检测

选择选项2，输入视频文件路径。

### Web界面

启动Web界面：

```bash
python web_app.py
```

浏览器会自动打开 `http://localhost:7860`

功能：
- 拖拽上传图片
- 实时检测
- 调整参数
- 查看详细信息

## 高级功能

### 模型导出

导出为不同格式以适应不同部署场景：

```bash
python export_model.py
```

支持格式：
- **ONNX**: 跨平台推理
- **TensorRT**: NVIDIA GPU加速
- **OpenVINO**: Intel硬件加速
- **CoreML**: iOS/macOS部署
- **TFLite**: 移动端部署

### 模型对比

对比不同模型的性能：

```bash
python compare_models.py
```

输入多个模型路径，系统会生成：
- 精度对比表格
- 速度对比图表
- 模型大小对比

### 配置管理

使用配置文件管理所有参数：

```python
from config_manager import get_config

config = get_config()
epochs = config.get('train.epochs')
```

## 常见问题

### Q1: 训练时显存不足

**解决方案**:
1. 减小batch大小
2. 使用更小的模型（如yolov8n）
3. 减小图片尺寸（如改为320）

### Q2: 训练速度慢

**解决方案**:
1. 使用GPU训练
2. 增加workers数量
3. 使用更小的模型

### Q3: 检测精度不高

**解决方案**:
1. 增加训练轮数
2. 使用更大的模型
3. 调整数据增强参数
4. 检查数据集质量

### Q4: 推理速度慢

**解决方案**:
1. 使用更小的模型
2. 导出为TensorRT格式
3. 降低输入图片尺寸
4. 使用GPU推理

### Q5: 模型过拟合

**解决方案**:
1. 增加数据增强
2. 使用更小的模型
3. 增加正则化
4. 早停（patience参数）

## 最佳实践

### 训练最佳实践

1. **数据质量优先**
   - 确保标注准确
   - 数据分布均衡
   - 充足的训练样本

2. **渐进式训练**
   - 先用小模型快速验证
   - 再用大模型提升精度

3. **监控验证集**
   - 关注验证集指标
   - 避免过拟合

4. **保存检查点**
   - 定期保存模型
   - 保留最佳模型

### 部署最佳实践

1. **选择合适的格式**
   - 服务器: PyTorch或ONNX
   - 边缘设备: TensorRT或TFLite
   - 移动端: CoreML或TFLite

2. **优化推理速度**
   - 使用FP16精度
   - 批量处理
   - 异步推理

3. **监控系统性能**
   - 记录推理时间
   - 监控资源使用
   - 定期评估精度

### 维护最佳实践

1. **版本管理**
   - 记录模型版本
   - 保存训练配置
   - 版本控制代码

2. **持续改进**
   - 收集错误案例
   - 定期重新训练
   - 更新数据集

3. **文档记录**
   - 记录实验结果
   - 文档化配置
   - 分享经验

## 性能基准

### 模型性能对比

| 模型 | mAP50 | mAP50-95 | 速度(ms) | 大小(MB) |
|------|-------|----------|----------|----------|
| YOLOv8n | 0.85 | 0.65 | 10 | 6.2 |
| YOLOv8s | 0.88 | 0.70 | 15 | 22 |
| YOLOv8m | 0.90 | 0.73 | 25 | 50 |
| YOLOv8l | 0.92 | 0.75 | 40 | 87 |
| YOLOv8x | 0.93 | 0.77 | 60 | 136 |

*注: 实际性能取决于硬件和数据集*

### 硬件要求

**最低配置**:
- CPU: 4核
- 内存: 8GB
- 存储: 10GB

**推荐配置**:
- CPU: 8核+
- GPU: NVIDIA GTX 1660+
- 内存: 16GB+
- 存储: 50GB+

**最佳配置**:
- CPU: 16核+
- GPU: NVIDIA RTX 3080+
- 内存: 32GB+
- 存储: 100GB+ SSD

## 技术支持

如有问题或建议：
1. 查看项目文档
2. 搜索常见问题
3. 提交Issue
4. 联系技术支持

---

© 2024 墙面裂缝检测系统
