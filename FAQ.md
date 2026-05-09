# 常见问题解答 (FAQ)

## 目录

1. [安装和环境](#安装和环境)
2. [数据集相关](#数据集相关)
3. [训练相关](#训练相关)
4. [推理相关](#推理相关)
5. [性能优化](#性能优化)
6. [错误处理](#错误处理)
7. [部署相关](#部署相关)

---

## 安装和环境

### Q1: 如何安装依赖？

**A**: 运行以下命令：

```bash
pip install -r requirements.txt
```

如果遇到网络问题，可以使用国内镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 是否必须使用GPU？

**A**: 不是必须的，但强烈推荐。

- **CPU训练**: 可行但非常慢，100轮可能需要数天
- **GPU训练**: 推荐，100轮约1-2小时（取决于GPU性能）

### Q3: 支持哪些Python版本？

**A**: Python 3.8 - 3.11

推荐使用Python 3.9或3.10以获得最佳兼容性。

### Q4: CUDA版本要求？

**A**:
- PyTorch 2.0+: CUDA 11.7+
- 推荐: CUDA 11.8或12.1

检查CUDA版本：
```bash
nvidia-smi
```

---

## 数据集相关

### Q5: 数据集格式是什么？

**A**: YOLO分割格式

每个图片对应一个txt标注文件，格式为：
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

坐标为归一化坐标（0-1之间）。

### Q6: 如何准备自己的数据集？

**A**:

1. 按照YOLO格式组织数据：
```
my_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

2. 创建data.yaml：
```yaml
path: my_dataset
train: train/images
val: valid/images
names:
  0: crack
```

3. 修改训练脚本中的数据路径

### Q7: 数据集太小怎么办？

**A**:

1. **数据增强**: 已内置多种增强方法
2. **迁移学习**: 使用预训练模型
3. **收集更多数据**: 最根本的解决方案
4. **合成数据**: 使用GAN等方法生成

---

## 训练相关

### Q8: 训练时显存不足怎么办？

**A**:

1. **减小batch大小**: 从16改为8或4
2. **使用更小的模型**: yolov8n而不是yolov8x
3. **减小图片尺寸**: 从640改为320
4. **使用梯度累积**: 在代码中添加
5. **清理显存**:
```python
torch.cuda.empty_cache()
```

### Q9: 训练多少轮合适？

**A**:

- **默认**: 100轮通常足够
- **观察验证集**: 如果验证集loss不再下降，可以提前停止
- **使用早停**: patience=20（默认已设置）
- **过拟合**: 如果训练集很好但验证集差，减少轮数

### Q10: 如何判断模型是否过拟合？

**A**:

观察训练曲线：
- 训练loss持续下降，验证loss上升 → 过拟合
- 训练和验证loss都下降 → 正常
- 训练和验证loss都不变 → 欠拟合

解决过拟合：
1. 增加数据增强
2. 使用更小的模型
3. 添加正则化
4. 减少训练轮数

### Q11: 训练中断了怎么办？

**A**:

YOLOv8支持断点续训：

```python
model = YOLO('runs/segment/crack_seg/weights/last.pt')
model.train(resume=True)
```

### Q12: 如何调整学习率？

**A**:

在config.yaml中修改：

```yaml
train:
  lr0: 0.01      # 初始学习率
  lrf: 0.01      # 最终学习率
  momentum: 0.937
  weight_decay: 0.0005
```

---

## 推理相关

### Q13: 如何提高检测精度？

**A**:

1. **降低置信度阈值**: 从0.25降到0.15
2. **使用更大的模型**: yolov8m或yolov8l
3. **增加训练数据**: 特别是困难样本
4. **调整NMS阈值**: iou参数
5. **使用TTA**: 测试时增强

### Q14: 检测速度太慢怎么办？

**A**:

1. **使用更小的模型**: yolov8n
2. **减小输入尺寸**: 从640改为320
3. **导出为TensorRT**: 显著提速
4. **使用FP16**: 半精度推理
5. **批量处理**: 而不是逐张处理

### Q15: 如何批量处理图片？

**A**:

```bash
python batch_process.py
```

输入图片目录，系统会自动处理所有图片并生成报告。

---

## 性能优化

### Q16: 如何加速训练？

**A**:

1. **使用更好的GPU**: RTX 3080/4090等
2. **增加batch大小**: 充分利用GPU
3. **使用混合精度**: AMP (自动混合精度)
4. **多GPU训练**: 使用DDP
5. **优化数据加载**: 增加workers数量

### Q17: 如何减小模型大小？

**A**:

1. **使用更小的模型**: yolov8n (6MB)
2. **模型量化**: INT8量化
3. **模型剪枝**: 移除不重要的权重
4. **知识蒸馏**: 用大模型训练小模型

### Q18: 如何优化推理速度？

**A**:

| 方法 | 加速比 | 精度损失 |
|------|--------|----------|
| FP16 | 1.5-2x | 极小 |
| INT8 | 2-4x | 小 |
| TensorRT | 3-5x | 极小 |
| 模型剪枝 | 1.5-3x | 中等 |

推荐组合: TensorRT + FP16

---

## 错误处理

### Q19: 遇到"CUDA out of memory"错误

**A**:

```python
# 方法1: 减小batch
batch = 8  # 或更小

# 方法2: 清理显存
import torch
torch.cuda.empty_cache()

# 方法3: 使用CPU
device = 'cpu'
```

### Q20: 遇到"No module named 'ultralytics'"

**A**:

```bash
pip install ultralytics
```

### Q21: 训练时loss为NaN

**A**:

原因：
1. 学习率过大
2. 数据标注错误
3. 数值溢出

解决：
1. 降低学习率
2. 检查数据集
3. 使用混合精度训练

### Q22: 预测结果为空

**A**:

1. **降低置信度阈值**: conf=0.1
2. **检查模型是否正确加载**
3. **检查输入图片格式**
4. **查看模型训练效果**

---

## 部署相关

### Q23: 如何部署到生产环境？

**A**:

1. **导出模型**:
```bash
python export_model.py
```

2. **选择部署方式**:
   - 服务器: ONNX/TensorRT
   - 边缘设备: TensorRT/OpenVINO
   - 移动端: TFLite/CoreML
   - Web: ONNX.js

3. **容器化**: 使用Docker

### Q24: 如何创建REST API？

**A**:

使用FastAPI:

```python
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO

app = FastAPI()
model = YOLO('best.pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    # 处理图片
    results = model.predict(contents)
    return {"results": results}
```

### Q25: 如何监控模型性能？

**A**:

1. **记录推理时间**
2. **记录检测结果**
3. **定期评估精度**
4. **收集错误案例**
5. **使用监控工具**: Prometheus, Grafana

### Q26: 如何更新模型？

**A**:

1. **收集新数据**
2. **重新训练**
3. **A/B测试**: 对比新旧模型
4. **灰度发布**: 逐步替换
5. **回滚机制**: 保留旧模型

---

## 其他问题

### Q27: 如何贡献代码？

**A**:

1. Fork项目
2. 创建分支
3. 提交代码
4. 发起Pull Request

### Q28: 如何报告Bug？

**A**:

提供以下信息：
1. 错误描述
2. 复现步骤
3. 环境信息（Python版本、PyTorch版本等）
4. 错误日志
5. 相关代码

### Q29: 商业使用是否需要授权？

**A**:

YOLOv8使用AGPL-3.0许可证：
- 开源项目: 免费使用
- 商业项目: 需要购买商业许可或开源代码

详情请查看Ultralytics官网。

### Q30: 如何获取技术支持？

**A**:

1. 查看文档: README.md, GUIDE.md
2. 搜索FAQ
3. 查看Issues
4. 提交新Issue
5. 联系维护者

---

**最后更新**: 2024-05-03

如有其他问题，欢迎提Issue或联系技术支持。
