# 项目文件清单

## 核心脚本 (10个)

1. **main.py** - 主启动脚本，提供交互式菜单
2. **train_crack_seg.py** - 模型训练脚本
3. **predict_crack_seg.py** - 模型预测脚本
4. **batch_process.py** - 批量处理脚本
5. **realtime_detect.py** - 实时检测脚本
6. **web_app.py** - Web界面脚本
7. **export_model.py** - 模型导出脚本
8. **compare_models.py** - 模型对比脚本
9. **analyze_dataset.py** - 数据集分析脚本
10. **visualize_results.py** - 结果可视化脚本

## 辅助脚本 (2个)

11. **quick_start.py** - 快速开始脚本
12. **config_manager.py** - 配置管理模块

## 配置文件 (2个)

13. **config.yaml** - 主配置文件
14. **requirements.txt** - Python依赖列表

## 文档文件 (2个)

15. **README.md** - 项目说明文档
16. **GUIDE.md** - 详细使用指南

## 启动脚本 (2个)

17. **start.bat** - Windows启动脚本
18. **start.sh** - Linux/Mac启动脚本

## 数据集

19. **crack-seg/** - 裂缝数据集目录
    - data.yaml - 数据集配置
    - train/ - 训练集 (3717张图片)
    - valid/ - 验证集 (112张图片)
    - test/ - 测试集 (200张图片)

---

**总计**: 18个脚本/配置文件 + 1个数据集

## 功能覆盖

✅ 数据准备和分析
✅ 模型训练
✅ 模型评估
✅ 单张/批量预测
✅ 实时检测
✅ Web界面
✅ 模型导出和优化
✅ 模型对比
✅ 结果可视化
✅ 配置管理
✅ 完整文档

## 使用流程

1. 运行 `python quick_start.py` 检查环境
2. 运行 `python analyze_dataset.py` 分析数据
3. 运行 `python train_crack_seg.py` 训练模型
4. 运行 `python predict_crack_seg.py` 进行预测
5. 运行 `python web_app.py` 启动Web界面

或直接运行 `python main.py` 使用交互式菜单。
