"""
模型对比脚本
对比不同模型的性能
"""
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import os
import font_config  # 解决中文显示异常问题

class ModelComparison:
    def __init__(self, data_yaml='crack-seg/data.yaml'):
        """
        初始化模型对比

        Args:
            data_yaml: 数据集配置文件
        """
        self.data_yaml = data_yaml
        self.results = []

    def evaluate_model(self, model_path, model_name=None):
        """
        评估单个模型

        Args:
            model_path: 模型路径
            model_name: 模型名称

        Returns:
            评估结果字典
        """
        if not os.path.exists(model_path):
            print(f"模型不存在: {model_path}")
            return None

        if model_name is None:
            model_name = os.path.basename(model_path)

        print(f"\n评估模型: {model_name}")
        print("-"*60)

        try:
            # 加载模型
            model = YOLO(model_path)

            # 评估性能
            start_time = time.time()
            metrics = model.val(data=self.data_yaml, verbose=False)
            eval_time = time.time() - start_time

            # 获取模型大小
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB

            # 提取指标
            result = {
                'model_name': model_name,
                'model_path': model_path,
                'model_size_mb': round(model_size, 2),
                'eval_time_s': round(eval_time, 2),
                'box_map50': round(metrics.box.map50, 4),
                'box_map50_95': round(metrics.box.map, 4),
                'box_precision': round(metrics.box.mp, 4),
                'box_recall': round(metrics.box.mr, 4),
                'seg_map50': round(metrics.seg.map50, 4),
                'seg_map50_95': round(metrics.seg.map, 4),
                'seg_precision': round(metrics.seg.mp, 4),
                'seg_recall': round(metrics.seg.mr, 4),
            }

            # 测试推理速度
            print("测试推理速度...")
            test_image = 'crack-seg/test/images'
            if os.path.exists(test_image):
                test_files = [f for f in os.listdir(test_image) if f.endswith(('.jpg', '.png'))]
                if test_files:
                    test_img = os.path.join(test_image, test_files[0])

                    # 预热
                    for _ in range(3):
                        model.predict(test_img, verbose=False)

                    # 测速
                    start_time = time.time()
                    num_runs = 10
                    for _ in range(num_runs):
                        model.predict(test_img, verbose=False)
                    inference_time = (time.time() - start_time) / num_runs * 1000  # ms

                    result['inference_time_ms'] = round(inference_time, 2)
                    result['fps'] = round(1000 / inference_time, 2)

            self.results.append(result)

            print(f"✓ 评估完成")
            print(f"  - mAP50 (Seg): {result['seg_map50']:.4f}")
            print(f"  - mAP50-95 (Seg): {result['seg_map50_95']:.4f}")
            print(f"  - 推理时间: {result.get('inference_time_ms', 'N/A')} ms")

            return result

        except Exception as e:
            print(f"✗ 评估失败: {e}")
            return None

    def compare_models(self, model_paths):
        """
        对比多个模型

        Args:
            model_paths: 模型路径列表或字典 {名称: 路径}
        """
        self.results = []

        if isinstance(model_paths, dict):
            for name, path in model_paths.items():
                self.evaluate_model(path, name)
        else:
            for path in model_paths:
                self.evaluate_model(path)

        if not self.results:
            print("没有成功评估的模型")
            return

        # 生成对比报告
        self.generate_report()

    def generate_report(self):
        """生成对比报告"""
        if not self.results:
            print("没有结果可生成报告")
            return

        print("\n" + "="*80)
        print("模型对比报告")
        print("="*80)

        # 创建DataFrame
        df = pd.DataFrame(self.results)

        # 打印表格
        print("\n【性能指标对比】")
        print(df.to_string(index=False))

        # 保存CSV
        output_dir = 'comparison_results'
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, 'model_comparison.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ CSV报告已保存: {csv_path}")

        # 生成可视化
        self.plot_comparison(df, output_dir)

    def plot_comparison(self, df, output_dir):
        """
        生成对比图表

        Args:
            df: 结果DataFrame
            output_dir: 输出目录
        """
        print("\n生成对比图表...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('模型性能对比', fontsize=16)

        model_names = df['model_name'].tolist()

        # 1. mAP50对比
        ax = axes[0, 0]
        x = range(len(model_names))
        ax.bar(x, df['seg_map50'], alpha=0.7, label='Seg mAP50', color='#1f77b4')
        ax.bar([i+0.3 for i in x], df['box_map50'], alpha=0.7, label='Box mAP50', color='#ff7f0e', width=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('mAP50')
        ax.set_title('mAP@0.5 对比')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. mAP50-95对比
        ax = axes[0, 1]
        ax.bar(x, df['seg_map50_95'], alpha=0.7, label='Seg mAP50-95', color='#1f77b4')
        ax.bar([i+0.3 for i in x], df['box_map50_95'], alpha=0.7, label='Box mAP50-95', color='#ff7f0e', width=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('mAP50-95')
        ax.set_title('mAP@0.5:0.95 对比')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Precision和Recall对比
        ax = axes[0, 2]
        ax.bar(x, df['seg_precision'], alpha=0.7, label='Precision', color='#2ca02c')
        ax.bar([i+0.3 for i in x], df['seg_recall'], alpha=0.7, label='Recall', color='#d62728', width=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('Precision & Recall 对比')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 模型大小对比
        ax = axes[1, 0]
        ax.bar(x, df['model_size_mb'], color='#9467bd')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('Size (MB)')
        ax.set_title('模型大小对比')
        ax.grid(True, alpha=0.3)

        # 5. 推理速度对比
        ax = axes[1, 1]
        if 'inference_time_ms' in df.columns:
            ax.bar(x, df['inference_time_ms'], color='#8c564b')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel('Time (ms)')
            ax.set_title('推理时间对比')
            ax.grid(True, alpha=0.3)

        # 6. FPS对比
        ax = axes[1, 2]
        if 'fps' in df.columns:
            ax.bar(x, df['fps'], color='#e377c2')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel('FPS')
            ax.set_title('FPS对比')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        plot_path = os.path.join(output_dir, 'model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ 对比图表已保存: {plot_path}")

        plt.show()

def main():
    print("模型对比工具")
    print("="*60)

    # 获取要对比的模型
    print("\n请输入要对比的模型路径（每行一个，输入空行结束）:")
    model_paths = {}
    idx = 1

    while True:
        path = input(f"模型 {idx} 路径: ").strip()
        if not path:
            break

        if os.path.exists(path):
            name = input(f"模型 {idx} 名称 (默认: {os.path.basename(path)}): ").strip()
            if not name:
                name = os.path.basename(path)
            model_paths[name] = path
            idx += 1
        else:
            print(f"模型不存在: {path}")

    if not model_paths:
        print("\n使用默认模型进行对比...")
        # 查找所有训练好的模型
        models_dir = 'runs/segment'
        if os.path.exists(models_dir):
            for exp_dir in os.listdir(models_dir):
                best_pt = os.path.join(models_dir, exp_dir, 'weights', 'best.pt')
                if os.path.exists(best_pt):
                    model_paths[exp_dir] = best_pt

    if not model_paths:
        print("未找到可对比的模型")
        return

    print(f"\n将对比以下模型:")
    for name, path in model_paths.items():
        print(f"  - {name}: {path}")

    # 创建对比器
    comparator = ModelComparison()

    # 开始对比
    comparator.compare_models(model_paths)

    print("\n对比完成!")

if __name__ == '__main__':
    main()
