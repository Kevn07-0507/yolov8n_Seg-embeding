"""
性能基准测试脚本
测试不同模型和配置的性能
"""
from ultralytics import YOLO
import time
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import font_config  # 解决中文显示异常问题

class PerformanceBenchmark:
    def __init__(self):
        """初始化性能测试器"""
        self.results = []

    def benchmark_model(self, model_path, test_images, num_runs=100, warmup_runs=10):
        """
        测试单个模型的性能

        Args:
            model_path: 模型路径
            test_images: 测试图片列表
            num_runs: 测试运行次数
            warmup_runs: 预热运行次数

        Returns:
            性能统计字典
        """
        print(f"\n测试模型: {os.path.basename(model_path)}")
        print("-"*60)

        # 加载模型
        model = YOLO(model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 预热
        print(f"预热 {warmup_runs} 次...")
        for _ in range(warmup_runs):
            for img in test_images[:3]:
                model.predict(img, verbose=False, device=device)

        # 测试推理时间
        print(f"测试 {num_runs} 次...")
        inference_times = []

        for run in range(num_runs):
            for img in test_images:
                start_time = time.time()
                results = model.predict(img, verbose=False, device=device)
                inference_time = (time.time() - start_time) * 1000  # ms
                inference_times.append(inference_time)

            if (run + 1) % 10 == 0:
                print(f"  进度: {run+1}/{num_runs}")

        # 计算统计信息
        stats = {
            'model_name': os.path.basename(model_path),
            'model_path': model_path,
            'device': device,
            'num_runs': num_runs * len(test_images),
            'mean_time_ms': np.mean(inference_times),
            'std_time_ms': np.std(inference_times),
            'min_time_ms': np.min(inference_times),
            'max_time_ms': np.max(inference_times),
            'median_time_ms': np.median(inference_times),
            'p95_time_ms': np.percentile(inference_times, 95),
            'p99_time_ms': np.percentile(inference_times, 99),
            'fps': 1000 / np.mean(inference_times),
        }

        # 获取模型大小
        if os.path.exists(model_path):
            stats['model_size_mb'] = os.path.getsize(model_path) / (1024 * 1024)

        print(f"\n结果:")
        print(f"  平均推理时间: {stats['mean_time_ms']:.2f} ms")
        print(f"  标准差: {stats['std_time_ms']:.2f} ms")
        print(f"  FPS: {stats['fps']:.2f}")

        return stats

    def benchmark_batch_sizes(self, model_path, test_image, batch_sizes=[1, 2, 4, 8, 16]):
        """
        测试不同batch大小的性能

        Args:
            model_path: 模型路径
            test_image: 测试图片
            batch_sizes: batch大小列表

        Returns:
            性能统计列表
        """
        print(f"\n测试不同batch大小的性能")
        print("-"*60)

        model = YOLO(model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        results = []

        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")

            # 创建batch
            batch_images = [test_image] * batch_size

            # 预热
            for _ in range(5):
                model.predict(batch_images, verbose=False, device=device)

            # 测试
            times = []
            for _ in range(20):
                start_time = time.time()
                model.predict(batch_images, verbose=False, device=device)
                elapsed = (time.time() - start_time) * 1000
                times.append(elapsed)

            avg_time = np.mean(times)
            throughput = batch_size * 1000 / avg_time

            result = {
                'batch_size': batch_size,
                'avg_time_ms': avg_time,
                'time_per_image_ms': avg_time / batch_size,
                'throughput_fps': throughput,
            }

            results.append(result)

            print(f"  平均时间: {avg_time:.2f} ms")
            print(f"  每张图片: {avg_time/batch_size:.2f} ms")
            print(f"  吞吐量: {throughput:.2f} FPS")

        return results

    def benchmark_image_sizes(self, model_path, test_image, image_sizes=[320, 416, 512, 640, 800]):
        """
        测试不同输入尺寸的性能

        Args:
            model_path: 模型路径
            test_image: 测试图片
            image_sizes: 图片尺寸列表

        Returns:
            性能统计列表
        """
        print(f"\n测试不同输入尺寸的性能")
        print("-"*60)

        model = YOLO(model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 读取原始图片
        img = cv2.imread(test_image)

        results = []

        for size in image_sizes:
            print(f"\n图片尺寸: {size}x{size}")

            # 调整图片大小
            resized = cv2.resize(img, (size, size))

            # 预热
            for _ in range(5):
                model.predict(resized, verbose=False, device=device)

            # 测试
            times = []
            for _ in range(20):
                start_time = time.time()
                model.predict(resized, verbose=False, device=device)
                elapsed = (time.time() - start_time) * 1000
                times.append(elapsed)

            avg_time = np.mean(times)

            result = {
                'image_size': size,
                'avg_time_ms': avg_time,
                'fps': 1000 / avg_time,
            }

            results.append(result)

            print(f"  平均时间: {avg_time:.2f} ms")
            print(f"  FPS: {1000/avg_time:.2f}")

        return results

    def plot_results(self, results, output_dir='benchmark_results'):
        """
        绘制性能测试结果

        Args:
            results: 结果列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(results)

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('性能基准测试结果', fontsize=16)

        # 1. 平均推理时间
        ax = axes[0, 0]
        ax.bar(df['model_name'], df['mean_time_ms'], color='#1f77b4')
        ax.set_ylabel('时间 (ms)')
        ax.set_title('平均推理时间')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        # 2. FPS
        ax = axes[0, 1]
        ax.bar(df['model_name'], df['fps'], color='#ff7f0e')
        ax.set_ylabel('FPS')
        ax.set_title('每秒帧数')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        # 3. 模型大小
        ax = axes[1, 0]
        if 'model_size_mb' in df.columns:
            ax.bar(df['model_name'], df['model_size_mb'], color='#2ca02c')
            ax.set_ylabel('大小 (MB)')
            ax.set_title('模型大小')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        # 4. 推理时间分布
        ax = axes[1, 1]
        ax.boxplot([df['mean_time_ms']], labels=['推理时间'])
        ax.set_ylabel('时间 (ms)')
        ax.set_title('推理时间分布')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        plot_path = os.path.join(output_dir, 'benchmark_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存: {plot_path}")

        plt.show()

def main():
    print("性能基准测试工具")
    print("="*60)

    benchmark = PerformanceBenchmark()

    print("\n请选择测试类型:")
    print("1. 测试单个模型")
    print("2. 对比多个模型")
    print("3. 测试不同batch大小")
    print("4. 测试不同输入尺寸")

    choice = input("\n请输入选项 (1-4): ").strip()

    if choice == '1':
        model_path = input("请输入模型路径: ").strip()
        test_dir = input("请输入测试图片目录 (默认: crack-seg/test/images): ").strip()
        test_dir = test_dir if test_dir else 'crack-seg/test/images'

        if os.path.exists(model_path) and os.path.exists(test_dir):
            test_images = [str(f) for f in Path(test_dir).glob('*.jpg')][:10]
            stats = benchmark.benchmark_model(model_path, test_images)
        else:
            print("路径不存在!")

    elif choice == '2':
        print("\n输入模型路径（每行一个，输入空行结束）:")
        model_paths = []
        while True:
            path = input().strip()
            if not path:
                break
            if os.path.exists(path):
                model_paths.append(path)
            else:
                print(f"模型不存在: {path}")

        if model_paths:
            test_dir = 'crack-seg/test/images'
            if os.path.exists(test_dir):
                test_images = [str(f) for f in Path(test_dir).glob('*.jpg')][:10]
                results = []
                for model_path in model_paths:
                    stats = benchmark.benchmark_model(model_path, test_images, num_runs=50)
                    results.append(stats)

                # 保存结果
                df = pd.DataFrame(results)
                df.to_csv('benchmark_results/comparison.csv', index=False)
                print("\n结果已保存: benchmark_results/comparison.csv")

                # 绘制图表
                benchmark.plot_results(results)

    elif choice == '3':
        model_path = input("请输入模型路径: ").strip()
        test_image = input("请输入测试图片路径: ").strip()

        if os.path.exists(model_path) and os.path.exists(test_image):
            results = benchmark.benchmark_batch_sizes(model_path, test_image)

            # 保存结果
            df = pd.DataFrame(results)
            os.makedirs('benchmark_results', exist_ok=True)
            df.to_csv('benchmark_results/batch_sizes.csv', index=False)
            print("\n结果已保存: benchmark_results/batch_sizes.csv")

    elif choice == '4':
        model_path = input("请输入模型路径: ").strip()
        test_image = input("请输入测试图片路径: ").strip()

        if os.path.exists(model_path) and os.path.exists(test_image):
            results = benchmark.benchmark_image_sizes(model_path, test_image)

            # 保存结果
            df = pd.DataFrame(results)
            os.makedirs('benchmark_results', exist_ok=True)
            df.to_csv('benchmark_results/image_sizes.csv', index=False)
            print("\n结果已保存: benchmark_results/image_sizes.csv")

    else:
        print("无效的选项!")

if __name__ == '__main__':
    main()
