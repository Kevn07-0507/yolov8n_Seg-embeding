"""
系统测试脚本
测试各个模块的基本功能
"""
import os
import sys
import importlib

class SystemTester:
    def __init__(self):
        """初始化测试器"""
        self.passed = 0
        self.failed = 0
        self.tests = []

    def test_imports(self):
        """测试依赖导入"""
        print("\n" + "="*60)
        print("测试依赖导入")
        print("="*60)

        dependencies = [
            'torch',
            'torchvision',
            'cv2',
            'numpy',
            'pandas',
            'matplotlib',
            'PIL',
            'yaml',
            'tqdm',
            'gradio',
            'ultralytics',
        ]

        for dep in dependencies:
            try:
                importlib.import_module(dep)
                print(f"✓ {dep:20s} - 已安装")
                self.passed += 1
            except ImportError:
                print(f"✗ {dep:20s} - 未安装")
                self.failed += 1

    def test_cuda(self):
        """测试CUDA"""
        print("\n" + "="*60)
        print("测试CUDA")
        print("="*60)

        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                print(f"✓ CUDA可用")
                print(f"  设备数量: {torch.cuda.device_count()}")
                print(f"  设备名称: {torch.cuda.get_device_name(0)}")
                self.passed += 1
            else:
                print(f"⚠ CUDA不可用（将使用CPU）")
                self.passed += 1
        except Exception as e:
            print(f"✗ CUDA测试失败: {e}")
            self.failed += 1

    def test_dataset(self):
        """测试数据集"""
        print("\n" + "="*60)
        print("测试数据集")
        print("="*60)

        dataset_path = 'crack-seg'
        data_yaml = os.path.join(dataset_path, 'data.yaml')

        if os.path.exists(dataset_path):
            print(f"✓ 数据集目录存在: {dataset_path}")
            self.passed += 1
        else:
            print(f"✗ 数据集目录不存在: {dataset_path}")
            self.failed += 1
            return

        if os.path.exists(data_yaml):
            print(f"✓ 数据集配置存在: {data_yaml}")
            self.passed += 1
        else:
            print(f"✗ 数据集配置不存在: {data_yaml}")
            self.failed += 1

        # 检查子目录
        subdirs = ['train/images', 'train/labels', 'valid/images', 'valid/labels', 'test/images']
        for subdir in subdirs:
            path = os.path.join(dataset_path, subdir)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if not f.startswith('.')])
                print(f"✓ {subdir:20s} - {count} 个文件")
                self.passed += 1
            else:
                print(f"✗ {subdir:20s} - 不存在")
                self.failed += 1

    def test_scripts(self):
        """测试脚本文件"""
        print("\n" + "="*60)
        print("测试脚本文件")
        print("="*60)

        scripts = [
            'main.py',
            'train_crack_seg.py',
            'predict_crack_seg.py',
            'batch_process.py',
            'realtime_detect.py',
            'web_app.py',
            'export_model.py',
            'compare_models.py',
            'analyze_dataset.py',
            'visualize_results.py',
            'quick_start.py',
            'config_manager.py',
            'utils.py',
            'logger.py',
            'benchmark.py',
            'preview_augmentation.py',
        ]

        for script in scripts:
            if os.path.exists(script):
                print(f"✓ {script}")
                self.passed += 1
            else:
                print(f"✗ {script} - 不存在")
                self.failed += 1

    def test_config(self):
        """测试配置文件"""
        print("\n" + "="*60)
        print("测试配置文件")
        print("="*60)

        config_files = [
            'config.yaml',
            'requirements.txt',
            'README.md',
            'GUIDE.md',
        ]

        for config in config_files:
            if os.path.exists(config):
                print(f"✓ {config}")
                self.passed += 1
            else:
                print(f"✗ {config} - 不存在")
                self.failed += 1

    def test_model_loading(self):
        """测试模型加载"""
        print("\n" + "="*60)
        print("测试模型加载")
        print("="*60)

        try:
            from ultralytics import YOLO
            print("正在下载预训练模型...")
            model = YOLO('yolov8n-seg.pt')
            print(f"✓ 模型加载成功")
            self.passed += 1
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            self.failed += 1

    def test_inference(self):
        """测试推理"""
        print("\n" + "="*60)
        print("测试推理")
        print("="*60)

        try:
            from ultralytics import YOLO
            import numpy as np

            # 创建测试图片
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            model = YOLO('yolov8n-seg.pt')
            results = model.predict(test_image, verbose=False)

            print(f"✓ 推理测试成功")
            self.passed += 1
        except Exception as e:
            print(f"✗ 推理测试失败: {e}")
            self.failed += 1

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("墙面裂缝检测系统 - 系统测试")
        print("="*60)

        self.test_imports()
        self.test_cuda()
        self.test_dataset()
        self.test_scripts()
        self.test_config()
        self.test_model_loading()
        self.test_inference()

        # 打印总结
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        print(f"通过: {self.passed}")
        print(f"失败: {self.failed}")
        print(f"总计: {self.passed + self.failed}")

        if self.failed == 0:
            print("\n✓ 所有测试通过！系统准备就绪。")
            return True
        else:
            print(f"\n⚠ {self.failed} 个测试失败，请检查上述错误。")
            return False

def main():
    tester = SystemTester()
    success = tester.run_all_tests()

    print("\n" + "="*60)
    if success:
        print("系统测试完成！可以开始使用。")
        print("\n下一步:")
        print("1. 运行主菜单: python main.py")
        print("2. 训练模型: python train_crack_seg.py")
        print("3. 启动Web界面: python web_app.py")
    else:
        print("系统测试未完全通过，请先解决上述问题。")
        print("\n建议:")
        print("1. 安装缺失的依赖: pip install -r requirements.txt")
        print("2. 检查数据集是否正确解压")
        print("3. 查看FAQ.md获取帮助")

    print("="*60)

if __name__ == '__main__':
    main()
