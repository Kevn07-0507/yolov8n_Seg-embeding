"""
配置管理模块
读取和管理配置文件
"""
import yaml
import os
from pathlib import Path

class Config:
    """配置管理类"""

    def __init__(self, config_path='config.yaml'):
        """
        初始化配置

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            print(f"警告: 配置文件不存在 - {self.config_path}")
            return self.get_default_config()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"错误: 无法加载配置文件 - {e}")
            return self.get_default_config()

    def get_default_config(self):
        """获取默认配置"""
        return {
            'train': {
                'model': 'yolov8n-seg.pt',
                'data': 'crack-seg/data.yaml',
                'epochs': 100,
                'batch': 16,
                'imgsz': 640,
                'device': '',
                'workers': 4,
            },
            'predict': {
                'conf': 0.25,
                'iou': 0.7,
                'max_det': 300,
            },
            'paths': {
                'dataset': 'crack-seg',
                'models': 'runs/segment',
                'results': 'results',
            }
        }

    def get(self, key, default=None):
        """
        获取配置值

        Args:
            key: 配置键，支持点号分隔的嵌套键 (如 'train.epochs')
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key, value):
        """
        设置配置值

        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path=None):
        """
        保存配置到文件

        Args:
            path: 保存路径，默认为原路径
        """
        save_path = path or self.config_path

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            print(f"配置已保存: {save_path}")
            return True
        except Exception as e:
            print(f"错误: 无法保存配置 - {e}")
            return False

    def print_config(self):
        """打印当前配置"""
        print("\n当前配置:")
        print("="*60)
        print(yaml.dump(self.config, default_flow_style=False, allow_unicode=True))
        print("="*60)

# 全局配置实例
_config = None

def get_config(config_path='config.yaml'):
    """
    获取全局配置实例

    Args:
        config_path: 配置文件路径

    Returns:
        Config实例
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config

if __name__ == '__main__':
    # 测试配置管理
    config = Config()
    config.print_config()

    # 测试获取配置
    print(f"\n训练轮数: {config.get('train.epochs')}")
    print(f"置信度阈值: {config.get('predict.conf')}")

    # 测试设置配置
    config.set('train.epochs', 150)
    print(f"\n修改后的训练轮数: {config.get('train.epochs')}")
