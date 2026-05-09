"""
日志管理模块
提供统一的日志记录功能
"""
import logging
import os
from datetime import datetime
from pathlib import Path

class Logger:
    """日志管理类"""

    def __init__(self, name='crack_detection', log_dir='logs', level=logging.INFO):
        """
        初始化日志器

        Args:
            name: 日志器名称
            log_dir: 日志目录
            level: 日志级别
        """
        self.name = name
        self.log_dir = log_dir
        self.level = level

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 创建日志器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # 避免重复添加处理器
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """设置日志处理器"""
        # 文件处理器
        log_file = os.path.join(
            self.log_dir,
            f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        )

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.level)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)

        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, message):
        """记录调试信息"""
        self.logger.debug(message)

    def info(self, message):
        """记录一般信息"""
        self.logger.info(message)

    def warning(self, message):
        """记录警告信息"""
        self.logger.warning(message)

    def error(self, message):
        """记录错误信息"""
        self.logger.error(message)

    def critical(self, message):
        """记录严重错误信息"""
        self.logger.critical(message)

    def exception(self, message):
        """记录异常信息（包含堆栈跟踪）"""
        self.logger.exception(message)

# 全局日志器实例
_logger = None

def get_logger(name='crack_detection', log_dir='logs', level=logging.INFO):
    """
    获取全局日志器实例

    Args:
        name: 日志器名称
        log_dir: 日志目录
        level: 日志级别

    Returns:
        Logger实例
    """
    global _logger
    if _logger is None:
        _logger = Logger(name, log_dir, level)
    return _logger

if __name__ == '__main__':
    # 测试日志功能
    logger = get_logger()

    logger.debug("这是调试信息")
    logger.info("这是一般信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    logger.critical("这是严重错误信息")

    try:
        1 / 0
    except Exception as e:
        logger.exception("捕获到异常")
