"""日志配置"""
import logging
import os


def setup_logging(level: str = "INFO") -> None:
    """配置全局日志

    Args:
        level: 日志级别，如 "DEBUG", "INFO", "WARNING"
    """
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 根 logger
    root_logger = logging.getLogger("auto_slr")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_logger.addHandler(console_handler)

    # 文件 handler
    from datetime import datetime
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # 文件始终记录 DEBUG 级别
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_logger.addHandler(file_handler)

    root_logger.info(f"日志系统初始化完成，级别: {level}，日志文件: {log_file}")
