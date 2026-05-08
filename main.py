"""Automatic Systematic Literature Review - 主入口

使用 multi-agent 策略，核心优化点在 memory management：
- 每个 agent 只保留自己工作内容相关的 memory/context
- 避免超长 context 问题
"""
import os
from datetime import datetime

from config import DATA_DIR
from coordinator import Coordinator
from logging_config import setup_logging


def generate_task_id() -> str:
    """生成任务 ID（基于时间戳）"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    setup_logging("INFO")
    import logging
    logger = logging.getLogger("auto_slr")

    # 确保数据目录存在
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("  Automatic Systematic Literature Review")
    print("  基于 Multi-Agent 的自动系统文献综述系统")
    print("=" * 60)
    print()
    print("系统包含以下 Agent：")
    print("  1. Coordinator  - 流程控制")
    print("  2. Assistant     - 信息抽取与记忆管理")
    print("  3. Executor      - 论文搜索与筛选")
    print("  4. Contactor     - 用户交互")
    print()

    user_query = input("请输入你的文献综述需求: ").strip()
    if not user_query:
        print("输入不能为空！")
        return

    task_id = generate_task_id()
    print(f"\n任务 ID: {task_id}")
    print(f"数据将保存在: {DATA_DIR}")

    # 创建 coordinator 并运行
    coordinator = Coordinator(task_id)
    try:
        coordinator.run(user_query)
    except KeyboardInterrupt:
        print("\n\n任务已中断。已保存的数据不会丢失。")
    except Exception as e:
        print(f"\n\n任务出错: {e}")
        raise

    print("\n任务完成！")


if __name__ == "__main__":
    main()
