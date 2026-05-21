"""Memory Store - 基于文件的记忆管理模块

负责管理所有持久化的记忆文件：
- user_info_{user_id}.txt: 用户身份、学术背景、偏好（每个用户独立文件）
- research_task_{task_id}.txt: 研究任务信息
- paper_list_{task_id}.json: 全量论文列表
- satisfied_papers_{task_id}.json: 达标论文列表
- scoring_criteria_{task_id}.txt: 打分标准
- scored_papers_{task_id}.json: 打分排序后的论文列表
- timeline_{task_id}.log: 时间线日志
"""
import json
import os
from datetime import datetime

from config import DATA_DIR


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _path(filename: str) -> str:
    _ensure_dir(os.path.join(DATA_DIR, filename))
    return os.path.join(DATA_DIR, filename)


# ─── 通用读写 ────────────────────────────────────────────

def read_text(filename: str) -> str:
    """读取文本文件，不存在则返回空字符串"""
    filepath = _path(filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def write_text(filename: str, content: str) -> None:
    """覆写文本文件"""
    filepath = _path(filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def read_json(filename: str) -> dict | list:
    """读取 JSON 文件，不存在则返回空列表"""
    filepath = _path(filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def write_json(filename: str, data: dict | list) -> None:
    """覆写 JSON 文件"""
    filepath = _path(filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── 时间线日志 ──────────────────────────────────────────

def append_timeline(task_id: str, role: str, content: str) -> None:
    """向时间线日志追加一条记录"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [{role}] {content}\n"
    filepath = _path(f"timeline_{task_id}.log")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(entry)


# ─── 快捷访问函数 ────────────────────────────────────────

def read_user_info(user_id: str = "") -> str:
    filename = f"user_info_{user_id}.txt" if user_id else "user_info.txt"
    return read_text(filename)


def write_user_info(content: str, user_id: str = "") -> None:
    filename = f"user_info_{user_id}.txt" if user_id else "user_info.txt"
    write_text(filename, content)


def read_research_task(task_id: str) -> str:
    return read_text(f"research_task_{task_id}.txt")


def write_research_task(task_id: str, content: str) -> None:
    write_text(f"research_task_{task_id}.txt", content)


def read_paper_list(task_id: str) -> list:
    return read_json(f"paper_list_{task_id}.json")


def write_paper_list(task_id: str, data: list) -> None:
    write_json(f"paper_list_{task_id}.json", data)


def read_satisfied_papers(task_id: str) -> list:
    return read_json(f"satisfied_papers_{task_id}.json")


def write_satisfied_papers(task_id: str, data: list) -> None:
    write_json(f"satisfied_papers_{task_id}.json", data)


def read_scoring_criteria(task_id: str) -> str:
    return read_text(f"scoring_criteria_{task_id}.txt")


def write_scoring_criteria(task_id: str, content: str) -> None:
    write_text(f"scoring_criteria_{task_id}.txt", content)


def read_scored_papers(task_id: str) -> list:
    return read_json(f"scored_papers_{task_id}.json")


def write_scored_papers(task_id: str, data: list) -> None:
    write_json(f"scored_papers_{task_id}.json", data)


# ─── 内存+文件混合存储 ─────────────────────────────────────

class InMemoryStore:
    """基于内存+实时文件写入的存储，替换 memory_store 的文件 I/O

    每次 write 调用同时更新内存和写入文件，与正常流程 memory_store 行为一致。
    适用于测试场景，每个 review 使用独立实例，天然线程安全。
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.user_info: dict[str, str] = {}
        self.research_task: dict[str, str] = {}
        self.paper_list: dict[str, list] = {}
        self.satisfied_papers: dict[str, list] = {}
        self.scoring_criteria: dict[str, str] = {}
        self.scored_papers: dict[str, list] = {}
        self.timeline: dict[str, list] = {}

    # ─── 辅助方法 ───

    def _write_text(self, filename: str, content: str) -> None:
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def _write_json(self, filename: str, data) -> None:
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _append_text(self, filename: str, content: str) -> None:
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(content + "\n")

    # ─── 读写方法 ───

    def read_user_info(self, user_id: str = "") -> str:
        return self.user_info.get(user_id, "")

    def write_user_info(self, content: str, user_id: str = "") -> None:
        self.user_info[user_id] = content
        filename = f"user_info_{user_id}.txt" if user_id else "user_info.txt"
        self._write_text(filename, content)

    def read_research_task(self, task_id: str) -> str:
        return self.research_task.get(task_id, "")

    def write_research_task(self, task_id: str, content: str) -> None:
        self.research_task[task_id] = content
        self._write_text(f"research_task_{task_id}.txt", content)

    def read_paper_list(self, task_id: str) -> list:
        return self.paper_list.get(task_id, [])

    def write_paper_list(self, task_id: str, data: list) -> None:
        self.paper_list[task_id] = data
        self._write_json(f"paper_list_{task_id}.json", data)

    def read_satisfied_papers(self, task_id: str) -> list:
        return self.satisfied_papers.get(task_id, [])

    def write_satisfied_papers(self, task_id: str, data: list) -> None:
        self.satisfied_papers[task_id] = data
        self._write_json(f"satisfied_papers_{task_id}.json", data)

    def read_scoring_criteria(self, task_id: str) -> str:
        return self.scoring_criteria.get(task_id, "")

    def write_scoring_criteria(self, task_id: str, content: str) -> None:
        self.scoring_criteria[task_id] = content
        self._write_text(f"scoring_criteria_{task_id}.txt", content)

    def read_scored_papers(self, task_id: str) -> list:
        return self.scored_papers.get(task_id, [])

    def write_scored_papers(self, task_id: str, data: list) -> None:
        self.scored_papers[task_id] = data
        self._write_json(f"scored_papers_{task_id}.json", data)

    def append_timeline(self, task_id: str, role: str, content: str) -> None:
        if task_id not in self.timeline:
            self.timeline[task_id] = []
        entry = f"[{role}] {content}"
        self.timeline[task_id].append(entry)
        self._append_text(f"timeline_{task_id}.log", entry)
