"""Contactor Agent - 用户交互

职责：
1. 与用户直接对话
2. 只记住前三轮左右的 context 和 coordinator/assistant/executor 提供的信息
3. 询问用户额外信息
4. 让用户确认样例论文是否满意
"""
import logging

import memory_store as ms
from llm_client import chat

logger = logging.getLogger("auto_slr")

ASK_INFO_SYSTEM_PROMPT = """你是一个友好的学术助手，帮助用户进行系统性文献综述。
你需要引导用户提供足够的信息来进行论文搜索。

当前缺失的信息：{missing_info}

已有信息：
{existing_info}

请以友好且专业的方式询问用户缺失的信息。不要一次问太多问题——最多 2-3 个关键问题。
直接输出你想对用户说的话，不要输出其他内容。"""

PRESENT_SAMPLE_SYSTEM_PROMPT = """你是一个学术助手，正在向用户展示筛选后的样例论文，并请求用户确认。

以下是筛选出的样例论文：
{sample_papers}

请向用户展示这些论文，并询问：
1. 哪些论文符合他们的期望？
2. 哪些不符合？如果不符合，请说明原因。

请以友好且清晰的方式展示，包括论文标题、年份、作者和筛选理由。"""

GENERAL_CHAT_SYSTEM_PROMPT = """你是一个学术文献综述助手。你正在与用户交流其文献综述需求和反馈。
请保持友好、专业、简洁。"""


class Contactor:
    """Contactor Agent：与用户直接交流"""

    def __init__(self, context_rounds: int = 3):
        self.context_rounds = context_rounds
        self._history: list[dict] = []

    def _add_to_history(self, role: str, content: str) -> None:
        """添加消息到历史，并保持只保留最近 context_rounds 轮"""
        self._history.append({"role": role, "content": content})
        # 每轮包含 user + assistant，所以保留 2 * context_rounds 条
        max_messages = self._context_max()
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def _context_max(self) -> int:
        return self.context_rounds * 2 + 1  # +1 for system message room

    def ask_missing_info(self, missing_info: str, task_id: str, user_id: str = "") -> str:
        """询问用户缺少的信息"""
        user_info = ms.read_user_info(user_id)
        research_task = ms.read_research_task(task_id)

        logger.info(f"[Contactor] ask_missing_info 输入: missing_info={missing_info}")

        message = chat(
            [
                {
                    "role": "system",
                    "content": ASK_INFO_SYSTEM_PROMPT.format(
                        missing_info=missing_info,
                        existing_info=f"用户信息: {user_info}\n研究任务: {research_task}",
                    ),
                },
                {"role": "user", "content": "请向用户询问缺失的信息。"},
            ]
        )
        self._add_to_history("assistant", message)
        ms.append_timeline(task_id, "contactor", f"询问用户缺少信息: {missing_info}")
        logger.info(f"[Contactor] ask_missing_info 输出: {message[:300]}{'...' if len(message) > 300 else ''}")
        return message

    def present_sample_papers(self, included_papers: list[dict], task_id: str) -> str:
        """向用户展示样例论文"""
        import json
        logger.info(f"[Contactor] present_sample_papers 输入: {len(included_papers)} 篇论文")

        papers_text = "\n\n".join(
            f"[论文 {i+1}]\n"
            f"标题: {p.get('title', 'N/A')}\n"
            f"年份: {p.get('year', 'N/A')}\n"
            f"作者: {p.get('authors', 'N/A')}\n"
            f"筛选理由: {p.get('include_reason', 'N/A')}"
            for i, p in enumerate(included_papers)
        )

        message = chat(
            [
                {
                    "role": "system",
                    "content": PRESENT_SAMPLE_SYSTEM_PROMPT.format(sample_papers=papers_text),
                },
                {"role": "user", "content": "请向用户展示样例论文，并请求反馈。"},
            ]
        )
        self._add_to_history("assistant", message)
        ms.append_timeline(task_id, "contactor", f"展示 {len(included_papers)} 篇样例论文")
        logger.info(f"[Contactor] present_sample_papers 输出: {message[:300]}{'...' if len(message) > 300 else ''}")
        return message

    def chat_with_user(self, user_message: str, task_id: str, extra_context: str = "") -> str:
        """与用户自由对话"""
        logger.info(f"[Contactor] chat_with_user 输入: user_message={user_message[:200]}")
        self._add_to_history("user", user_message)

        system_content = GENERAL_CHAT_SYSTEM_PROMPT
        if extra_context:
            system_content += f"\n\n额外上下文:\n{extra_context}"

        messages = [{"role": "system", "content": system_content}] + self._history
        response = chat(messages)
        self._add_to_history("assistant", response)
        ms.append_timeline(task_id, "contactor", f"与用户对话: {user_message[:50]}...")
        logger.info(f"[Contactor] chat_with_user 输出: {response[:300]}{'...' if len(response) > 300 else ''}")
        return response

    def receive_user_input(self, user_input: str) -> None:
        """接收用户输入（供外部调用，不自动回复）"""
        self._add_to_history("user", user_input)

    def get_recent_context(self) -> list[dict]:
        """获取最近的对话上下文"""
        return self._history.copy()

    def reset_history(self) -> None:
        """重置对话历史"""
        self._history = []
