"""LLM Client - 封装 OpenAI API 调用

提供两套客户端：
- 系统 LLM (GLM): 用于 assistant/executor/contactor 等 agent
- 模拟 LLM (DeepSeek): 用于模拟硕士学生对话
"""
import json
import logging
import re

from openai import OpenAI

from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
    SIMULATOR_API_KEY, SIMULATOR_BASE_URL, SIMULATOR_MODEL,
)

logger = logging.getLogger("auto_slr")

# 系统用客户端 (GLM)
_system_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# 模拟学生用客户端 (DeepSeek)
_simulator_client = OpenAI(api_key=SIMULATOR_API_KEY, base_url=SIMULATOR_BASE_URL)


def extract_json(text: str) -> dict | list | None:
    """从 LLM 返回文本中提取 JSON。

    支持以下格式：
    - 纯 JSON
    - ```json ... ``` 代码块包裹
    - ``` ... ``` 代码块包裹
    - JSON 前后有多余文字
    """
    # 尝试直接解析
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 或 ``` ... ``` 代码块
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试找最外层的 { ... } 或 [ ... ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start != -1:
            # 从末尾找最后一个匹配的结束符
            end = text.rfind(end_char)
            if end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass

    logger.warning(f"JSON 提取失败，原始文本: {text[:200]}...")
    return None


# ─── 系统 LLM (GLM) ──────────────────────────────────────

def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """系统 LLM 聊天请求"""
    model_name = model or OPENAI_MODEL
    logger.debug(f"[System LLM] 请求, model={model_name}, messages={len(messages)}条")
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        logger.debug(f"[System LLM] 输入[{i}] role={role}: {content[:500]}{'...' if len(content) > 500 else ''}")
    response = _system_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    logger.debug(f"[System LLM] 响应: {content}")
    return content


def chat_with_system(
    system_prompt: str,
    user_message: str = "请执行上述请求。",
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """系统 LLM 快捷方法：一条系统消息 + 一条用户消息"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)


# ─── 模拟学生 LLM (DeepSeek) ────────────────────────────

def simulator_chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """模拟学生 LLM 聊天请求"""
    model_name = model or SIMULATOR_MODEL
    logger.debug(f"[Simulator LLM] 请求, model={model_name}, messages={len(messages)}条")
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        logger.debug(f"[Simulator LLM] 输入[{i}] role={role}: {content[:500]}{'...' if len(content) > 500 else ''}")
    response = _simulator_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    logger.debug(f"[Simulator LLM] 响应: {content}")
    return content
