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


def _clean_json_text(text: str) -> str:
    """清理 LLM 返回的 JSON 文本中的常见问题"""
    import re as _re
    # 移除 // 行注释（不在字符串内的）
    # 简单处理：移除行尾 // 注释
    text = _re.sub(r'(?<!:)//.*?$', '', text, flags=_re.MULTILINE)
    # 移除尾部逗号（}, ] 前的逗号）—— 这是 LLM 最常见的问题
    text = _re.sub(r',\s*([}\]])', r'\1', text)
    # 转义字符串值内的裸换行符（LLM 常在长字符串中插入未转义换行）
    # 只处理双引号内的裸换行，不处理已转义的 \n
    result = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == '\\' and in_string:
            # 转义序列，原样保留（包括 \n \t 等）
            result.append(c)
            if i + 1 < len(text):
                i += 1
                result.append(text[i])
            i += 1
            continue
        if c == '"':
            in_string = not in_string
            result.append(c)
        elif in_string and c == '\n':
            result.append('\\n')
        elif in_string and c == '\r':
            # 跳过 \r，只保留转义后的 \n
            pass
        elif in_string and c == '\t':
            result.append('\\t')
        else:
            result.append(c)
        i += 1
    return ''.join(result)


def _try_parse_markdown_table(text: str) -> list[dict] | None:
    """从 markdown table 中提取结构化数据。

    当 LLM 返回 markdown 表格而非 JSON 时使用。
    例如：
    | Drug | Effects | Study Type |
    |------|---------|------------|
    | Amiodarone | Sudden Death | RCT |
    → [{"Drug": "Amiodarone", "Effects": "Sudden Death", "Study Type": "RCT"}]
    """
    # 匹配 markdown table 行
    lines = text.strip().split('\n')
    header = None
    rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith('|'):
            continue
        cells = [c.strip() for c in stripped.split('|')[1:-1]]  # 去掉首尾空元素
        if not cells:
            continue
        # 跳过分隔行 (|---|---|)
        if all(set(c.strip()) <= {'-', ':', ' '} for c in cells):
            continue
        if header is None:
            header = cells
        else:
            rows.append(cells)

    if header is None or not rows:
        return None

    result = []
    for row in rows:
        item = {}
        for i, h in enumerate(header):
            item[h] = row[i] if i < len(row) else ""
        result.append(item)

    logger.info(f"从 markdown table 中提取了 {len(result)} 行数据")
    return result


def _try_parse_natural_language_screening(text: str) -> list | None:
    """从 LLM 返回的自然语言文本中尝试提取筛选结果。

    当 LLM 没有返回 JSON 格式，而是用自然语言描述每个论文的筛选判断时，
    尝试解析出结构化结果，如：
      "1. 论文1：排除，因为..." → {"index": 1, "decision": "exclude", "reason": "..."}
      "论文 2：纳入" → {"index": 2, "decision": "include", "reason": ""}
    """
    results = []
    # 按行或按编号条目分割
    # 匹配模式: 数字. 或 数字) 开头的条目
    pattern = re.compile(
        r'(?:^|\n)\s*(\d+)\s*[.、)：]\s*(.*?)(?=(?:\n\s*\d+\s*[.、)：]\s*)|$)',
        re.DOTALL,
    )
    matches = pattern.findall(text)
    if not matches:
        return None

    include_keywords = {"纳入", "符合", "include", "accept", "relevant", "yes", "入选"}
    exclude_keywords = {"排除", "不符合", "exclude", "reject", "irrelevant", "no", "剔除", "不纳入"}

    for num_str, content in matches:
        idx = int(num_str)
        content_lower = content.lower().strip()
        if not content_lower:
            continue

        decision = "not_sure"
        # 统计关键词命中数，取多的
        include_hits = sum(1 for kw in include_keywords if kw in content_lower)
        exclude_hits = sum(1 for kw in exclude_keywords if kw in content_lower)
        if include_hits > exclude_hits:
            decision = "include"
        elif exclude_hits > include_hits:
            decision = "exclude"

        # 提取原因：去掉判断关键词后的剩余文本，或"因为/由于"后的文本
        reason = ""
        reason_match = re.search(r'(?:因为|由于|原因[是为]|:|：)\s*(.*)', content, re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip().rstrip('。；;')
        elif decision != "not_sure":
            # 没有明确的原因标记，把去掉编号和判断关键词后的内容作为原因
            reason = re.sub(r'(?:初步判断|判断|决定)[：:]\s*(纳入|排除|include|exclude)\s*[，,]?\s*', '', content, flags=re.IGNORECASE).strip().rstrip('。；;')

        results.append({
            "index": idx,
            "decision": decision,
            "reason": reason,
        })

    if not results:
        return None
    logger.info(f"从自然语言中提取了 {len(results)} 条筛选结果")
    return results


def _try_extract_json_by_bracket_matching(text: str) -> dict | list | None:
    """通过括号匹配找到第一个完整 JSON 对象/数组。

    处理文本中有多个 {...} 或 [...] 嵌套、且简单 find/rfind
    无法正确定位起止位置的情况。
    """
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        i = 0
        while i < len(text):
            start = text.find(start_char, i)
            if start == -1:
                break
            depth = 0
            in_string = False
            escape_next = False
            for j in range(start, len(text)):
                c = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if c == '\\':
                    escape_next = True
                    continue
                if c == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == start_char:
                    depth += 1
                elif c == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:j + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            pass
                        try:
                            return json.loads(_clean_json_text(candidate))
                        except json.JSONDecodeError:
                            pass
                        try:
                            return json.loads(_clean_json_text(candidate), strict=False)
                        except json.JSONDecodeError:
                            pass
                        # 这个起始位置不行，尝试下一个
                        break
            i = start + 1
    return None


def extract_json(text: str) -> dict | list | None:
    """从 LLM 返回文本中提取 JSON。

    支持以下格式：
    - 纯 JSON
    - ```json ... ``` 代码块包裹
    - ``` ... ``` 代码块包裹
    - JSON 前后有多余文字
    """
    text = text.strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_clean_json_text(text))
    except json.JSONDecodeError:
        pass
    # strict=False 允许字符串内的控制字符
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_clean_json_text(text), strict=False)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 或 ``` ... ``` 代码块
    # 使用贪婪匹配，避免 JSON 内容中的 ``` 导致提前截断
    code_block_pattern = r"```(?:[a-zA-Z]*)?\s*\n?(.*)\n?\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        block_content = match.group(1).strip()
        try:
            return json.loads(block_content)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_clean_json_text(block_content))
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(block_content, strict=False)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_clean_json_text(block_content), strict=False)
        except json.JSONDecodeError:
            pass

    # 尝试找最外层的 { ... } 或 [ ... ]（在去掉代码块标记后的文本中）
    # 先去掉 ``` 标记以免干扰括号匹配
    clean_text = re.sub(r"```[a-zA-Z]*\s*", "", text).strip()
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = clean_text.find(start_char)
        if start != -1:
            end = clean_text.rfind(end_char)
            if end > start:
                candidate = clean_text[start:end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
                try:
                    return json.loads(_clean_json_text(candidate))
                except json.JSONDecodeError:
                    pass
                try:
                    return json.loads(candidate, strict=False)
                except json.JSONDecodeError:
                    pass
                try:
                    return json.loads(_clean_json_text(candidate), strict=False)
                except json.JSONDecodeError:
                    pass

    # 最后的后备：尝试逐字符找到第一个合法 JSON 的起止位置
    result = _try_extract_json_by_bracket_matching(clean_text)
    if result is not None:
        return result

    # 后备：尝试从 markdown table 中提取结构化数据
    table_result = _try_parse_markdown_table(text)
    if table_result is not None:
        return table_result

    # 后备：尝试从自然语言文本中提取结构化筛选结果
    # 当 LLM 未返回 JSON 而是返回了自然语言列表时使用
    nl_result = _try_parse_natural_language_screening(text)
    if nl_result is not None:
        return nl_result

    logger.warning(f"JSON 提取失败，原始文本: {text}")
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


def chat_with_system_for_json(
    system_prompt: str,
    user_message: str = "请执行上述请求。",
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> dict | list | None:
    """带 JSON 提取和重试的系统 LLM 请求。

    第一次请求如果 extract_json 失败，会用更强调格式的提示重试。
    返回解析后的 dict/list，或 None（所有重试均失败）。
    """
    result_str = chat_with_system(system_prompt, user_message, model, temperature, max_tokens)
    result = extract_json(result_str)
    if result is not None:
        return result

    # 重试：用更强约束的提示
    for attempt in range(1, max_retries + 1):
        logger.warning(f"JSON 解析失败，第 {attempt}/{max_retries} 次重试")
        retry_user = (
            f"你上次的回复不是合法的 JSON 格式，请重新输出。\n\n"
            f"【严格要求】只输出 JSON，不要输出任何其他文字、解释或 markdown 代码块标记。\n\n"
            f"原始请求：\n{user_message}"
        )
        result_str = chat_with_system(system_prompt, retry_user, model, temperature, max_tokens)
        result = extract_json(result_str)
        if result is not None:
            logger.info(f"第 {attempt} 次重试成功提取 JSON")
            return result

    logger.warning(f"所有重试均失败，返回 None")
    return None


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
