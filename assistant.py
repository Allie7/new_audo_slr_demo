"""Assistant Agent - 信息抽取与记录

职责：
1. 从用户 query 和其他 agent 输出中提取有效信息
2. 将信息归类记录到对应文件（user_info_{user_id}.txt, research_task_{task_id}.txt）
3. 在 executor 执行任务前，提供相关背景信息
4. 每次记录新信息前读取原记录，比对去重/修正矛盾后重写
5. 生成打分标准
"""
import json
import logging

from memory_store import InMemoryStore
from llm_client import chat_with_system, chat_with_system_for_json, extract_json

logger = logging.getLogger("auto_slr")

EXTRACT_SYSTEM_PROMPT = """你是一个信息抽取助手。你的任务是从给定的对话内容中提取以下类别的信息：

1. **用户身份与背景**：如学生身份、学术领域、教育水平
2. **用户偏好**：如论文语言偏好、发表类型偏好
3. **研究方向**：具体的研究主题/方向
4. **论文筛选要求**：时间范围、关键词、排除标准等
5. **用户拒绝论文的原因**：对某篇论文不满意的原因
6. **其他有用信息**

请严格以 JSON 格式输出，键为类别名称，值为提取到的信息（字符串）。如果某个类别没有提取到信息，对应的值为空字符串。
示例：
{{
  "user_identity": "计算机专业硕士研究生",
  "user_preference": "",
  "research_direction": "LLM Agent 记忆管理",
  "filter_requirements": "近两年",
  "rejection_reasons": "",
  "other_info": ""
}}

【重要】只输出上面的 JSON 格式，不要输出任何其他文字、解释或 markdown 格式。不要用 ```json``` 包裹。直接输出花括号开头的 JSON 文本。"""

MERGE_USER_INFO_PROMPT = """你是一个信息整合助手。以下是[已有的用户信息记录]和[新提取的用户相关信息]。
请将两者合并，去除重复，并解决矛盾（以最新信息为准）。输出完整的整合版本。
只输出整合后的文本，不要输出其他内容。

已有记录：
{existing}

新提取的信息：
{new_info}"""

MERGE_RESEARCH_TASK_PROMPT = """你是一个信息整合助手。以下是[已有的研究任务记录]和[新提取的研究任务相关信息]。
请将两者合并，去除重复，并解决矛盾（以最新信息为准）。输出完整的整合版本。
只输出整合后的文本，不要输出其他内容。

已有记录：
{existing}

新提取的信息：
{new_info}"""

PROVIDE_CONTEXT_PROMPT = """你是一个信息提炼助手。你的任务是从原始记录中提炼出对论文搜索/筛选最有价值的关键信息，去除冗余、矛盾和无关细节。

当前用户信息：
{user_info}

当前研究任务：
{research_task}

当前达标论文列表：
{satisfied_papers}

执行者即将执行的任务：{task_description}

注意：当前仅基于论文摘要进行初筛，因此只需保留能在摘要层面判断的信息，去除过于细节、需要阅读全文才能判断的条件。

请按以下结构输出提炼后的信息（某项无内容则省略该行）：

【研究方向】用一句话概括核心研究主题
【关键词】3-8个核心关键词，用逗号分隔
【筛选要求】基于摘要可判断的硬性条件（如特定疾病、特定方法、特定人群），不要列出需要全文才能判断的条件
【排除标准】基于摘要可判断的排除方向
【已确认满意论文的共同特征】从达标论文摘要中总结的共性

提炼原则：
1. 矛盾信息以最新为准，不重复列出
2. 去除模糊、笼统的描述（如"高质量""前沿"），只保留可操作的筛选条件
3. 去除与论文搜索/筛选无关的个人信息（如用户身份、学校）
4. 去除过于细节的条件（如样本量下限、特定统计方法、效应量要求等需要全文才能判断的内容）
5. 时间范围、语言要求、发表类型等如无明确要求则不列出，由后续阶段处理
6. 只输出提炼结果，不要输出解释或原始信息"""

GENERATE_CRITERIA_PROMPT = """你是一个学术文献综述打分标准设计助手。请根据以下信息生成一套系统的、多维度的论文打分标准。

用户信息：
{user_info}

研究任务要求：
{research_task}

用户确认满意的论文特征（从达标论文中总结）：
{satisfied_papers_summary}

请生成 4-6 个评分维度，严格遵循以下规则：
1. 每个维度按 1-5 分制打分（1=很差，2=较差，3=一般，4=较好，5=很好）
2. 所有维度权重相同
3. 总分 = 所有维度分数之和

请以 JSON 格式输出：
{{
  "dimensions": [
    {{
      "name": "维度名称",
      "description": "维度描述",
      "scale": "1-5",
      "criteria": {{
        "1": "1分的评分标准描述",
        "2": "2分的评分标准描述",
        "3": "3分的评分标准描述",
        "4": "4分的评分标准描述",
        "5": "5分的评分标准描述"
      }}
    }}
  ],
  "total_max": 最高总分
}}

只输出 JSON，不要输出其他内容。"""


class Assistant:
    """Assistant Agent：信息抽取与记忆管理
    
    Args:
        store: InMemoryStore 实例，用于读写记忆数据。如果为 None，则使用全局 memory_store 模块（兼容旧代码）。
    """

    def __init__(self, store: InMemoryStore | None = None):
        self.store = store

    def _ms(self):
        """获取存储后端：优先使用传入的 store，否则使用全局 memory_store 模块"""
        if self.store is not None:
            return self.store
        import memory_store as ms
        return ms

    def extract_and_record(self, content: str, task_id: str, user_id: str = "") -> dict:
        """从内容中抽取信息并记录到对应文件"""
        logger.info(f"[Assistant] extract_and_record 输入: {content[:300]}{'...' if len(content) > 300 else ''}")

        # 1. 抽取信息
        extracted = chat_with_system_for_json(EXTRACT_SYSTEM_PROMPT, content)
        if extracted is None:
            logger.warning(f"信息抽取 JSON 解析失败（已重试3次）")
            # 对原始文本做提炼，而非整段塞入 other_info
            extracted_str = chat_with_system(EXTRACT_SYSTEM_PROMPT, content)
            summary = chat_with_system(
                "你是一个信息提炼助手。请从以下文本中提取关键信息，用简洁的中文要点列出，不要使用表格或 JSON。"
                "只输出提炼后的要点，每条一行，不超过5条。如果文本无有价值信息，输出'无'。",
                extracted_str,
            )
            extracted = {"other_info": summary.strip()}
        if not isinstance(extracted, dict):
            extracted = {"other_info": str(extracted)}

        logger.info(f"[Assistant] extract_and_record 输出: {json.dumps(extracted, ensure_ascii=False)}")

        # 2. 分类记录
        user_related_keys = ["user_identity", "user_preference"]
        task_related_keys = [
            "research_direction",
            "filter_requirements",
            "rejection_reasons",
            "other_info",
        ]

        # 合并用户信息
        user_new = "\n".join(
            f"{k}: {extracted.get(k, '')}" for k in user_related_keys if extracted.get(k)
        )
        if user_new:
            existing = self._ms().read_user_info(user_id)
            merged = chat_with_system(
                MERGE_USER_INFO_PROMPT.format(existing=existing or "(none)", new_info=user_new)
            )
            self._ms().write_user_info(merged, user_id)

        # 合并研究任务信息
        task_new = "\n".join(
            f"{k}: {extracted.get(k, '')}" for k in task_related_keys if extracted.get(k)
        )
        if task_new:
            existing = self._ms().read_research_task(task_id)
            merged = chat_with_system(
                MERGE_RESEARCH_TASK_PROMPT.format(
                    existing=existing or "(none)", new_info=task_new
                )
            )
            self._ms().write_research_task(task_id, merged)

        # 3. 记录时间线
        self._ms().append_timeline(task_id, "assistant", f"抽取并记录信息: {json.dumps(extracted, ensure_ascii=False)}")

        return extracted

    def provide_context(self, task_description: str, task_id: str, user_id: str = "") -> str:
        """为 executor 提供与当前任务相关的背景信息"""
        user_info = self._ms().read_user_info(user_id)
        research_task = self._ms().read_research_task(task_id)
        satisfied_papers = self._ms().read_satisfied_papers(task_id)

        logger.info(f"[Assistant] provide_context 输入: task_description={task_description}")

        context = chat_with_system(
            PROVIDE_CONTEXT_PROMPT.format(
                user_info=user_info or "(none)",
                research_task=research_task or "(none)",
                satisfied_papers=json.dumps(satisfied_papers, ensure_ascii=False) if satisfied_papers else "(none)",
                task_description=task_description,
            )
        )
        logger.info(f"[Assistant] provide_context 输出: {context[:300]}{'...' if len(context) > 300 else ''}")
        self._ms().append_timeline(task_id, "assistant", f"为 executor 提供背景信息，任务: {task_description}")
        return context

    def generate_scoring_criteria(self, task_id: str, user_id: str = "") -> str:
        """生成论文打分标准，固定为 1-5 分 scale，生成后锁定不再改变"""
        # 如果已有标准，直接返回，不再重新生成
        existing = self._ms().read_scoring_criteria(task_id)
        if existing and existing.strip():
            logger.info(f"打分标准已存在，使用已有标准")
            return existing

        user_info = self._ms().read_user_info(user_id)
        research_task = self._ms().read_research_task(task_id)
        satisfied_papers = self._ms().read_satisfied_papers(task_id)

        # 总结达标论文特征
        satisfied_summary = ""
        if satisfied_papers:
            titles = [p.get("title", "unknown") for p in satisfied_papers]
            reasons = [p.get("include_reason", "") for p in satisfied_papers]
            satisfied_summary = "\n".join(
                f"- {t}: {r}" for t, r in zip(titles, reasons) if r
            )

        criteria_str = chat_with_system(
            GENERATE_CRITERIA_PROMPT.format(
                user_info=user_info or "(none)",
                research_task=research_task or "(none)",
                satisfied_papers_summary=satisfied_summary or "(尚无达标论文)",
            )
        )

        # 解析并验证标准格式
        criteria = extract_json(criteria_str)
        if criteria is None or not isinstance(criteria, dict) or "dimensions" not in criteria:
            logger.warning(f"打分标准 JSON 解析失败，原始返回:\n{criteria_str}")
            # 使用默认标准
            criteria = {
                "dimensions": [
                    {"name": "相关性", "description": "论文与研究主题的相关程度", "scale": "1-5",
                     "criteria": {"1": "完全无关", "2": "略微相关", "3": "部分相关", "4": "高度相关", "5": "完全匹配"}},
                    {"name": "时效性", "description": "论文发表日期的新近程度", "scale": "1-5",
                     "criteria": {"1": "超过10年", "2": "5-10年", "3": "3-5年", "4": "1-3年", "5": "1年以内"}},
                    {"name": "方法论质量", "description": "研究方法的严谨性和创新性", "scale": "1-5",
                     "criteria": {"1": "方法存在严重缺陷", "2": "方法低于平均水平", "3": "方法基本合格", "4": "方法良好", "5": "方法优秀"}},
                    {"name": "影响力", "description": "论文学术影响力（引用量等）", "scale": "1-5",
                     "criteria": {"1": "无引用", "2": "引用较少", "3": "引用中等", "4": "引用较多", "5": "高引用"}},
                ],
                "total_max": 20,
            }

        # 确保每个维度都是 1-5 scale
        for dim in criteria.get("dimensions", []):
            dim["scale"] = "1-5"
        criteria["total_max"] = len(criteria.get("dimensions", [])) * 5

        # 固定保存为 JSON 文本
        criteria_text = json.dumps(criteria, ensure_ascii=False, indent=2)
        self._ms().write_scoring_criteria(task_id, criteria_text)
        self._ms().append_timeline(task_id, "assistant", f"打分标准已生成并锁定: {len(criteria.get('dimensions', []))} 个维度, 总分 {criteria['total_max']}")
        logger.info(f"打分标准已生成并锁定: {criteria.get('dimensions', [])}")
        return criteria_text
