"""CSMeD-FT 数据集测试脚本

使用 CSMeD-FT (Cochrane Systematic Review - Full Text) 数据集测试
assistant → coordinator → executor 工作流。

数据源（已预处理合并）：
- CSMeD-FT-all_reviews_metadata.json : 213 个 review 的元数据（title, abstract,
  review_type, criteria, search_strategy, criteria_text）
- CSMeD-FT-papers.json : 3333 篇去重论文（document_id 为 key，含 title/abstract/main_text 等）
- CSMeD-FT-gold_standard.json : 3333 条 review_id-document_id 决策记录（included/excluded）

工作流：
1. 遍历 all_reviews_metadata 中的每条 review
2. 以 review 的 title 作为初始 user query
3. 将 criteria, search_strategy, criteria_text, abstract, review_type 等
   作为后续轮 user 输入，assistant 持续抽取和丰富上下文
4. 基于论文标题进行快速初筛
5. executor 迭代样例筛选：找到3篇include时暂停，模拟用户反馈后继续
6. assistant 生成打分标准 → executor 对 include+not_sure 打分排序
7. 将系统判定的 include 论文与 gold_standard 中该 review 的 included 论文对比
8. 输出每条 review 的 precision/recall/F1 以及整体汇总报告
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

# ── 路径设置 ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 测试数据输出目录
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "data_csmed_test")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# CSMeD-FT 数据集路径
CSMED_DATA_DIR = r"D:\School\SAT404\pilot_study_code\systematic-review-datasets\data\CSMeD\CSMeD-FT\CSMeD-FT"

# ── 日志配置 ──────────────────────────────────────────────
log_filename = f"test_csmed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(TEST_DATA_DIR, log_filename), encoding="utf-8"),
    ],
)
logger = logging.getLogger("test_csmed")

# ── 导入项目模块 ──────────────────────────────────────────
from llm_client import chat_with_system, extract_json
from assistant import Assistant
from executor import SCREEN_SYSTEM_PROMPT, SCREEN_BATCH_SIZE, SCORE_SYSTEM_PROMPT, SCORE_BATCH_SIZE, TITLE_SCREEN_SYSTEM_PROMPT, TITLE_SCREEN_BATCH_SIZE
import memory_store as ms


# ═══════════════════════════════════════════════════════════
# 数据集加载（使用合并后的文件）
# ═══════════════════════════════════════════════════════════

def load_csmed_data():
    """加载合并后的 CSMeD-FT 数据集

    Returns:
        (reviews_metadata, papers_dict, gold_standard)
        - reviews_metadata: dict[review_id -> dict]，每个 review 的元数据
        - papers_dict: dict[document_id -> dict]，去重后的论文信息
        - gold_standard: dict[review_id -> dict[document_id -> "included"/"excluded"]]
    """
    # 1. 加载 reviews metadata
    reviews_path = os.path.join(CSMED_DATA_DIR, "CSMeD-FT-all_reviews_metadata.json")
    with open(reviews_path, encoding="utf-8") as f:
        reviews_metadata = json.load(f)
    logger.info(f"Reviews metadata 加载完成: {len(reviews_metadata)} 个 reviews")

    # 2. 加载去重论文集
    papers_path = os.path.join(CSMED_DATA_DIR, "CSMeD-FT-papers.json")
    with open(papers_path, encoding="utf-8") as f:
        papers_list = json.load(f)
    papers_dict = {p["document_id"]: p for p in papers_list}
    logger.info(f"论文集加载完成: {len(papers_dict)} 篇去重论文")

    # 3. 加载 gold standard
    gold_path = os.path.join(CSMED_DATA_DIR, "CSMeD-FT-gold_standard.json")
    with open(gold_path, encoding="utf-8") as f:
        gold_list = json.load(f)

    # 构建 review_id -> {document_id -> {"decision": ..., "reason_for_exclusion": ...}} 索引
    gold_standard: dict[str, dict[str, dict]] = {}
    for item in gold_list:
        rid = item["review_id"]
        did = item["document_id"]
        if rid not in gold_standard:
            gold_standard[rid] = {}
        gold_standard[rid][did] = {
            "decision": item.get("decision", ""),
            "reason_for_exclusion": item.get("reason_for_exclusion", "") or "",
        }

    logger.info(f"Gold standard 加载完成: {len(gold_standard)} 个 reviews, {len(gold_list)} 条决策")

    return reviews_metadata, papers_dict, gold_standard


def get_gold_included_ids(gold_standard: dict, review_id: str) -> set[str]:
    """获取某个 review 在 gold standard 中判定为 included 的 document_id 集合"""
    review_gold = gold_standard.get(review_id, {})
    return {did for did, info in review_gold.items() if (info if isinstance(info, dict) else {"decision": info}).get("decision") == "included"}


def get_gold_paper_ids(gold_standard: dict, review_id: str) -> set[str]:
    """获取某个 review 在 gold standard 中涉及的所有 document_id"""
    return set(gold_standard.get(review_id, {}).keys())


def build_paper_list_for_review(
    paper_ids: set[str],
    papers_dict: dict,
) -> list[dict]:
    """将 document_id 集合转为系统内部 paper_list 格式

    只包含在 papers_dict 中能找到的论文（跳过缺失的）
    """
    result = []
    for did in sorted(paper_ids):
        p = papers_dict.get(did)
        if p is None:
            logger.warning(f"论文 {did} 在 papers_dict 中不存在，跳过")
            continue
        result.append({
            "_id": did,
            "title": p.get("title", "") or "",
            "abstract": p.get("abstract", "") or "",
            "text": p.get("abstract", "") or "",
            "main_text": p.get("main_text", "") or "",
            "journal": p.get("journal", "") or "",
            "year": p.get("year", ""),
            "authors": p.get("authors", "") or "",
            "source": "csmed",
        })
    return result


# ═══════════════════════════════════════════════════════════
# 构建 user query 和后续输入
# ═══════════════════════════════════════════════════════════

def build_initial_query(review_meta: dict) -> str:
    """用 review 的 title 作为初始 user query"""
    return review_meta.get("title", "")


def build_supplementary_inputs(review_meta: dict) -> list[str]:
    """将 review_metadata 中的其他字段构建为后续轮 user 输入

    依次提供：
    1. review_type + abstract（研究类型和背景）
    2. criteria_text（纳入/排除标准）
    3. criteria（结构化纳入/排除标准）
    4. search_strategy（搜索策略）
    """
    inputs = []

    # 第1轮补充：研究类型 + 综述摘要
    review_type = review_meta.get("review_type", "")
    abstract = review_meta.get("abstract", "")
    if review_type or abstract:
        parts = []
        if review_type:
            parts.append(f"研究类型 (review_type): {review_type}")
        if abstract:
            parts.append(f"系统综述摘要:\n{abstract}")
        inputs.append("\n\n".join(parts))

    # 第2轮补充：纳入/排除标准文本
    criteria_text = review_meta.get("criteria_text", "")
    if criteria_text:
        inputs.append(f"纳入/排除标准 (criteria_text):\n{criteria_text}")

    # 第3轮补充：结构化纳入/排除标准
    criteria = review_meta.get("criteria", "")
    if criteria:
        if isinstance(criteria, dict):
            criteria_str = json.dumps(criteria, ensure_ascii=False, indent=2)
        else:
            criteria_str = str(criteria)
        inputs.append(f"结构化纳入/排除标准 (criteria):\n{criteria_str}")

    # 第4轮补充：搜索策略
    search_strategy = review_meta.get("search_strategy", "")
    if search_strategy:
        if isinstance(search_strategy, dict):
            ss_str = json.dumps(search_strategy, ensure_ascii=False, indent=2)
        else:
            ss_str = str(search_strategy)
        inputs.append(f"搜索策略 (search_strategy):\n{ss_str}")

    return inputs


# ═══════════════════════════════════════════════════════════
# 内存版 memory_store
# ═══════════════════════════════════════════════════════════

class InMemoryStore:
    """基于内存+实时文件写入的存储，替换 memory_store 的文件 I/O

    每次 write 调用同时更新内存和写入文件，与正常流程 memory_store 行为一致。
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


# ═══════════════════════════════════════════════════════════
# memory_store 热替换
# ═══════════════════════════════════════════════════════════

_original_ms_functions = {}


def _patch_memory_store(store: InMemoryStore) -> None:
    global _original_ms_functions
    if not _original_ms_functions:
        _original_ms_functions = {
            "read_user_info": ms.read_user_info,
            "write_user_info": ms.write_user_info,
            "read_research_task": ms.read_research_task,
            "write_research_task": ms.write_research_task,
            "read_paper_list": ms.read_paper_list,
            "write_paper_list": ms.write_paper_list,
            "read_satisfied_papers": ms.read_satisfied_papers,
            "write_satisfied_papers": ms.write_satisfied_papers,
            "read_scoring_criteria": ms.read_scoring_criteria,
            "write_scoring_criteria": ms.write_scoring_criteria,
            "read_scored_papers": ms.read_scored_papers,
            "write_scored_papers": ms.write_scored_papers,
            "append_timeline": ms.append_timeline,
        }
    ms.read_user_info = store.read_user_info
    ms.write_user_info = store.write_user_info
    ms.read_research_task = store.read_research_task
    ms.write_research_task = store.write_research_task
    ms.read_paper_list = store.read_paper_list
    ms.write_paper_list = store.write_paper_list
    ms.read_satisfied_papers = store.read_satisfied_papers
    ms.write_satisfied_papers = store.write_satisfied_papers
    ms.read_scoring_criteria = store.read_scoring_criteria
    ms.write_scoring_criteria = store.write_scoring_criteria
    ms.read_scored_papers = store.read_scored_papers
    ms.write_scored_papers = store.write_scored_papers
    ms.append_timeline = store.append_timeline


def _restore_memory_store() -> None:
    global _original_ms_functions
    if _original_ms_functions:
        for name, func in _original_ms_functions.items():
            setattr(ms, name, func)
        _original_ms_functions = {}


# ═══════════════════════════════════════════════════════════
# Metadata 初筛（与正常流程 executor.search_papers 中的元数据过滤一致）
# ═══════════════════════════════════════════════════════════

def metadata_prescreen(all_papers: list[dict], year_range: str | None = None) -> list[dict]:
    """基于论文元数据的质量过滤

    规则（与 executor.search_papers 一致）：
    - 排除无摘要的论文（无法进行后续筛选）
    - 排除无标题的论文
    - 排除无年份的论文（如果指定了年份范围）
    - 排除超出年份范围的论文

    注意：CSMeD 数据集论文均有完整 metadata，此步骤主要模拟正常流程中的 metadata 过滤。
    """
    filtered_papers = []
    filtered_out = 0

    for p in all_papers:
        # 排除无摘要的论文
        if not p.get("abstract") or not p["abstract"].strip():
            filtered_out += 1
            continue
        # 排除无标题的论文
        if not p.get("title") or not p["title"].strip():
            filtered_out += 1
            continue
        # 排除无年份的论文（如果用户指定了年份范围）
        if year_range and p.get("year") is None:
            filtered_out += 1
            continue
        # 排除超出年份范围的论文
        if year_range and p.get("year") is not None:
            try:
                parts_yr = year_range.split("-")
                start_yr = int(parts_yr[0])
                end_yr = int(parts_yr[1]) if len(parts_yr) > 1 else start_yr
                if not (start_yr <= p["year"] <= end_yr):
                    filtered_out += 1
                    continue
            except (ValueError, IndexError):
                pass
        filtered_papers.append(p)

    if filtered_out > 0:
        logger.info(f"[Metadata初筛] 排除 {filtered_out} 篇不符合要求的论文 (无摘要/无标题/年份不符)")
    logger.info(f"[Metadata初筛] 全量 {len(all_papers)} 篇 → 保留 {len(filtered_papers)} 篇, 排除 {filtered_out} 篇")
    return filtered_papers


# ═══════════════════════════════════════════════════════════
# 评估指标计算
# ═══════════════════════════════════════════════════════════

def compute_metrics(retrieved_ids: set[str], relevant_ids: set[str]) -> dict:
    """计算检索评估指标"""
    if not relevant_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_relevant": 0, "num_retrieved": len(retrieved_ids), "num_true_positives": 0}
    if not retrieved_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_relevant": len(relevant_ids), "num_retrieved": 0, "num_true_positives": 0}

    true_positives = retrieved_ids & relevant_ids
    precision = len(true_positives) / len(retrieved_ids)
    recall = len(true_positives) / len(relevant_ids)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "num_relevant": len(relevant_ids),
        "num_retrieved": len(retrieved_ids),
        "num_true_positives": len(true_positives),
    }


# ═══════════════════════════════════════════════════════════
# 核心：单个 review 的测试流程
# ═══════════════════════════════════════════════════════════

def run_single_review(
    review_id: str,
    review_meta: dict,
    paper_list: list[dict],
    gold_included_ids: set[str],
    store: InMemoryStore,
    skip_prescreen: bool = False,
    gold_standard: dict[str, dict[str, str]] | None = None,
) -> dict:
    """对单个 review 运行完整的测试流程

    流程（与 untitled.txt 设计一致，仅 user simulation 方式不同）：
    1. 以 review title 作为初始 user query → assistant 抽取信息
    2. 逐步提供 criteria/search_strategy/abstract 等作为后续 user 输入
    3. 基于论文 metadata 初筛（与正常流程一致）
    4. Title 筛选：仅根据标题快速排除明显不相关的论文（保守策略）
    5. 迭代样例筛选 + 用户反馈：
       - executor 根据 abstract 筛选，找到 3 篇 include 时暂停
       - 用 gold standard 模拟用户反馈，满意的记入 satisfied_papers
       - 不满意的反馈给 assistant 学习
       - 退出条件：本轮所有 include 论文用户都满意，或累计 satisfied ≥ 5
       - 未满足则继续迭代
    6. 全量筛选：用学习到的偏好对剩余所有论文进行筛选
    7. assistant 生成打分标准 → executor 对全量论文打分排序
    8. 对比系统判定的 include 论文与 gold standard

    Returns:
        各阶段的评估指标 + 与 gold standard 的对比结果
    """
    task_id = review_id
    user_id = f"test_{review_id}"

    logger.info(f"{'='*60}")
    logger.info(f"开始处理 review: {review_id}")
    logger.info(f"Review title: {review_meta.get('title', '')[:200]}")
    logger.info(f"总论文数: {len(paper_list)}, gold included: {len(gold_included_ids)}")

    # ─── 阶段 1: 信息抽取 ───
    logger.info(f"[阶段1] 信息抽取：以 review title 作为初始 query，逐步补充信息")

    _patch_memory_store(store)
    assistant = Assistant()

    # 1a. 初始 query = review title
    initial_query = build_initial_query(review_meta)
    logger.info(f"[阶段1a] 初始 query: {initial_query[:200]}")
    assistant.extract_and_record(initial_query, task_id, user_id)

    # 1b. 后续轮补充输入
    supplementary_inputs = build_supplementary_inputs(review_meta)
    for i, supp_input in enumerate(supplementary_inputs):
        logger.info(f"[阶段1b-{i+1}] 补充输入 (前200字): {supp_input[:200]}")
        assistant.extract_and_record(supp_input, task_id, user_id)

    # 1c. assistant 提供上下文
    assistant_context = assistant.provide_context("筛选相关论文", task_id, user_id)

    # ─── 阶段 1.5: Metadata 初筛 ───
    all_papers_count = len(paper_list)

    if skip_prescreen:
        logger.info(f"[阶段1.5] 跳过 metadata 初筛，全量 {all_papers_count} 篇进入筛选")
        prescreened_papers = list(paper_list)
        prescreen_kept_ids = {p["_id"] for p in prescreened_papers}
    else:
        logger.info(f"[阶段1.5] Metadata 初筛: 从 {all_papers_count} 篇论文中过滤不符合元数据要求的")
        prescreened_papers = metadata_prescreen(paper_list)
        prescreen_kept_ids = {p["_id"] for p in prescreened_papers}

    store.write_paper_list(task_id, prescreened_papers)
    store.append_timeline(task_id, "coordinator", f"Metadata初筛完成: {all_papers_count} → {len(prescreened_papers)} 篇")

    prescreen_metrics = compute_metrics(prescreen_kept_ids, gold_included_ids)
    logger.info(f"[阶段1.5评估] Metadata初筛: {prescreen_metrics}")

    if not prescreened_papers:
        logger.warning(f"Metadata初筛后无论文保留，跳过后续阶段")
        return {
            "review_id": review_id,
            "review_title": review_meta.get("title", "")[:200],
            "review_type": review_meta.get("review_type", ""),
            "num_papers": all_papers_count,
            "num_gold_included": len(gold_included_ids),
            "prescreen_metrics": prescreen_metrics,
            "screening_metrics": None,
            "screening_include_metrics": None,
            "scoring_metrics": None,
            "gold_comparison": None,
        }

    # ─── 阶段 1.8: Title 筛选 ───
    # 仅根据标题快速排除明显不相关的论文，减少后续 abstract 筛选的工作量
    # 保守策略：标题明显无关才排除，不确定的保留到后续详细筛选
    title_screening_context = assistant.provide_context(
        "仅根据论文标题判断是否与研究方向相关，做保守筛选", task_id, user_id
    )
    title_excluded = []
    title_kept = []

    logger.info(f"[阶段1.8] Title 筛选: {len(prescreened_papers)} 篇论文")

    for batch_start in range(0, len(prescreened_papers), TITLE_SCREEN_BATCH_SIZE):
        batch = prescreened_papers[batch_start:batch_start + TITLE_SCREEN_BATCH_SIZE]

        titles_text = ""
        for j, paper in enumerate(batch):
            titles_text += f"\n--- 论文 {j+1} ---\n"
            titles_text += f"标题: {paper.get('title', '')}\n"
            titles_text += f"年份: {paper.get('year', '')}\n"

        result_str = chat_with_system(
            TITLE_SCREEN_SYSTEM_PROMPT,
            f"用户的研究需求和偏好:\n{title_screening_context}\n\n论文列表:{titles_text}",
        )
        result = extract_json(result_str)
        if result is None:
            logger.warning(f"  Title 筛选 JSON 解析失败，本批全部保留")
            for paper in batch:
                paper["title_screening_decision"] = "not_sure"
                title_kept.append(paper)
            continue

        if isinstance(result, dict):
            result = [result]

        decisions = {}
        for item in result:
            idx = item.get("index", 0) - 1
            if 0 <= idx < len(batch):
                decisions[idx] = item

        for j, paper in enumerate(batch):
            item = decisions.get(j, {"decision": "not_sure", "reason": "未返回判断结果"})
            decision = item.get("decision", "not_sure")
            reason = item.get("reason", "")

            paper["title_screening_decision"] = decision
            paper["title_screening_reason"] = reason

            if decision == "exclude":
                title_excluded.append(paper)
            else:
                # include 和 not_sure 都保留
                title_kept.append(paper)

    # 更新 prescreened_papers 为 title 筛选后保留的论文
    prescreened_papers = title_kept
    prescreen_kept_ids = {p["_id"] for p in prescreened_papers}

    store.write_paper_list(task_id, prescreened_papers)
    store.append_timeline(task_id, "coordinator", f"Title筛选完成: 排除 {len(title_excluded)} 篇, 保留 {len(title_kept)} 篇")

    title_screen_metrics = compute_metrics(prescreen_kept_ids, gold_included_ids)
    logger.info(f"[阶段1.8评估] Title筛选后: 排除={len(title_excluded)}, 保留={len(title_kept)}, 指标: {title_screen_metrics}")

    if not prescreened_papers:
        logger.warning(f"Title筛选后无论文保留，跳过后续阶段")
        return {
            "review_id": review_id,
            "review_title": review_meta.get("title", "")[:200],
            "review_type": review_meta.get("review_type", ""),
            "num_papers": all_papers_count,
            "num_gold_included": len(gold_included_ids),
            "prescreen_metrics": prescreen_metrics,
            "title_screen_metrics": title_screen_metrics,
            "screening_metrics": None,
            "screening_include_metrics": None,
            "scoring_metrics": None,
            "gold_comparison": None,
        }

    # ─── 阶段 2: 迭代样例筛选 + 用户反馈 ───
    # 设计逻辑（与 untitled.txt 步骤3-4 一致）：
    #   executor 根据 abstract 筛选，找到 3 篇 include 时暂停
    #   → 用 gold standard 模拟用户反馈，让 assistant 学习偏好
    #   → 退出条件：本轮所有 include 论文用户都满意，或累计 satisfied ≥ 5
    #   → 未满足退出条件则继续迭代（assistant 已学习不满意反馈，重新筛选）
    SAMPLE_INCLUDE_THRESHOLD = 3    # 每轮样例筛选找到 3 篇 include 就停
    MAX_SCREENING_ROUNDS = 5        # 兜底：最多迭代轮次
    SATISFIED_THRESHOLD = 5         # 累计满意论文达到此阈值即可进入全量筛选

    satisfied_papers = []
    all_included = []
    all_excluded = []
    all_not_sure = []
    screening_round = 0

    # 标记所有论文为未筛选
    for p in prescreened_papers:
        if "screening_decision" not in p:
            p["screening_decision"] = "unscreened"

    # 先检查是否已有历史 satisfied 记录（断点续跑等场景）
    existing_satisfied = store.read_satisfied_papers(task_id)
    if existing_satisfied:
        satisfied_papers = existing_satisfied
        logger.info(f"[阶段2] 已有 {len(satisfied_papers)} 篇历史满意论文")

    while screening_round < MAX_SCREENING_ROUNDS:
        screening_round += 1
        logger.info(f"[阶段2 轮次{screening_round}] 开始: satisfied={len(satisfied_papers)}/{SATISFIED_THRESHOLD}")

        # 检查累计 satisfied 是否已达标
        if len(satisfied_papers) >= SATISFIED_THRESHOLD:
            logger.info(f"  累计满意论文 {len(satisfied_papers)} ≥ {SATISFIED_THRESHOLD}，退出迭代")
            break

        # 2a. assistant 提供筛选上下文（每轮重新生成，因为 assistant 已学习新反馈）
        screening_context = assistant.provide_context(
            "对论文进行详细筛选，根据摘要判断是否纳入", task_id, user_id
        )

        # 2b. executor 批量筛选未处理的论文，找到 SAMPLE_INCLUDE_THRESHOLD 篇 include 就停
        round_included = []
        round_excluded = []
        round_not_sure = []

        unscreened = [(i, p) for i, p in enumerate(prescreened_papers) if p.get("screening_decision") == "unscreened"]
        logger.info(f"  未筛选论文: {len(unscreened)} 篇")

        if not unscreened:
            logger.info(f"  所有论文已筛选完毕，退出迭代")
            break

        for batch_start in range(0, len(unscreened), SCREEN_BATCH_SIZE):
            if len(round_included) >= SAMPLE_INCLUDE_THRESHOLD:
                logger.info(f"  已找到 {len(round_included)} 篇 include，暂停本轮筛选")
                break

            batch = unscreened[batch_start:batch_start + SCREEN_BATCH_SIZE]

            papers_text = ""
            for j, (orig_idx, paper) in enumerate(batch):
                papers_text += f"\n--- 论文 {j+1} ---\n"
                papers_text += f"标题: {paper.get('title', '')}\n"
                papers_text += f"摘要: {(paper.get('abstract', '') or paper.get('text', ''))[:500]}\n"

            result_str = chat_with_system(
                SCREEN_SYSTEM_PROMPT,
                f"用户需求和偏好:\n{screening_context}\n\n论文列表:{papers_text}",
            )
            result = extract_json(result_str)
            if result is None:
                logger.warning(f"  筛选 JSON 解析失败")
                for _, paper in batch:
                    paper["screening_decision"] = "not_sure"
                    paper["screening_reason"] = "解析失败"
                    round_not_sure.append(paper)
                continue

            if isinstance(result, dict):
                result = [result]

            decisions = {}
            for item in result:
                idx = item.get("index", 0) - 1
                if 0 <= idx < len(batch):
                    decisions[idx] = item

            for j, (orig_idx, paper) in enumerate(batch):
                item = decisions.get(j, {"decision": "not_sure", "reason": "未返回判断结果"})
                decision = item.get("decision", "not_sure")
                reason = item.get("reason", "")

                paper["screening_decision"] = decision
                paper["screening_reason"] = reason

                if decision == "include":
                    paper["include_reason"] = reason
                    round_included.append(paper)
                elif decision == "exclude":
                    round_excluded.append(paper)
                else:
                    round_not_sure.append(paper)

        logger.info(f"  本轮筛选: included={len(round_included)}, excluded={len(round_excluded)}, not_sure={len(round_not_sure)}")

        all_included.extend(round_included)
        all_excluded.extend(round_excluded)
        all_not_sure.extend(round_not_sure)

        store.write_paper_list(task_id, prescreened_papers)
        store.append_timeline(
            task_id, "executor",
            f"轮次{screening_round}筛选完成: {len(round_included)} include, {len(round_excluded)} exclude, {len(round_not_sure)} not sure"
        )

        # 2c. 模拟用户确认（基于 gold standard），让 assistant 学习偏好
        if not gold_standard or not round_included:
            if not round_included:
                logger.info(f"  本轮无 include 论文，退出迭代")
                break
            if not gold_standard:
                logger.info(f"  无 gold_standard，跳过用户反馈")
                break

        review_gold = gold_standard.get(review_id, {})
        round_satisfied = []
        round_rejected = []

        for paper in round_included + round_not_sure:
            doc_id = paper.get("_id", "")
            gold_info = review_gold.get(doc_id, {})
            gold_decision = gold_info.get("decision", "") if isinstance(gold_info, dict) else ""
            gold_reason = gold_info.get("reason_for_exclusion", "") if isinstance(gold_info, dict) else ""

            if gold_decision == "included":
                paper["user_satisfied"] = True
                paper["include_reason"] = paper.get("include_reason", "符合研究需求")
                round_satisfied.append(paper)
                satisfied_papers.append(paper)
                assistant.extract_and_record(
                    f"用户对论文 '{paper.get('title', '')}' 感到满意，认为该论文符合研究需求。",
                    task_id, user_id,
                )
            elif gold_decision == "excluded":
                paper["user_satisfied"] = False
                exclusion_reason = gold_reason or "该论文不符合研究需求"
                paper["rejection_reason"] = exclusion_reason
                round_rejected.append(paper)
                assistant.extract_and_record(
                    f"用户对论文 '{paper.get('title', '')}' 不满意，排除原因: {exclusion_reason}",
                    task_id, user_id,
                )
            else:
                paper["user_satisfied"] = "not_sure"

        # 记录满意的论文
        if satisfied_papers:
            store.write_satisfied_papers(task_id, satisfied_papers)

        store.append_timeline(
            task_id, "simulated_user",
            f"轮次{screening_round}用户反馈: {len(round_satisfied)} 满意, {len(round_rejected)} 不满意, 累计满意: {len(satisfied_papers)}/{SATISFIED_THRESHOLD}"
        )
        logger.info(f"  本轮反馈: satisfied={len(round_satisfied)}, rejected={len(round_rejected)}, 累计satisfied={len(satisfied_papers)}/{SATISFIED_THRESHOLD}")

        # 退出条件1：本轮所有 include 论文用户都满意 → 退出迭代，进入全量筛选
        # 退出条件2：累计 satisfied ≥ SATISFIED_THRESHOLD → 退出迭代
        all_round_include_satisfied = len(round_rejected) == 0 and len(round_satisfied) > 0
        if all_round_include_satisfied:
            logger.info(f"  本轮所有 include 论文用户都满意，退出迭代，进入全量筛选")
            break
        if len(satisfied_papers) >= SATISFIED_THRESHOLD:
            logger.info(f"  累计满意论文 {len(satisfied_papers)} ≥ {SATISFIED_THRESHOLD}，退出迭代")
            break

    logger.info(f"[阶段2 完成] 共 {screening_round} 轮迭代, satisfied={len(satisfied_papers)}, included={len(all_included)}, excluded={len(all_excluded)}, not_sure={len(all_not_sure)}")

    # ─── 阶段 2.5: 全量筛选剩余 unscreened 论文 ───
    # 迭代样例筛选只是为了收集用户反馈、学习偏好，每轮3篇include就停
    # 现在需要对剩余未筛选的论文做全量筛选（不再限制3篇就停）
    remaining_unscreened = [(i, p) for i, p in enumerate(prescreened_papers) if p.get("screening_decision") == "unscreened"]
    if remaining_unscreened:
        logger.info(f"[阶段2.5] 全量筛选剩余 {len(remaining_unscreened)} 篇 unscreened 论文")
        screening_context = assistant.provide_context(
            "对论文进行详细筛选，根据摘要判断是否纳入", task_id, user_id
        )

        for batch_start in range(0, len(remaining_unscreened), SCREEN_BATCH_SIZE):
            batch = remaining_unscreened[batch_start:batch_start + SCREEN_BATCH_SIZE]

            papers_text = ""
            for j, (orig_idx, paper) in enumerate(batch):
                papers_text += f"\n--- 论文 {j+1} ---\n"
                papers_text += f"标题: {paper.get('title', '')}\n"
                papers_text += f"摘要: {(paper.get('abstract', '') or paper.get('text', ''))[:500]}\n"

            result_str = chat_with_system(
                SCREEN_SYSTEM_PROMPT,
                f"用户需求和偏好:\n{screening_context}\n\n论文列表:{papers_text}",
            )
            result = extract_json(result_str)
            if result is None:
                logger.warning(f"  全量筛选 JSON 解析失败")
                for _, paper in batch:
                    paper["screening_decision"] = "not_sure"
                    paper["screening_reason"] = "解析失败"
                    all_not_sure.append(paper)
                continue

            if isinstance(result, dict):
                result = [result]

            decisions = {}
            for item in result:
                idx = item.get("index", 0) - 1
                if 0 <= idx < len(batch):
                    decisions[idx] = item

            for j, (orig_idx, paper) in enumerate(batch):
                item = decisions.get(j, {"decision": "not_sure", "reason": "未返回判断结果"})
                decision = item.get("decision", "not_sure")
                reason = item.get("reason", "")

                paper["screening_decision"] = decision
                paper["screening_reason"] = reason

                if decision == "include":
                    paper["include_reason"] = reason
                    all_included.append(paper)
                elif decision == "exclude":
                    all_excluded.append(paper)
                else:
                    all_not_sure.append(paper)

        store.write_paper_list(task_id, prescreened_papers)
        store.append_timeline(
            task_id, "executor",
            f"全量筛选完成: 累计 included={len(all_included)}, excluded={len(all_excluded)}, not_sure={len(all_not_sure)}"
        )
        logger.info(f"[阶段2.5 完成] 全量筛选后: included={len(all_included)}, excluded={len(all_excluded)}, not_sure={len(all_not_sure)}")
    else:
        logger.info(f"[阶段2.5] 无剩余 unscreened 论文，跳过全量筛选")

    # 评估筛选阶段
    screening_kept_ids = {p["_id"] for p in all_included + all_not_sure}
    screening_metrics = compute_metrics(screening_kept_ids, gold_included_ids)
    logger.info(f"[阶段2评估] 筛选阶段 (include+not_sure): {screening_metrics}")

    screening_include_ids = {p["_id"] for p in all_included}
    screening_include_metrics = compute_metrics(screening_include_ids, gold_included_ids)
    logger.info(f"[阶段2评估] 筛选阶段 (include-only): {screening_include_metrics}")

    # ─── 与 gold standard 对比 ───
    system_included_ids = screening_include_ids
    gold_comparison = compute_metrics(system_included_ids, gold_included_ids)
    logger.info(f"[Gold对比] 系统 include vs Gold included: P={gold_comparison.get('precision',0)}, R={gold_comparison.get('recall',0)}, F1={gold_comparison.get('f1',0)}")
    logger.info(f"  系统 include: {len(system_included_ids)}, Gold included: {len(gold_included_ids)}, 交集: {gold_comparison.get('num_true_positives',0)}")

    missed_ids = gold_included_ids - system_included_ids
    false_positive_ids = system_included_ids - gold_included_ids
    correct_ids = system_included_ids & gold_included_ids
    logger.info(f"  正确纳入 (TP): {len(correct_ids)}")
    logger.info(f"  遗漏 (FN): {len(missed_ids)}")
    logger.info(f"  误判 (FP): {len(false_positive_ids)}")

    # ─── 阶段 4: 打分排序（对全量论文列表打分） ───
    logger.info(f"[阶段4] 生成打分标准并对全量论文打分排序")

    criteria_text = assistant.generate_scoring_criteria(task_id, user_id)
    logger.info(f"打分标准已生成: {criteria_text[:200]}...")

    papers_to_score = prescreened_papers
    if not papers_to_score:
        logger.warning(f"没有需要打分的论文，跳过打分阶段")
        scoring_metrics = compute_metrics(set(), gold_included_ids)
    else:
        scored_papers = []

        for batch_start in range(0, len(papers_to_score), SCORE_BATCH_SIZE):
            batch = papers_to_score[batch_start:batch_start + SCORE_BATCH_SIZE]

            papers_text = ""
            for j, paper in enumerate(batch):
                papers_text += f"\n--- 论文 {j+1} ---\n"
                papers_text += f"标题: {paper.get('title', '')}\n"
                papers_text += f"摘要: {(paper.get('abstract', '') or paper.get('text', ''))[:500]}\n"

            result_str = chat_with_system(
                SCORE_SYSTEM_PROMPT,
                f"打分标准:\n{criteria_text}\n\n论文列表:{papers_text}",
            )
            result = extract_json(result_str)
            if result is None:
                logger.warning(f"打分 JSON 解析失败")
                for paper in batch:
                    paper["scores"] = {}
                    paper["total_score"] = 0
                    paper["brief_comment"] = "解析失败"
                continue

            if isinstance(result, dict):
                result = [result]

            score_map = {}
            for item in result:
                idx = item.get("index", 0) - 1
                if 0 <= idx < len(batch):
                    score_map[idx] = item

            for j, paper in enumerate(batch):
                item = score_map.get(j, {"total_score": 0, "scores": {}, "brief_comment": "未返回评分"})
                paper["scores"] = item.get("scores", {})
                paper["total_score"] = item.get("total_score", 0)
                paper["brief_comment"] = item.get("brief_comment", "")

        papers_to_score.sort(key=lambda x: x.get("total_score", 0), reverse=True)
        scored_papers = papers_to_score

        store.write_scored_papers(task_id, scored_papers)
        store.append_timeline(task_id, "executor", f"打分排序完成，共 {len(scored_papers)} 篇论文")

        scoring_retrieved_ids = {p["_id"] for p in scored_papers}
        scoring_metrics = compute_metrics(scoring_retrieved_ids, gold_included_ids)
        logger.info(f"[阶段4评估] 打分阶段: {scoring_metrics}")

        # Top-N 结果
        top_n = min(10, len(scored_papers))
        logger.info(f"Top {top_n} 论文:")
        for i, paper in enumerate(scored_papers[:top_n]):
            gt = "Y" if paper["_id"] in gold_included_ids else "N"
            logger.info(
                f"  [{i+1}] [GT:{gt}] {paper.get('title', 'N/A')[:80]} | "
                f"分数: {paper.get('total_score', 0)} | "
                f"评语: {paper.get('brief_comment', '')[:50]}"
            )

    # 文件已在写入时实时持久化，无需额外 dump

    # 保存详细的 gold 对比信息
    comparison_detail = {
        "review_id": review_id,
        "system_included_ids": sorted(system_included_ids),
        "gold_included_ids": sorted(gold_included_ids),
        "correct_ids": sorted(correct_ids),
        "missed_ids": sorted(missed_ids),
        "false_positive_ids": sorted(false_positive_ids),
    }
    with open(os.path.join(store.output_dir, f"gold_comparison_{task_id}.json"), "w", encoding="utf-8") as f:
        json.dump(comparison_detail, f, ensure_ascii=False, indent=2)

    return {
        "review_id": review_id,
        "review_title": review_meta.get("title", "")[:200],
        "review_type": review_meta.get("review_type", ""),
        "num_papers": all_papers_count,
        "num_gold_included": len(gold_included_ids),
        "num_screening_rounds": screening_round,
        "num_satisfied": len(satisfied_papers),
        "prescreen_metrics": prescreen_metrics,
        "title_screen_metrics": title_screen_metrics,
        "screening_metrics": screening_metrics,
        "screening_include_metrics": screening_include_metrics,
        "scoring_metrics": scoring_metrics,
        "gold_comparison": gold_comparison,
        "gold_comparison_detail": {
            "num_correct": len(correct_ids),
            "num_missed": len(missed_ids),
            "num_false_positive": len(false_positive_ids),
        },
    }


# ═══════════════════════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════════════════════

def generate_report(all_results: list[dict], test_params: dict) -> str:
    """生成测试报告

    Returns:
        报告文本
    """
    lines = []
    lines.append("=" * 80)
    lines.append("CSMeD-FT 测试报告")
    lines.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"测试参数: {json.dumps(test_params, ensure_ascii=False)}")
    lines.append("=" * 80)

    valid_results = [r for r in all_results if "error" not in r]
    error_results = [r for r in all_results if "error" in r]

    lines.append(f"\n总测试 review 数: {len(all_results)}")
    lines.append(f"成功: {len(valid_results)}, 失败: {len(error_results)}")

    if error_results:
        lines.append(f"\n--- 失败的 review ---")
        for r in error_results:
            lines.append(f"  {r['review_id']}: {r.get('error', 'unknown')}")

    # ─── 每个 review 的详细结果 ───
    lines.append(f"\n{'='*80}")
    lines.append("逐条 Review 结果")
    lines.append(f"{'='*80}")

    for r in valid_results:
        lines.append(f"\n  Review: {r['review_id']} ({r.get('review_type', '?')})")
        lines.append(f"    Title: {r.get('review_title', '')[:100]}")
        lines.append(f"    论文数: {r.get('num_papers', 0)}, Gold included: {r.get('num_gold_included', 0)}")
        lines.append(f"    迭代轮次: {r.get('num_screening_rounds', 0)}, 满意论文: {r.get('num_satisfied', 0)}")
        pm = r.get("prescreen_metrics")
        lines.append(f"    Metadata初筛: {pm}")
        tsm = r.get("title_screen_metrics")
        lines.append(f"    Title筛选: {tsm}")
        sm = r.get("screening_metrics")
        lines.append(f"    筛选(include+not_sure): {sm}")
        sim = r.get("screening_include_metrics")
        lines.append(f"    筛选(include-only): {sim}")
        scm = r.get("scoring_metrics")
        lines.append(f"    打分: {scm}")
        gc = r.get("gold_comparison")
        if gc:
            lines.append(f"    *** Gold对比: P={gc.get('precision',0)}, R={gc.get('recall',0)}, F1={gc.get('f1',0)} ***")
            gd = r.get("gold_comparison_detail", {})
            lines.append(f"        TP={gd.get('num_correct',0)}, FN={gd.get('num_missed',0)}, FP={gd.get('num_false_positive',0)}")

    # ─── 汇总统计 ───
    lines.append(f"\n{'='*80}")
    lines.append("汇总统计")
    lines.append(f"{'='*80}")

    # 计算各阶段平均指标
    stages = ["prescreen_metrics", "title_screen_metrics", "screening_metrics", "screening_include_metrics", "scoring_metrics", "gold_comparison"]
    avg_metrics = {}
    for stage in stages:
        avg = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        count = 0
        for r in valid_results:
            m = r.get(stage)
            if m:
                avg["precision"] += m["precision"]
                avg["recall"] += m["recall"]
                avg["f1"] += m["f1"]
                count += 1
        if count > 0:
            avg["precision"] = round(avg["precision"] / count, 4)
            avg["recall"] = round(avg["recall"] / count, 4)
            avg["f1"] = round(avg["f1"] / count, 4)
        avg_metrics[stage] = avg

    lines.append(f"\n  Metadata初筛 (平均): {avg_metrics['prescreen_metrics']}")
    lines.append(f"  Title筛选 (平均): {avg_metrics['title_screen_metrics']}")
    lines.append(f"  筛选 include+not_sure (平均): {avg_metrics['screening_metrics']}")
    lines.append(f"  筛选 include-only (平均): {avg_metrics['screening_include_metrics']}")
    lines.append(f"  打分排序 (平均): {avg_metrics['scoring_metrics']}")
    lines.append(f"\n  *** Gold Standard 对比 (平均): {avg_metrics['gold_comparison']} ***")

    # ─── 按综述类型分组统计 ───
    by_type: dict[str, list] = {}
    for r in valid_results:
        rt = r.get("review_type", "Unknown")
        if rt not in by_type:
            by_type[rt] = []
        by_type[rt].append(r)

    lines.append(f"\n--- 按 review_type 分组 ---")
    for rt, results in sorted(by_type.items()):
        n = len(results)
        try:
            avg_p = sum(r.get("gold_comparison", {}).get("precision", 0) for r in results) / n if n else 0
            avg_r = sum(r.get("gold_comparison", {}).get("recall", 0) for r in results) / n if n else 0
            avg_f1 = sum(r.get("gold_comparison", {}).get("f1", 0) for r in results) / n if n else 0
        except (TypeError, ZeroDivisionError):
            avg_p, avg_r, avg_f1 = 0.0, 0.0, 0.0
        lines.append(f"  {rt} ({n} reviews): P={avg_p:.4f}, R={avg_r:.4f}, F1={avg_f1:.4f}")

    # ─── 汇总 TP/FP/FN ───
    try:
        total_tp = sum(r.get("gold_comparison_detail", {}).get("num_correct", 0) for r in valid_results)
        total_fp = sum(r.get("gold_comparison_detail", {}).get("num_false_positive", 0) for r in valid_results)
        total_fn = sum(r.get("gold_comparison_detail", {}).get("num_missed", 0) for r in valid_results)
        total_gold = sum(r.get("num_gold_included", 0) for r in valid_results)
    except (TypeError, KeyError):
        total_tp, total_fp, total_fn, total_gold = 0, 0, 0, 0

    lines.append(f"\n--- 全局 TP/FP/FN 汇总 ---")
    lines.append(f"  总 Gold included 论文数: {total_gold}")
    lines.append(f"  总 TP (系统 include & Gold include): {total_tp}")
    lines.append(f"  总 FP (系统 include & Gold exclude): {total_fp}")
    lines.append(f"  总 FN (系统 exclude & Gold include): {total_fn}")

    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    lines.append(f"  全局 Precision: {overall_p:.4f}")
    lines.append(f"  全局 Recall: {overall_r:.4f}")
    lines.append(f"  全局 F1: {overall_f1:.4f}")

    lines.append(f"\n{'='*80}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="CSMeD-FT 数据集测试脚本")
    parser.add_argument(
        "--num-reviews", type=int, default=0,
        help="测试的 review 数量（0=全部）",
    )
    parser.add_argument("--offset", type=int, default=0, help="从第几个 review 开始（默认 0）")
    args = parser.parse_args()

    # CSMeD 数据集有完整 metadata，做 metadata 初筛（与正常流程一致）
    skip_prescreen = False

    test_params = {
        "num_reviews": args.num_reviews,
        "offset": args.offset,
        "skip_prescreen": skip_prescreen,
    }

    logger.info(f"测试参数: {test_params}")

    # 加载数据集
    reviews_metadata, papers_dict, gold_standard = load_csmed_data()

    # 只处理在 gold_standard 中有记录的 review
    review_ids = sorted(set(reviews_metadata.keys()) & set(gold_standard.keys()))
    logger.info(f"可测试的 review 数: {len(review_ids)}")

    # 选取测试 review
    start = args.offset
    end = len(review_ids) if args.num_reviews == 0 else min(start + args.num_reviews, len(review_ids))
    selected_ids = review_ids[start:end]
    logger.info(f"本次测试 {len(selected_ids)} 个 review: {selected_ids}")

    all_results = []
    for idx, rid in enumerate(selected_ids):
        # 每条 review 使用独立的内存存储，实时写入对应目录
        result_dir = os.path.join(TEST_DATA_DIR, rid)
        store = InMemoryStore(result_dir)

        review_meta = reviews_metadata[rid]

        # 构建 paper list：使用 CSMeD-FT-papers 中的所有论文（而非仅 gold_standard 中的）
        paper_list = build_paper_list_for_review(set(papers_dict.keys()), papers_dict)

        # gold standard 中 included 的论文
        gold_included = get_gold_included_ids(gold_standard, rid)

        try:
            result = run_single_review(rid, review_meta, paper_list, gold_included, store, skip_prescreen, gold_standard)
            all_results.append(result)
        except Exception as e:
            logger.error(f"处理 review {rid} 时出错: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "review_id": rid,
                "review_title": review_meta.get("title", "")[:200],
                "error": str(e),
            })

        # review 间短暂等待
        if idx < len(selected_ids) - 1:
            logger.info("等待 3 秒...")
            time.sleep(3)

    # 恢复原始 memory_store
    _restore_memory_store()

    # ─── 生成报告 ───
    report = generate_report(all_results, test_params)
    logger.info(f"\n{report}")

    # 保存报告
    report_path = os.path.join(TEST_DATA_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"报告已保存: {report_path}")

    # 保存详细 JSON 结果
    summary = {
        "test_time": datetime.now().isoformat(),
        "test_params": test_params,
        "num_reviews": len([r for r in all_results if "error" not in r]),
        "per_review_results": all_results,
    }
    summary_path = os.path.join(TEST_DATA_DIR, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"详细结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
