"""LitSearchRetrieval 数据集测试脚本

使用 LitSearchRetrieval 数据集测试项目的 assistant → coordinator → executor 工作流。
仅使用数据集内部数据，不调用 arXiv / Semantic Scholar API，也不使用 simulated user。

工作流：
1. 用户提出 initial query（来自数据集的 queries），assistant 抽取信息并记录，
   coordinator 读取 query 和 assistant 提供的信息，将全量 corpus 论文列表交给 executor。
   由于数据集论文无 metadata 可供初步筛选，先用 LLM 根据论文标题进行快速初筛，
   排除标题明显不相关的论文，替代原 workflow 中的 metadata 筛选。
2. executor 对标题初筛后保留的论文，根据 abstract 进行详细筛选
   （include/exclude/not_sure），include 的论文记录理由。
3. assistant 生成多维度打分标准，coordinator 对 include 和 not_sure 的论文打分，
   按分数降序排序，记录到单独文件中。

评估：用 qrels 标注计算 recall/precision 等指标。
"""
import json
import logging
import os
import sys
import time
from datetime import datetime

# ── 路径设置 ──────────────────────────────────────────────
# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 测试数据输出目录（不影响原 data 目录）
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "data_litsearch_test")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# LitSearchRetrieval 数据集路径
DATASET_DIR = os.path.join(PROJECT_ROOT, "LitSearchRetrieval")

# ── 日志配置 ──────────────────────────────────────────────
log_filename = f"test_litsearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(TEST_DATA_DIR, log_filename), encoding="utf-8"),
    ],
)
logger = logging.getLogger("test_litsearch")

# ── 导入项目模块 ──────────────────────────────────────────
# 必须在 sys.path 设置之后
from llm_client import chat_with_system, extract_json
from assistant import Assistant
from executor import SCREEN_SYSTEM_PROMPT, SCREEN_BATCH_SIZE, SCORE_SYSTEM_PROMPT, SCORE_BATCH_SIZE
import memory_store as ms


# ═══════════════════════════════════════════════════════════
# 数据集加载（无需 pandas，使用 pyarrow 原生接口）
# ═══════════════════════════════════════════════════════════

def load_dataset():
    """加载 LitSearchRetrieval 数据集的三个表，返回 (corpus, queries, qrels)"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("需要安装 pyarrow: pip install pyarrow")
        sys.exit(1)

    corpus_table = pq.read_table(os.path.join(DATASET_DIR, "corpus", "test-00000-of-00001.parquet"))
    queries_table = pq.read_table(os.path.join(DATASET_DIR, "queries", "test-00000-of-00001.parquet"))
    qrels_table = pq.read_table(os.path.join(DATASET_DIR, "qrels", "test-00000-of-00001.parquet"))

    corpus = corpus_table.to_pydict()
    queries = queries_table.to_pydict()
    qrels = qrels_table.to_pydict()

    logger.info(f"数据集加载完成: corpus={len(corpus['_id'])}, queries={len(queries['_id'])}, qrels={len(qrels['query-id'])}")
    return corpus, queries, qrels


def corpus_to_paper_list(corpus: dict) -> list[dict]:
    """将 corpus 转换为项目内部格式的论文列表"""
    papers = []
    for i in range(len(corpus["_id"])):
        papers.append({
            "_id": corpus["_id"][i],
            "title": corpus["title"][i] if corpus["title"][i] else "",
            "abstract": corpus["text"][i] if corpus["text"][i] else "",
            "text": corpus["text"][i] if corpus["text"][i] else "",
            "source": "litsearch_corpus",
        })
    return papers


def get_relevant_corpus_ids(qrels: dict, query_id: str) -> set[str]:
    """获取某个 query 的所有相关 corpus-id（qrels 中 score > 0 的）"""
    relevant = set()
    for i in range(len(qrels["query-id"])):
        if qrels["query-id"][i] == query_id and qrels["score"][i] > 0:
            relevant.add(qrels["corpus-id"][i])
    return relevant


# ═══════════════════════════════════════════════════════════
# 内存版 memory_store（不影响原 data 目录）
# ═══════════════════════════════════════════════════════════

class InMemoryStore:
    """基于内存的存储，替换 memory_store 的文件 I/O，测试完成后不残留文件"""

    def __init__(self):
        self.user_info: dict[str, str] = {}
        self.research_task: dict[str, str] = {}
        self.paper_list: dict[str, list] = {}
        self.satisfied_papers: dict[str, list] = {}
        self.scoring_criteria: dict[str, str] = {}
        self.scored_papers: dict[str, list] = {}
        self.timeline: dict[str, list] = {}

    # ── 读取 ──
    def read_user_info(self, user_id: str = "") -> str:
        return self.user_info.get(user_id, "")

    def write_user_info(self, content: str, user_id: str = "") -> None:
        self.user_info[user_id] = content

    def read_research_task(self, task_id: str) -> str:
        return self.research_task.get(task_id, "")

    def write_research_task(self, task_id: str, content: str) -> None:
        self.research_task[task_id] = content

    def read_paper_list(self, task_id: str) -> list:
        return self.paper_list.get(task_id, [])

    def write_paper_list(self, task_id: str, data: list) -> None:
        self.paper_list[task_id] = data

    def read_satisfied_papers(self, task_id: str) -> list:
        return self.satisfied_papers.get(task_id, [])

    def write_satisfied_papers(self, task_id: str, data: list) -> None:
        self.satisfied_papers[task_id] = data

    def read_scoring_criteria(self, task_id: str) -> str:
        return self.scoring_criteria.get(task_id, "")

    def write_scoring_criteria(self, task_id: str, content: str) -> None:
        self.scoring_criteria[task_id] = content

    def read_scored_papers(self, task_id: str) -> list:
        return self.scored_papers.get(task_id, [])

    def write_scored_papers(self, task_id: str, data: list) -> None:
        self.scored_papers[task_id] = data

    def append_timeline(self, task_id: str, role: str, content: str) -> None:
        if task_id not in self.timeline:
            self.timeline[task_id] = []
        self.timeline[task_id].append(f"[{role}] {content}")

    def dump_to_files(self, task_id: str, output_dir: str) -> None:
        """将内存数据持久化到文件（用于结果分析）"""
        os.makedirs(output_dir, exist_ok=True)

        if task_id in self.research_task:
            with open(os.path.join(output_dir, f"research_task_{task_id}.txt"), "w", encoding="utf-8") as f:
                f.write(self.research_task[task_id])

        if task_id in self.paper_list:
            with open(os.path.join(output_dir, f"paper_list_{task_id}.json"), "w", encoding="utf-8") as f:
                json.dump(self.paper_list[task_id], f, ensure_ascii=False, indent=2)

        if task_id in self.scoring_criteria:
            with open(os.path.join(output_dir, f"scoring_criteria_{task_id}.txt"), "w", encoding="utf-8") as f:
                f.write(self.scoring_criteria[task_id])

        if task_id in self.scored_papers:
            with open(os.path.join(output_dir, f"scored_papers_{task_id}.json"), "w", encoding="utf-8") as f:
                json.dump(self.scored_papers[task_id], f, ensure_ascii=False, indent=2)

        if task_id in self.timeline:
            with open(os.path.join(output_dir, f"timeline_{task_id}.log"), "w", encoding="utf-8") as f:
                for entry in self.timeline[task_id]:
                    f.write(entry + "\n")


# ═══════════════════════════════════════════════════════════
# 标题初筛（替代 metadata 筛选）
# ═══════════════════════════════════════════════════════════

TITLE_PRESCREEN_BATCH_SIZE = 50  # 标题短，一批可以处理更多

TITLE_PRESCREEN_PROMPT = """你是一名学术文献筛选专家。你将看到一批论文的标题，请根据用户的研究需求，判断每篇论文的标题是否可能与该研究相关。

注意：仅根据标题做快速判断，宁可多保留也不要误排除。如果标题看起来有可能相关，就标记为 "maybe"；
只有标题明显不相关时才标记为 "irrelevant"。

请输出一个 JSON 数组，每个元素包含：
- "index": 论文编号（从1开始）
- "decision": "maybe" 或 "irrelevant"

只输出 JSON 数组，不要输出其他内容。"""


def title_prescreen(all_papers: list[dict], query_text: str, assistant_context: str) -> list[dict]:
    """基于论文标题的快速初筛，排除明显不相关的论文，替代 metadata 筛选。
    
    返回标题看起来可能相关的论文列表。
    """
    maybe_papers = []
    irrelevant_count = 0

    for batch_start in range(0, len(all_papers), TITLE_PRESCREEN_BATCH_SIZE):
        batch = all_papers[batch_start:batch_start + TITLE_PRESCREEN_BATCH_SIZE]

        titles_text = ""
        for j, paper in enumerate(batch):
            titles_text += f"{j+1}. {paper.get('title', 'N/A')}\n"

        result_str = chat_with_system(
            TITLE_PRESCREEN_PROMPT,
            f"用户研究需求:\n{query_text}\n\n补充背景:\n{assistant_context}\n\n论文标题列表:\n{titles_text}",
        )
        result = extract_json(result_str)

        if result is None:
            logger.warning(f"[标题初筛] 批次 {batch_start//TITLE_PRESCREEN_BATCH_SIZE + 1} JSON 解析失败，整批保留")
            maybe_papers.extend(batch)
            continue

        if isinstance(result, dict):
            result = [result]

        # 构建 index -> decision 映射
        decisions: dict[int, str] = {}
        for item in result:
            idx = item.get("index", 0) - 1
            if 0 <= idx < len(batch):
                decisions[idx] = item.get("decision", "maybe")

        for j, paper in enumerate(batch):
            decision = decisions.get(j, "maybe")  # 默认保留
            if decision == "irrelevant":
                irrelevant_count += 1
            else:
                maybe_papers.append(paper)

    logger.info(f"[标题初筛] 全量 {len(all_papers)} 篇 → 保留 {len(maybe_papers)} 篇, 排除 {irrelevant_count} 篇")
    return maybe_papers


# ═══════════════════════════════════════════════════════════
# 评估指标计算
# ═══════════════════════════════════════════════════════════

def compute_metrics(retrieved_ids: set[str], relevant_ids: set[str]) -> dict:
    """计算检索评估指标"""
    if not relevant_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_relevant": 0, "num_retrieved": len(retrieved_ids)}
    if not retrieved_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_relevant": len(relevant_ids), "num_retrieved": 0}

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
# 核心：单条 query 的测试流程
# ═══════════════════════════════════════════════════════════

def run_single_query(
    query_id: str,
    query_text: str,
    all_papers: list[dict],
    relevant_ids: set[str],
    store: InMemoryStore,
) -> dict:
    """
    对单条 query 运行完整的测试流程：
    1. assistant 抽取信息 → 全量 corpus 论文加载
    1.5. 基于论文标题的快速初筛（替代 metadata 筛选，排除标题明显不相关的论文）
    2. executor 对标题初筛保留的论文，根据 abstract 进行详细筛选（include/exclude/not_sure）
    3. assistant 生成打分标准 → executor 对 include+not_sure 打分排序

    返回各阶段的评估指标。
    """
    task_id = query_id
    user_id = f"test_{query_id}"

    logger.info(f"{'='*60}")
    logger.info(f"开始处理 query: {query_id}")
    logger.info(f"Query 文本: {query_text[:200]}")
    logger.info(f"相关论文数: {len(relevant_ids)}")

    # ─── 阶段 1: 信息抽取 + 全量论文列表 ───
    logger.info(f"[阶段1] 信息抽取与全量论文加载")

    # 1a. assistant 抽取信息
    assistant = Assistant()
    # 替换 memory_store 为内存版本
    _patch_memory_store(store)

    extracted = assistant.extract_and_record(query_text, task_id, user_id)
    logger.info(f"信息抽取结果: {json.dumps(extracted, ensure_ascii=False)}")

    # 1b. assistant 提供上下文（用于后续筛选阶段）
    assistant_context = assistant.provide_context("筛选相关论文", task_id, user_id)

    # 1c. 全量论文
    all_papers_count = len(all_papers)

    # ─── 阶段 1.5: 标题初筛（替代 metadata 筛选） ───
    logger.info(f"[阶段1.5] 标题初筛: 从 {all_papers_count} 篇论文中排除标题明显不相关的")

    prescreened_papers = title_prescreen(all_papers, query_text, assistant_context)
    store.write_paper_list(task_id, prescreened_papers)
    store.append_timeline(task_id, "coordinator", f"标题初筛完成: {all_papers_count} → {len(prescreened_papers)} 篇")

    # 评估标题初筛阶段
    prescreen_kept_ids = {p["_id"] for p in prescreened_papers}
    prescreen_metrics = compute_metrics(prescreen_kept_ids, relevant_ids)
    logger.info(f"[阶段1.5评估] 标题初筛: {prescreen_metrics}")

    if not prescreened_papers:
        logger.warning(f"标题初筛后无论文保留，跳过后续阶段")
        return {
            "query_id": query_id,
            "query_text": query_text[:200],
            "search_metrics": compute_metrics({p["_id"] for p in all_papers}, relevant_ids),
            "prescreen_metrics": prescreen_metrics,
            "screening_metrics": None,
            "scoring_metrics": None,
        }

    # ─── 阶段 2: abstract 详细筛选 ───
    logger.info(f"[阶段2] abstract 详细筛选 (include/exclude/not_sure), 共 {len(prescreened_papers)} 篇")

    # 2a. assistant 提供上下文给 executor
    screening_context = assistant.provide_context(
        "对论文进行详细筛选，根据摘要判断是否纳入", task_id, user_id
    )

    # 2b. executor 筛选（对标题初筛保留的论文逐批按 abstract 筛选）
    papers = prescreened_papers
    included = []
    excluded = []
    not_sure = []

    unscreened = [(i, p) for i, p in enumerate(papers) if "screening_decision" not in p]
    logger.info(f"初步筛选: 共 {len(papers)} 篇, 未筛选 {len(unscreened)} 篇")

    for batch_start in range(0, len(unscreened), SCREEN_BATCH_SIZE):
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
            logger.warning(f"筛选 JSON 解析失败")
            for _, paper in batch:
                paper["screening_decision"] = "not_sure"
                paper["screening_reason"] = "解析失败"
                not_sure.append(paper)
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
                included.append(paper)
            elif decision == "exclude":
                excluded.append(paper)
            else:
                not_sure.append(paper)

    store.write_paper_list(task_id, papers)
    store.append_timeline(
        task_id, "executor",
        f"筛选完成: {len(included)} include, {len(excluded)} exclude, {len(not_sure)} not sure"
    )
    logger.info(f"[阶段2] 筛选结果: included={len(included)}, excluded={len(excluded)}, not_sure={len(not_sure)}")

    # 评估筛选阶段（include + not_sure 视为"保留"）
    screening_kept_ids = {p["_id"] for p in included + not_sure}
    screening_metrics = compute_metrics(screening_kept_ids, relevant_ids)
    logger.info(f"[阶段2评估] 筛选阶段 (include+not_sure): {screening_metrics}")

    # ─── 阶段 3: 打分排序 ───
    logger.info(f"[阶段3] 生成打分标准并对论文打分排序")

    # 3a. assistant 生成打分标准
    criteria_text = assistant.generate_scoring_criteria(task_id, user_id)
    logger.info(f"打分标准已生成: {criteria_text[:200]}...")

    # 3b. 只对 include + not_sure 的论文打分
    papers_to_score = included + not_sure
    if not papers_to_score:
        logger.warning(f"没有需要打分的论文，跳过打分阶段")
        scoring_metrics = compute_metrics(set(), relevant_ids)
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

        # 按总分降序排序
        papers_to_score.sort(key=lambda x: x.get("total_score", 0), reverse=True)
        scored_papers = papers_to_score

        store.write_scored_papers(task_id, scored_papers)
        store.append_timeline(task_id, "executor", f"打分排序完成，共 {len(scored_papers)} 篇论文")

        # 评估打分阶段
        scoring_retrieved_ids = {p["_id"] for p in scored_papers}
        scoring_metrics = compute_metrics(scoring_retrieved_ids, relevant_ids)
        logger.info(f"[阶段3评估] 打分阶段: {scoring_metrics}")

        # 输出 Top 论文
        top_n = min(10, len(scored_papers))
        logger.info(f"Top {top_n} 论文:")
        for i, paper in enumerate(scored_papers[:top_n]):
            logger.info(f"  [{i+1}] {paper.get('title', 'N/A')[:80]} | 分数: {paper.get('total_score', 0)} | 评语: {paper.get('brief_comment', '')[:50]}")

    # 持久化到文件
    result_dir = os.path.join(TEST_DATA_DIR, task_id)
    store.dump_to_files(task_id, result_dir)

    return {
        "query_id": query_id,
        "query_text": query_text[:200],
        "search_metrics": compute_metrics({p["_id"] for p in all_papers}, relevant_ids),
        "prescreen_metrics": prescreen_metrics,
        "screening_metrics": screening_metrics,
        "scoring_metrics": scoring_metrics,
    }


# ═══════════════════════════════════════════════════════════
# memory_store 热替换（测试期间用内存存储替代文件存储）
# ═══════════════════════════════════════════════════════════

_original_ms_functions = {}


def _patch_memory_store(store: InMemoryStore) -> None:
    """将 memory_store 模块的函数替换为内存版本，不影响原文件"""
    global _original_ms_functions

    # 只在第一次调用时保存原始函数
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
    """恢复 memory_store 的原始函数"""
    global _original_ms_functions
    if _original_ms_functions:
        for name, func in _original_ms_functions.items():
            setattr(ms, name, func)
        _original_ms_functions = {}


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LitSearchRetrieval 数据集测试脚本")
    parser.add_argument("--num-queries", type=int, default=5, help="测试的 query 数量（默认 5）")
    parser.add_argument("--offset", type=int, default=0, help="从第几条 query 开始（默认 0）")
    args = parser.parse_args()

    logger.info(f"测试参数: num_queries={args.num_queries}, offset={args.offset}, 全量screen模式")

    # 加载数据集
    corpus, queries, qrels = load_dataset()

    # 构建论文列表
    all_papers = corpus_to_paper_list(corpus)
    logger.info(f"构建论文列表完成: {len(all_papers)} 篇")

    # 获取所有 query ID
    query_ids = queries["_id"]
    query_texts = queries["text"]

    # 选取测试 query
    start = args.offset
    end = min(start + args.num_queries, len(query_ids))

    all_results = []
    for i in range(start, end):
        qid = query_ids[i]
        qtext = query_texts[i]
        relevant = get_relevant_corpus_ids(qrels, qid)

        # 每条 query 使用独立的内存存储
        store = InMemoryStore()
        _patch_memory_store(store)

        try:
            result = run_single_query(qid, qtext, all_papers, relevant, store)
            all_results.append(result)
        except Exception as e:
            logger.error(f"处理 query {qid} 时出错: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "query_id": qid,
                "query_text": qtext[:200],
                "error": str(e),
            })

        # query 间短暂等待，避免 LLM API 限流
        if i < end - 1:
            logger.info("等待 3 秒...")
            time.sleep(3)

    # 恢复原始 memory_store
    _restore_memory_store()

    # ─── 汇总结果 ───
    logger.info(f"\n{'='*60}")
    logger.info(f"汇总结果 ({len(all_results)} 条 query)")
    logger.info(f"{'='*60}")

    # 计算平均指标
    avg_search = {"precision": 0, "recall": 0, "f1": 0}
    avg_prescreen = {"precision": 0, "recall": 0, "f1": 0}
    avg_screening = {"precision": 0, "recall": 0, "f1": 0}
    avg_scoring = {"precision": 0, "recall": 0, "f1": 0}
    valid_count = 0

    for r in all_results:
        if "error" in r:
            logger.info(f"  Query {r['query_id']}: ERROR - {r['error']}")
            continue
        valid_count += 1
        logger.info(f"  Query {r['query_id']}:")
        logger.info(f"    全量: {r['search_metrics']}")
        logger.info(f"    标题初筛: {r.get('prescreen_metrics')}")
        logger.info(f"    abstract筛选: {r['screening_metrics']}")
        logger.info(f"    打分: {r['scoring_metrics']}")

        for stage, avg in [
            ("search_metrics", avg_search),
            ("prescreen_metrics", avg_prescreen),
            ("screening_metrics", avg_screening),
            ("scoring_metrics", avg_scoring),
        ]:
            m = r.get(stage)
            if m:
                avg["precision"] += m["precision"]
                avg["recall"] += m["recall"]
                avg["f1"] += m["f1"]

    if valid_count > 0:
        for avg in [avg_search, avg_prescreen, avg_screening, avg_scoring]:
            avg["precision"] = round(avg["precision"] / valid_count, 4)
            avg["recall"] = round(avg["recall"] / valid_count, 4)
            avg["f1"] = round(avg["f1"] / valid_count, 4)

    logger.info(f"\n平均指标 ({valid_count} 条有效 query):")
    logger.info(f"  全量加载: {avg_search}")
    logger.info(f"  标题初筛: {avg_prescreen}")
    logger.info(f"  abstract筛选: {avg_screening}")
    logger.info(f"  打分排序: {avg_scoring}")

    # 保存汇总结果
    summary = {
        "test_time": datetime.now().isoformat(),
        "num_queries": valid_count,
        "avg_search_metrics": avg_search,
        "avg_prescreen_metrics": avg_prescreen,
        "avg_screening_metrics": avg_screening,
        "avg_scoring_metrics": avg_scoring,
        "per_query_results": all_results,
    }
    summary_path = os.path.join(TEST_DATA_DIR, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"汇总结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
