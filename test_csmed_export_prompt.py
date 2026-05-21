"""CSMeD-FT 数据集测试脚本 - 导出阶段2.5 executor prompt

基于 test_csmed.py，运行到阶段2.5时记录 executor 收到的完整 prompt，
然后跳过实际筛选和后续步骤。

输出: executor_prompt_record.txt，每行一个 JSON:
  {"review_id": "xxx", "executor_prompt": "xxx"}
"""

import json
import logging
import os
import re
import sys
from datetime import datetime


def _parse_index(raw_idx) -> int:
    """从 LLM 返回的 index 字段提取数字（兼容 '论文1', '1', 1 等格式）"""
    m = re.search(r'\d+', str(raw_idx))
    return int(m.group()) if m else 0

# ── 路径设置 ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 测试数据输出目录
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "data_csmed_prompt_export")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# prompt 记录文件
PROMPT_RECORD_PATH = os.path.join(TEST_DATA_DIR, "executor_prompt_record.txt")

# CSMeD-FT 数据集路径
CSMED_DATA_DIR = r"D:\School\SAT404\pilot_study_code\systematic-review-datasets\data\CSMeD\CSMeD-FT\CSMeD-FT"

# ── 日志配置 ──────────────────────────────────────────────
log_filename = f"export_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(TEST_DATA_DIR, log_filename), encoding="utf-8"),
    ],
)
logger = logging.getLogger("export_prompt")

# ── 导入项目模块 ──────────────────────────────────────────
from llm_client import chat_with_system, extract_json
from memory_store import InMemoryStore
from assistant import Assistant
from executor import SCREEN_SYSTEM_PROMPT, SCREEN_BATCH_SIZE, TITLE_SCREEN_SYSTEM_PROMPT, TITLE_SCREEN_BATCH_SIZE


# ═══════════════════════════════════════════════════════════
# 数据集加载（与 test_csmed.py 一致）
# ═══════════════════════════════════════════════════════════

def load_csmed_data():
    """加载合并后的 CSMeD-FT 数据集"""
    reviews_path = os.path.join(CSMED_DATA_DIR, "CSMeD-FT-all_reviews_metadata.json")
    with open(reviews_path, encoding="utf-8") as f:
        reviews_metadata = json.load(f)
    logger.info(f"Reviews metadata 加载完成: {len(reviews_metadata)} 个 reviews")

    papers_path = os.path.join(CSMED_DATA_DIR, "CSMeD-FT-papers.json")
    with open(papers_path, encoding="utf-8") as f:
        papers_list = json.load(f)
    papers_dict = {p["document_id"]: p for p in papers_list}
    logger.info(f"论文集加载完成: {len(papers_dict)} 篇去重论文")

    gold_path = os.path.join(CSMED_DATA_DIR, "CSMeD-FT-gold_standard.json")
    with open(gold_path, encoding="utf-8") as f:
        gold_list = json.load(f)

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
    review_gold = gold_standard.get(review_id, {})
    return {did for did, info in review_gold.items() if (info if isinstance(info, dict) else {"decision": info}).get("decision") == "included"}


def build_paper_list_for_review(paper_ids: set[str], papers_dict: dict) -> list[dict]:
    result = []
    for did in sorted(paper_ids):
        p = papers_dict.get(did)
        if p is None:
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
# 构建 user query 和后续输入（与 test_csmed.py 一致）
# ═══════════════════════════════════════════════════════════

def build_initial_query(review_meta: dict) -> str:
    return review_meta.get("title", "")


def build_supplementary_inputs(review_meta: dict) -> list[str]:
    inputs = []

    review_type = review_meta.get("review_type", "")
    abstract = review_meta.get("abstract", "")
    if review_type or abstract:
        parts = []
        if review_type:
            parts.append(f"研究类型 (review_type): {review_type}")
        if abstract:
            parts.append(f"系统综述摘要:\n{abstract}")
        inputs.append("\n\n".join(parts))

    criteria_text = review_meta.get("criteria_text", "")
    if criteria_text:
        inputs.append(f"纳入/排除标准 (criteria_text):\n{criteria_text}")

    criteria = review_meta.get("criteria", "")
    if criteria:
        if isinstance(criteria, dict):
            criteria_str = json.dumps(criteria, ensure_ascii=False, indent=2)
        else:
            criteria_str = str(criteria)
        inputs.append(f"结构化纳入/排除标准 (criteria):\n{criteria_str}")

    search_strategy = review_meta.get("search_strategy", "")
    if search_strategy:
        if isinstance(search_strategy, dict):
            ss_str = json.dumps(search_strategy, ensure_ascii=False, indent=2)
        else:
            ss_str = str(search_strategy)
        inputs.append(f"搜索策略 (search_strategy):\n{ss_str}")

    return inputs


# ═══════════════════════════════════════════════════════════
# Metadata 初筛（与 test_csmed.py 一致）
# ═══════════════════════════════════════════════════════════

def metadata_prescreen(all_papers: list[dict], year_range: str | None = None) -> list[dict]:
    filtered_papers = []
    filtered_out = 0

    for p in all_papers:
        if not p.get("abstract") or not p["abstract"].strip():
            filtered_out += 1
            continue
        if not p.get("title") or not p["title"].strip():
            filtered_out += 1
            continue
        if year_range and p.get("year") is None:
            filtered_out += 1
            continue
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
# 核心：单个 review 的导出流程
# ═══════════════════════════════════════════════════════════

def run_single_review_export_prompt(
    review_id: str,
    review_meta: dict,
    paper_list: list[dict],
    gold_included_ids: set[str],
    store: InMemoryStore,
    skip_prescreen: bool = False,
    gold_standard: dict[str, dict[str, str]] | None = None,
) -> dict:
    """运行到阶段2.5，记录 executor 收到的 prompt，然后跳过后续步骤。

    返回:
        {"review_id": xxx, "executor_prompt": xxx}
    """
    task_id = review_id
    user_id = f"test_{review_id}"

    logger.info(f"{'='*60}")
    logger.info(f"开始处理 review: {review_id}")
    logger.info(f"Review title: {review_meta.get('title', '')[:200]}")
    logger.info(f"总论文数: {len(paper_list)}, gold included: {len(gold_included_ids)}")

    # ─── 阶段 1: 信息抽取 ───
    logger.info(f"[阶段1] 信息抽取：以 review title 作为初始 query，逐步补充信息")

    assistant = Assistant(store=store)

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
    assistant.provide_context("筛选相关论文", task_id, user_id)

    # ─── 阶段 1.5: Metadata 初筛（已跳过）───
    all_papers_count = len(paper_list)
    prescreened_papers = list(paper_list)
    logger.info(f"[阶段1.5] 跳过 metadata 初筛，全量 {all_papers_count} 篇进入筛选")

    # ─── 阶段 1.8: Title 筛选（已跳过）───
    logger.info(f"[阶段1.8] 跳过 Title 筛选，全量 {len(prescreened_papers)} 篇进入阶段2")

    # ─── 阶段 2: 迭代样例筛选 + 用户反馈 ───
    SAMPLE_INCLUDE_THRESHOLD = 3
    MAX_SCREENING_ROUNDS = 5
    SATISFIED_THRESHOLD = 5

    satisfied_papers = []
    all_included = []
    all_excluded = []
    all_not_sure = []
    screening_round = 0

    for p in prescreened_papers:
        if "screening_decision" not in p:
            p["screening_decision"] = "unscreened"

    existing_satisfied = store.read_satisfied_papers(task_id)
    if existing_satisfied:
        satisfied_papers = existing_satisfied
        logger.info(f"[阶段2] 已有 {len(satisfied_papers)} 篇历史满意论文")

    while screening_round < MAX_SCREENING_ROUNDS:
        screening_round += 1
        logger.info(f"[阶段2 轮次{screening_round}] 开始: satisfied={len(satisfied_papers)}/{SATISFIED_THRESHOLD}")

        if len(satisfied_papers) >= SATISFIED_THRESHOLD:
            logger.info(f"  累计满意论文 {len(satisfied_papers)} ≥ {SATISFIED_THRESHOLD}，退出迭代")
            break

        screening_context = assistant.provide_context(
            "对论文进行详细筛选，根据摘要判断是否纳入", task_id, user_id
        )

        round_included = []
        round_excluded = []
        round_not_sure = []

        unscreened = [(i, p) for i, p in enumerate(prescreened_papers) if p.get("screening_decision") == "unscreened"]
        logger.info(f"  未筛选论文: {len(unscreened)} 篇")

        if not unscreened:
            logger.info(f"  所有论文已筛选完毕，退出迭代")
            break

        # 累计已有2篇include时，每轮筛选不超过20篇
        if len(all_included) >= 2 and len(unscreened) > 20:
            logger.info(f"  累计已有 {len(all_included)} 篇 include，本轮限制筛选 20 篇 (共 {len(unscreened)} 篇待筛选)")
            unscreened = unscreened[:20]

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
                idx = _parse_index(item.get("index", 0)) - 1
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

        # 2c. 模拟用户确认
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

        if satisfied_papers:
            store.write_satisfied_papers(task_id, satisfied_papers)

        store.append_timeline(
            task_id, "simulated_user",
            f"轮次{screening_round}用户反馈: {len(round_satisfied)} 满意, {len(round_rejected)} 不满意, 累计满意: {len(satisfied_papers)}/{SATISFIED_THRESHOLD}"
        )
        logger.info(f"  本轮反馈: satisfied={len(round_satisfied)}, rejected={len(round_rejected)}, 累计satisfied={len(satisfied_papers)}/{SATISFIED_THRESHOLD}")

        all_round_include_satisfied = len(round_rejected) == 0 and len(round_satisfied) > 0
        if all_round_include_satisfied:
            logger.info(f"  本轮所有 include 论文用户都满意，退出迭代，进入全量筛选")
            break
        if len(satisfied_papers) >= SATISFIED_THRESHOLD:
            logger.info(f"  累计满意论文 {len(satisfied_papers)} ≥ {SATISFIED_THRESHOLD}，退出迭代")
            break

    logger.info(f"[阶段2 完成] 共 {screening_round} 轮迭代, satisfied={len(satisfied_papers)}, included={len(all_included)}, excluded={len(all_excluded)}, not_sure={len(all_not_sure)}")

    # ─── 阶段 2.5: 逐篇构建 executor prompt 并记录（不执行筛选）───
    remaining_unscreened = [(i, p) for i, p in enumerate(prescreened_papers) if p.get("screening_decision") == "unscreened"]
    # 如果 Stage 2 已覆盖所有论文（无 unscreened），则把所有已筛选论文当作 unscreened 来生成 prompt
    if not remaining_unscreened:
        remaining_unscreened = [(i, p) for i, p in enumerate(prescreened_papers)]
        logger.info(f"[阶段2.5] Stage 2 已覆盖所有论文，将全部 {len(remaining_unscreened)} 篇论文视为 unscreened 生成 prompt")
    else:
        logger.info(f"[阶段2.5] 构建 executor prompt: 剩余 {len(remaining_unscreened)} 篇 unscreened 论文")

    screening_context = assistant.provide_context(
        "对论文进行详细筛选，根据摘要判断是否纳入", task_id, user_id
    )

    # 获取原始研究任务和达标论文，提供更详细的用户需求
    research_task = store.read_research_task(task_id) or ""
    satisfied_papers = store.read_satisfied_papers(task_id) or []
    satisfied_text = ""
    if satisfied_papers:
        satisfied_text = "\n\n用户已确认满意的论文:\n"
        for i, p in enumerate(satisfied_papers):
            satisfied_text += f"  {i+1}. {p.get('title', '')} - {p.get('include_reason', '符合研究需求')}\n"

    # 逐篇记录：system = SCREEN_SYSTEM_PROMPT + 原始研究任务 + LLM摘要偏好 + 达标论文, user = 单篇论文信息
    system_content = (
        f"{SCREEN_SYSTEM_PROMPT}\n\n"
        f"研究任务:\n{research_task}\n\n"
        f"用户的需求和偏好:\n{screening_context}"
        f"{satisfied_text}"
    )

    output_path = os.path.join(PROJECT_ROOT, "data_csmed_prompt_export", "executor_prompt_record.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    record_count = 0
    for orig_idx, paper in remaining_unscreened:
        user_content = f"--- 论文 1 ---\n标题: {paper.get('title', '')}\n摘要: {(paper.get('abstract', '') or paper.get('text', ''))}"

        record = {
            "review_id": review_id,
            "document_id": paper.get("_id", ""),
            "system_prompt": system_content,
            "user_prompt": user_content,
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        record_count += 1

    logger.info(f"[阶段2.5] 已记录 {record_count} 条逐篇 prompt 到 {output_path}")

    return {
        "review_id": review_id,
        "record_count": record_count,
    }


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="导出阶段2.5 executor prompt")
    parser.add_argument(
        "--num-reviews", type=int, default=0,
        help="测试的 review 数量（0=全部）",
    )
    parser.add_argument("--offset", type=int, default=0, help="从第几个 review 开始（默认 0）")
    parser.add_argument("--workers", type=int, default=3, help="并发线程数（默认 3）")
    args = parser.parse_args()

    skip_prescreen = False

    test_params = {
        "num_reviews": args.num_reviews,
        "offset": args.offset,
        "skip_prescreen": skip_prescreen,
        "workers": args.workers,
    }

    logger.info(f"测试参数: {test_params}")

    # 清空旧的逐篇 prompt 记录文件
    if os.path.exists(PROMPT_RECORD_PATH):
        os.remove(PROMPT_RECORD_PATH)
        logger.info(f"已清空旧记录: {PROMPT_RECORD_PATH}")

    # 加载数据集
    reviews_metadata, papers_dict, gold_standard = load_csmed_data()

    review_ids = sorted(set(reviews_metadata.keys()) & set(gold_standard.keys()))
    logger.info(f"可测试的 review 数: {len(review_ids)}")

    start = args.offset
    end = len(review_ids) if args.num_reviews == 0 else min(start + args.num_reviews, len(review_ids))
    selected_ids = review_ids[start:end]
    logger.info(f"本次测试 {len(selected_ids)} 个 review, 并发线程数: {args.workers}")

    # 构建参数
    review_args = []
    for rid in selected_ids:
        result_dir = os.path.join(TEST_DATA_DIR, rid)
        store = InMemoryStore(result_dir)
        review_meta = reviews_metadata[rid]
        gold_paper_ids = set(gold_standard.get(rid, {}).keys())
        paper_list = build_paper_list_for_review(gold_paper_ids, papers_dict)
        gold_included = get_gold_included_ids(gold_standard, rid)
        review_args.append((rid, review_meta, paper_list, gold_included, store, skip_prescreen, gold_standard))

    # 多线程并发执行
    all_results: list[dict] = [{} for _ in range(len(review_args))]

    def _run_single(idx, rid, review_meta, paper_list, gold_included, store, skip_prescreen, gold_standard):
        try:
            result = run_single_review_export_prompt(rid, review_meta, paper_list, gold_included, store, skip_prescreen, gold_standard)
            return (idx, result)
        except Exception as e:
            logger.error(f"处理 review {rid} 时出错: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return (idx, {
                "review_id": rid,
                "executor_prompt": "",
                "error": str(e),
            })

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_single, i, *args): i
            for i, args in enumerate(review_args)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            all_results[idx] = result
            rid = selected_ids[idx]
            logger.info(f"[完成] review {rid} ({idx+1}/{len(selected_ids)})")

    # executor_prompt_record.txt 已在阶段2.5逐条写入，此处仅汇总统计
    logger.info(f"导出完成！共 {len(all_results)} 个 review")
    total_records = sum(r.get("record_count", 0) for r in all_results)
    logger.info(f"总逐篇 prompt 记录数: {total_records}")


if __name__ == "__main__":
    main()
