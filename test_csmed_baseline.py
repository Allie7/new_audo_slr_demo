"""CSMeD-FT Baseline 测试脚本

不走 assistant/coordinator/executor 工作流，直接用 LLM 做 single-pass 筛选：
1. 将 review metadata（title, abstract, review_type, criteria, criteria_text, search_strategy）
   整理为一条自然语言 query
2. 把该 query + 全量论文（title + metadata + abstract）直接交给 LLM 做批量筛选
3. 统计筛选结果与 gold standard 的对比
4. 生成与 test_csmed.py 相同格式的报告
"""

import json
import logging
import os
import sys
from datetime import datetime

# ── 路径设置 ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 测试数据输出目录
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "data_csmed_test_baseline")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# CSMeD-FT 数据集路径
CSMED_DATA_DIR = r"D:\School\SAT404\pilot_study_code\systematic-review-datasets\data\CSMeD\CSMeD-FT\CSMeD-FT"

# ── 日志配置 ──────────────────────────────────────────────
log_filename = f"test_csmed_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(TEST_DATA_DIR, log_filename), encoding="utf-8"),
    ],
)
logger = logging.getLogger("test_csmed_baseline")

# ── 导入项目模块 ──────────────────────────────────────────
from llm_client import chat_with_system, extract_json

# ── Baseline 筛选配置 ────────────────────────────────────
BASELINE_BATCH_SIZE = 10  # 每批筛选论文数

BASELINE_SYSTEM_PROMPT = """你是一名学术文献筛选专家。你将收到：
1. 一条综合性的研究需求描述（包含研究背景、类型、纳入/排除标准等）
2. 多篇论文的标题、元数据和摘要

请根据研究需求描述，对每篇论文判断是否应纳入该系统综述，输出一个 JSON 数组：
[
  {{
    "index": 论文编号,
    "decision": "include" | "exclude",
    "reason": "判断理由"
  }},
  ...
]

判断标准：
- include: 论文与研究方向相关，且满足研究需求中描述的纳入标准
- exclude: 论文不相关，或不满足纳入标准，或符合排除标准

注意：
- 仅根据提供的信息做判断，不要做额外假设
- 如果信息不足以确定，倾向于 exclude

只输出 JSON 数组，不要输出其他内容。"""


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
    """获取某个 review 在 gold standard 中判定为 included 的 document_id 集合"""
    review_gold = gold_standard.get(review_id, {})
    return {did for did, info in review_gold.items()
            if (info if isinstance(info, dict) else {"decision": info}).get("decision") == "included"}


def get_gold_paper_ids(gold_standard: dict, review_id: str) -> set[str]:
    """获取某个 review 在 gold standard 中涉及的所有 document_id"""
    return set(gold_standard.get(review_id, {}).keys())


def build_paper_list_for_review(paper_ids: set[str], papers_dict: dict) -> list[dict]:
    """将 document_id 集合转为系统内部 paper_list 格式"""
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
# Metadata 初筛（与 test_csmed.py 一致）
# ═══════════════════════════════════════════════════════════

def metadata_prescreen(all_papers: list[dict], year_range: str | None = None) -> list[dict]:
    """基于论文元数据的质量过滤"""
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
                if not (start_yr <= int(float(p["year"])) <= end_yr):
                    filtered_out += 1
                    continue
            except (ValueError, IndexError, TypeError):
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
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "num_relevant": 0, "num_retrieved": len(retrieved_ids), "num_true_positives": 0}
    if not retrieved_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "num_relevant": len(relevant_ids), "num_retrieved": 0, "num_true_positives": 0}

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
# Baseline 核心：构建自然语言 query
# ═══════════════════════════════════════════════════════════

def build_baseline_query(review_meta: dict) -> str:
    """将 review metadata（除 review_id 和 doi 外）整理为一条自然语言 query

    包含字段：title, abstract, review_type, criteria, criteria_text, search_strategy
    """
    parts = []

    title = review_meta.get("title", "")
    if title:
        parts.append(f"研究标题: {title}")

    review_type = review_meta.get("review_type", "")
    if review_type:
        parts.append(f"研究类型: {review_type}")

    abstract = review_meta.get("abstract", "")
    if abstract:
        parts.append(f"研究背景与目的:\n{abstract}")

    criteria_text = review_meta.get("criteria_text", "")
    if criteria_text:
        parts.append(f"纳入/排除标准:\n{criteria_text}")

    criteria = review_meta.get("criteria", "")
    if criteria:
        if isinstance(criteria, dict):
            criteria_str = json.dumps(criteria, ensure_ascii=False, indent=2)
        else:
            criteria_str = str(criteria)
        parts.append(f"结构化纳入/排除标准:\n{criteria_str}")

    search_strategy = review_meta.get("search_strategy", "")
    if search_strategy:
        if isinstance(search_strategy, dict):
            ss_str = json.dumps(search_strategy, ensure_ascii=False, indent=2)
        else:
            ss_str = str(search_strategy)
        parts.append(f"搜索策略:\n{ss_str}")

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════
# Baseline 核心：单 review 筛选
# ═══════════════════════════════════════════════════════════

def run_baseline_single_review(
    review_id: str,
    review_meta: dict,
    paper_list: list[dict],
    gold_included_ids: set[str],
    gold_standard: dict,
) -> dict:
    """Baseline 测试：直接用 LLM 做单次筛选，不经过 workflow

    流程：
    1. Metadata 初筛（排除无摘要/无标题）
    2. 构建 baseline query
    3. 对初筛后的论文分批调用 LLM 做单次 include/exclude 判断
    4. 统计结果与 gold standard 对比
    """
    all_papers_count = len(paper_list)
    logger.info(f"[Baseline] 开始处理 review {review_id}, 共 {all_papers_count} 篇论文")

    # ─── 阶段 1: Metadata 初筛 ───
    prescreened_papers = metadata_prescreen(paper_list)
    prescreen_kept_ids = {p["_id"] for p in prescreened_papers}
    prescreen_metrics = compute_metrics(prescreen_kept_ids, gold_included_ids)
    logger.info(f"[阶段1评估] Metadata初筛: {prescreen_metrics}")

    if not prescreened_papers:
        return {
            "review_id": review_id,
            "review_title": review_meta.get("title", "")[:200],
            "review_type": review_meta.get("review_type", ""),
            "num_papers": all_papers_count,
            "num_gold_included": len(gold_included_ids),
            "prescreen_metrics": prescreen_metrics,
            "baseline_metrics": None,
            "gold_comparison": None,
        }

    # ─── 阶段 2: 构建 query + 单次 LLM 筛选 ───
    query = build_baseline_query(review_meta)
    logger.info(f"[Baseline] Query 已构建: {query[:200]}...")

    included_papers = []
    excluded_papers = []

    for batch_start in range(0, len(prescreened_papers), BASELINE_BATCH_SIZE):
        batch = prescreened_papers[batch_start:batch_start + BASELINE_BATCH_SIZE]

        papers_text = ""
        for j, paper in enumerate(batch):
            papers_text += f"\n--- 论文 {j+1} ---\n"
            papers_text += f"标题: {paper.get('title', '')}\n"
            papers_text += f"年份: {paper.get('year', '')}\n"
            papers_text += f"期刊: {paper.get('journal', '')}\n"
            papers_text += f"作者: {paper.get('authors', '')}\n"
            papers_text += f"摘要: {(paper.get('abstract', '') or paper.get('text', ''))[:500]}\n"

        result_str = chat_with_system(
            BASELINE_SYSTEM_PROMPT,
            f"研究需求:\n{query}\n\n论文列表:{papers_text}",
        )
        result = extract_json(result_str)

        if result is None:
            logger.warning(f"  Baseline 筛选 JSON 解析失败，本批全部标记为 exclude")
            for paper in batch:
                paper["baseline_decision"] = "exclude"
                paper["baseline_reason"] = "解析失败"
                excluded_papers.append(paper)
            continue

        if isinstance(result, dict):
            result = [result]

        decisions = {}
        for item in result:
            idx = item.get("index", 0) - 1
            if 0 <= idx < len(batch):
                decisions[idx] = item

        for j, paper in enumerate(batch):
            item = decisions.get(j, {"decision": "exclude", "reason": "未返回判断结果"})
            decision = item.get("decision", "exclude")
            reason = item.get("reason", "")

            paper["baseline_decision"] = decision
            paper["baseline_reason"] = reason

            if decision == "include":
                included_papers.append(paper)
            else:
                excluded_papers.append(paper)

    logger.info(f"[Baseline] 筛选完成: include={len(included_papers)}, exclude={len(excluded_papers)}")

    # ─── 评估 ───
    baseline_included_ids = {p["_id"] for p in included_papers}
    baseline_metrics = compute_metrics(baseline_included_ids, gold_included_ids)
    logger.info(f"[Baseline评估] 直接筛选: {baseline_metrics}")

    # ─── Gold 对比 ───
    gold_paper_ids = get_gold_paper_ids(gold_standard, review_id)
    system_included_ids = baseline_included_ids
    gold_comparison = compute_metrics(system_included_ids, gold_included_ids)

    correct_ids = system_included_ids & gold_included_ids
    missed_ids = gold_included_ids - system_included_ids
    false_positive_ids = system_included_ids - gold_included_ids

    gold_comparison_detail = {
        "num_correct": len(correct_ids),
        "num_missed": len(missed_ids),
        "num_false_positive": len(false_positive_ids),
        "correct_ids": sorted(correct_ids),
        "missed_ids": sorted(missed_ids),
        "false_positive_ids": sorted(false_positive_ids),
    }

    logger.info(
        f"[Gold对比] review {review_id}: "
        f"P={gold_comparison['precision']}, R={gold_comparison['recall']}, "
        f"F1={gold_comparison['f1']}, "
        f"TP={len(correct_ids)}, FN={len(missed_ids)}, FP={len(false_positive_ids)}"
    )

    return {
        "review_id": review_id,
        "review_title": review_meta.get("title", "")[:200],
        "review_type": review_meta.get("review_type", ""),
        "num_papers": all_papers_count,
        "num_prescreened": len(prescreened_papers),
        "num_baseline_included": len(included_papers),
        "num_gold_included": len(gold_included_ids),
        "prescreen_metrics": prescreen_metrics,
        "baseline_metrics": baseline_metrics,
        "gold_comparison": gold_comparison,
        "gold_comparison_detail": gold_comparison_detail,
    }


# ═══════════════════════════════════════════════════════════
# 报告生成（与 test_csmed.py 相同格式）
# ═══════════════════════════════════════════════════════════

def generate_report(all_results: list[dict], test_params: dict) -> str:
    """生成 Baseline 测试报告（与 test_csmed.py 相同格式）"""
    lines = []
    lines.append("=" * 80)
    lines.append("CSMeD-FT Baseline 测试报告")
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
        lines.append(f"    Baseline include: {r.get('num_baseline_included', 0)}")
        pm = r.get("prescreen_metrics")
        lines.append(f"    Metadata初筛: {pm}")
        bm = r.get("baseline_metrics")
        lines.append(f"    Baseline筛选: {bm}")
        gc = r.get("gold_comparison")
        if gc:
            lines.append(f"    *** Gold对比: P={gc.get('precision',0)}, R={gc.get('recall',0)}, F1={gc.get('f1',0)} ***")
            gd = r.get("gold_comparison_detail", {})
            lines.append(f"        TP={gd.get('num_correct',0)}, FN={gd.get('num_missed',0)}, FP={gd.get('num_false_positive',0)}")

    # ─── 汇总统计 ───
    lines.append(f"\n{'='*80}")
    lines.append("汇总统计")
    lines.append(f"{'='*80}")

    stages = ["prescreen_metrics", "baseline_metrics", "gold_comparison"]
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
    lines.append(f"  Baseline筛选 (平均): {avg_metrics['baseline_metrics']}")
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
    lines.append(f"  总 TP (Baseline include & Gold include): {total_tp}")
    lines.append(f"  总 FP (Baseline include & Gold exclude): {total_fp}")
    lines.append(f"  总 FN (Baseline exclude & Gold include): {total_fn}")

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
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="CSMeD-FT Baseline 测试脚本")
    parser.add_argument(
        "--num-reviews", type=int, default=0,
        help="测试的 review 数量（0=全部）",
    )
    parser.add_argument("--offset", type=int, default=0, help="从第几个 review 开始（默认 0）")
    parser.add_argument("--workers", type=int, default=3, help="并发线程数（默认 3）")
    parser.add_argument("--skip-prescreen", action="store_true", help="跳过 metadata 初筛")
    args = parser.parse_args()

    test_params = {
        "num_reviews": args.num_reviews,
        "offset": args.offset,
        "skip_prescreen": args.skip_prescreen,
        "workers": args.workers,
        "mode": "baseline",
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
    logger.info(f"本次测试 {len(selected_ids)} 个 review, 并发线程数: {args.workers}")

    # 构建每个 review 的参数
    review_args = []
    for rid in selected_ids:
        review_meta = reviews_metadata[rid]
        paper_list = build_paper_list_for_review(set(papers_dict.keys()), papers_dict)
        gold_included = get_gold_included_ids(gold_standard, rid)
        review_args.append((rid, review_meta, paper_list, gold_included, gold_standard))

    # 多线程并发执行
    all_results: list[dict] = [{} for _ in range(len(review_args))]

    def _run_single(idx, rid, review_meta, paper_list, gold_included, gold_standard):
        try:
            result = run_baseline_single_review(rid, review_meta, paper_list, gold_included, gold_standard)
            return (idx, result)
        except Exception as e:
            logger.error(f"处理 review {rid} 时出错: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return (idx, {
                "review_id": rid,
                "review_title": review_meta.get("title", "")[:200],
                "error": str(e),
            })

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_single, i, *args_tuple): i
            for i, args_tuple in enumerate(review_args)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            all_results[idx] = result
            rid = selected_ids[idx]
            logger.info(f"[完成] review {rid} ({idx+1}/{len(selected_ids)})")

    # ─── 生成报告 ───
    report = generate_report(all_results, test_params)
    logger.info(f"\n{report}")

    # 保存报告
    report_path = os.path.join(TEST_DATA_DIR, f"report_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
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
    summary_path = os.path.join(TEST_DATA_DIR, f"summary_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"详细结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
