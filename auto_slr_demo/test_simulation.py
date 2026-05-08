"""测试模块 - 使用 LLM 模拟硕士学生与系统对话

模拟学生使用独立的 LLM (DeepSeek)，与系统 LLM (GLM) 分开。
test_start_prompt.txt 中的 prompt 作为模拟学生的 system prompt，
由模拟学生 LLM 根据人设自行生成初始 query 并与系统对话。
"""
import json
import logging
import os
from datetime import datetime

from llm_client import simulator_chat, extract_json
from coordinator import Coordinator
import memory_store as ms
from config import DATA_DIR

logger = logging.getLogger("auto_slr")


# ─── 模拟学生配置 ────────────────────────────────────────

def _load_students_from_file() -> list[dict]:
    """从 test_start_prompt.txt 加载模拟学生 prompt"""
    import re
    prompt_path = os.path.join(os.path.dirname(__file__), "test_start_prompt.txt")
    students = []
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            # 提取行首编号作为 user_id
            num_match = re.match(r'^(\d+)\.\s*', line)
            user_id = num_match.group(1) if num_match else str(i + 1)
            cleaned = re.sub(r'^\d+\.\s*', '', line)
            cleaned = cleaned.strip('"').strip()
            students.append({
                "id": f"student_{i+1:02d}",
                "user_id": user_id,
                "system_prompt": cleaned,
            })
    return students


SIMULATED_STUDENTS = _load_students_from_file()

# 生成初始 query 的 user message
GENERATE_QUERY_MESSAGE = """你正在与一个自动文献综述系统对话。请根据你的人设，用简短自然的语言陈述你的文献综述需求。
你不需要重复整个人设——只需陈述你的核心需求，就像一个真实的学生那样。
直接输出你想对系统说的话，不要加任何前缀或说明。"""


# ─── 核心测试函数 ────────────────────────────────────────

def run_single_test(student: dict, auto_mode: bool = True) -> dict:
    """对单个模拟学生运行完整测试

    Args:
        student: 模拟学生配置，含 system_prompt
        auto_mode: True=自动模拟用户回复，False=手动输入

    Returns:
        测试结果字典
    """
    task_id = f"test_{student['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(DATA_DIR, exist_ok=True)

    coordinator = Coordinator(task_id, user_id=student.get("user_id", ""))
    results = {
        "student_id": student["id"],
        "task_id": task_id,
        "system_prompt": student["system_prompt"],
        "scores": None,
    }

    print(f"\n{'='*60}")
    print(f"测试学生: {student['id']}")
    print(f"设定: {student['system_prompt'][:80]}...")
    print(f"{'='*60}")

    if auto_mode:
        run_status = _run_auto(coordinator, student, task_id)
    else:
        # 手动模式：用模拟学生 LLM 生成初始 query，之后手动交互
        initial_query = _generate_student_query(student["system_prompt"])
        print(f"\n  [模拟学生初始 query]: {initial_query}")
        coordinator.run(initial_query)
        run_status = {"search_success": True, "search_count": -1, "satisfied_count": -1, "error": None}

    # 评估结果
    scored_papers = ms.read_scored_papers(task_id)
    if scored_papers:
        top_papers = _format_top_papers(scored_papers[:10])
        eval_result = _evaluate_results(student, top_papers)
        results["scores"] = eval_result
        print(f"\n评估结果: {json.dumps(eval_result, ensure_ascii=False, indent=2)}")
    else:
        # 如果没有打分后的论文，尝试用 satisfied_papers 进行评估
        satisfied_papers = ms.read_satisfied_papers(task_id)
        if satisfied_papers:
            top_papers = _format_top_papers(satisfied_papers[:10])
            eval_result = _evaluate_results(student, top_papers)
            results["scores"] = eval_result
            print(f"\n评估结果(使用 satisfied_papers): {json.dumps(eval_result, ensure_ascii=False, indent=2)}")
        elif not run_status.get("search_success", True):
            # 搜索就失败了，记录错误评分
            results["scores"] = {
                "error": run_status.get("error", "搜索未找到论文"),
                "relevance": 1, "coverage": 1, "ranking_quality": 1, "overall_satisfaction": 1,
                "comments": f"系统搜索失败: {run_status.get('error', '未找到论文')}"
            }
            print("\n评估结果: 搜索失败，无法评估")
        else:
            results["scores"] = {
                "error": "没有可供评估的论文", "relevance": 1, "coverage": 1,
                "ranking_quality": 1, "overall_satisfaction": 1,
                "comments": "搜索成功但筛选后无可用论文"
            }
            print("\n评估结果: 无可供评估的论文")

    return results


def _generate_student_query(system_prompt: str) -> str:
    """让模拟学生 LLM 根据人设生成初始 query"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": GENERATE_QUERY_MESSAGE},
    ]
    query = simulator_chat(messages)
    return query.strip()


def _simulate_student_reply(system_prompt: str, conversation_history: list[dict]) -> str:
    """让模拟学生 LLM 根据人设和对话历史回复系统

    Args:
        system_prompt: 学生的 system prompt (来自 test_start_prompt.txt)
        conversation_history: 对话历史 [{"role": "user"/"assistant", "content": "..."}]
            其中 user 是系统说的话，assistant 是学生之前的回复
    """
    messages = [{"role": "system", "content": system_prompt}] + conversation_history
    reply = simulator_chat(messages)
    return reply.strip()


def _run_auto(coordinator: Coordinator, student: dict, task_id: str) -> dict:
    """自动模式：用独立 LLM 模拟学生对话

    Returns:
        dict with keys:
            "search_success": bool - 是否搜索到论文
            "search_count": int - 搜索到的论文数
            "satisfied_count": int - 满意的论文数
            "error": str or None - 错误信息
    """
    import memory_store as ms_inner

    assistant = coordinator.assistant
    executor = coordinator.executor
    contactor = coordinator.contactor
    user_id = student.get("user_id", "")

    status = {"search_success": False, "search_count": 0, "satisfied_count": 0, "error": None}

    student_system_prompt = student["system_prompt"]
    # 学生的对话历史（用于模拟学生 LLM 的多轮上下文）
    student_history: list[dict] = []

    # ── 阶段1: 生成初始 query 并抽取信息 ──
    initial_query = _generate_student_query(student_system_prompt)
    student_history.append({"role": "assistant", "content": initial_query})
    print(f"  [模拟学生初始 query]: {initial_query[:100]}...")
    logger.info(f"[User(Simulated)] 初始query: {initial_query}")

    assistant.extract_and_record(initial_query, task_id, user_id)
    ms_inner.append_timeline(task_id, "user_simulated", initial_query)

    # 检查信息充分性（模拟学生回答追问）
    max_rounds = 3
    for _ in range(max_rounds):
        sufficiency = executor.check_query_sufficiency(task_id, user_id)
        if sufficiency.get("sufficient", False):
            break
        missing = sufficiency.get("missing_info", "")
        msg = contactor.ask_missing_info(missing, task_id, user_id)

        # 模拟学生回复系统的追问
        student_history.append({"role": "user", "content": msg})
        reply = _simulate_student_reply(student_system_prompt, student_history)
        student_history.append({"role": "assistant", "content": reply})
        logger.info(f"[User(Simulated)] 回复追问: {reply}")

        contactor.receive_user_input(reply)
        assistant.extract_and_record(reply, task_id, user_id)
        ms_inner.append_timeline(task_id, "user_simulated", reply)
        print(f"  [模拟学生回复追问]: {reply[:80]}...")

    # ── 阶段2: 搜索 ──
    ctx = assistant.provide_context("搜索相关论文", task_id, user_id)
    papers = executor.search_papers(task_id, ctx)
    status["search_count"] = len(papers)
    print(f"  搜索到 {len(papers)} 篇论文")

    if not papers:
        status["search_success"] = False
        status["error"] = f"未找到相关论文 (搜索到 {len(papers)} 篇)"
        print("  未找到论文，跳过后续步骤。")
        return status

    status["search_success"] = True

    # ── 阶段3: 样例筛选循环 ──
    satisfied = ms_inner.read_satisfied_papers(task_id)
    max_screening_rounds = 5

    for round_num in range(max_screening_rounds):
        if len(satisfied) >= 10:
            break

        ctx = assistant.provide_context("对论文进行初步筛选", task_id, user_id)
        screening = executor.screen_papers_sample(task_id, ctx)
        included = screening["included"]

        if not included:
            break

        # 展示样例论文给模拟学生
        msg = contactor.present_sample_papers(included, task_id)
        student_history.append({"role": "user", "content": msg})

        for paper in included:
            # 让模拟学生判断这篇论文
            paper_prompt = (
                f"系统向你展示了一篇论文:\n"
                f"标题: {paper.get('title', '')}\n"
                f"年份: {paper.get('year', '')}\n"
                f"筛选理由: {paper.get('include_reason', '')}\n"
                f"摘要: {paper.get('abstract', '')[:300]}...\n\n"
                f"请判断这篇论文是否符合你的研究兴趣，并简要表达你的态度和理由。"
            )
            student_history.append({"role": "user", "content": paper_prompt})
            reply = _simulate_student_reply(student_system_prompt, student_history)
            student_history.append({"role": "assistant", "content": reply})
            logger.info(f"[User(Simulated)] 论文反馈 '{paper.get('title', '')[:50]}': {reply[:200]}")

            # 判断学生态度
            positive_words = ["yes", "relevant", "good", "interested", "satisfied", "suitable", "great", "nice", "like", "fit", "符合", "满意", "相关", "适合", "很好", "不错", "喜欢"]
            negative_words = ["no", "not", "irrelevant", "unsatisfied", "exclude", "dislike", "不符合", "不满意", "排除", "不相关", "不喜欢", "不行"]
            reply_lower = reply.lower()
            pos = sum(1 for w in positive_words if w in reply_lower)
            neg = sum(1 for w in negative_words if w in reply_lower)

            if pos > neg:
                paper["user_satisfied"] = True
                satisfied.append(paper)
                ms_inner.write_satisfied_papers(task_id, satisfied)
                assistant.extract_and_record(f"用户对论文 '{paper.get('title', '')}' 感到满意: {reply[:100]}", task_id, user_id)
                print(f"  [满意] {paper.get('title', '')[:50]}")
            else:
                paper["user_satisfied"] = False
                paper["rejection_reason"] = reply
                assistant.extract_and_record(
                    f"用户对论文 '{paper.get('title', '')}' 不满意: {reply[:100]}", task_id, user_id
                )
                coordinator._remove_rejected_paper(paper)
                print(f"  [不满意] {paper.get('title', '')[:50]}: {reply[:50]}...")

            if len(satisfied) >= 10:
                break

    # ── 阶段4: 打分 ──
    assistant.generate_scoring_criteria(task_id, user_id)
    executor.score_papers(task_id)
    status["satisfied_count"] = len(satisfied)
    print(f"  打分完成，达标论文: {len(satisfied)} 篇")

    return status


def _format_top_papers(papers: list[dict]) -> str:
    """格式化 Top 论文用于评估"""
    lines = []
    for i, p in enumerate(papers):
        lines.append(
            f"{i+1}. {p.get('title', 'N/A')} ({p.get('year', 'N/A')}) "
            f"- 总分: {p.get('total_score', 0)} - {p.get('brief_comment', '')}"
        )
    return "\n".join(lines)


def _evaluate_results(student: dict, top_papers: str) -> dict:
    """让模拟学生 LLM 对筛选结果进行评分"""
    eval_prompt = f"""你是一名具有以下人设的硕士研究生，请对这个文献综述系统的筛选结果进行评估。

你的人设：
{student["system_prompt"]}

系统的论文筛选结果（Top 10）：
{top_papers}

请根据你的研究需求，对以下维度进行评分（1-5分）：
1. **相关性**: 筛选出的论文与你研究问题的相关程度
2. **覆盖度**: 结果对你研究领域的覆盖广度
3. **排序质量**: 论文排序是否合理（最相关的论文排在前面）
4. **整体满意度**: 总体评价

请输出 JSON：
{{
  "relevance": 分数,
  "coverage": 分数,
  "ranking_quality": 分数,
  "overall_satisfaction": 分数,
  "comments": "具体评价"
}}

只输出 JSON，不要输出其他内容。"""

    result_str = simulator_chat([
        {"role": "system", "content": student["system_prompt"]},
        {"role": "user", "content": eval_prompt},
    ])
    result = extract_json(result_str)
    if result is None:
        logger.warning(f"评估结果 JSON 解析失败，原始返回:\n{result_str}")
        return {"error": "评估结果解析失败", "raw": result_str}
    return result


def run_all_tests(auto_mode: bool = True) -> list[dict]:
    """运行全部模拟学生测试

    Args:
        auto_mode: True=自动模拟, False=手动

    Returns:
        所有测试结果列表
    """
    all_results = []

    print("=" * 60)
    print("  自动文献综述系统 - 批量测试")
    print(f"  模拟学生数: {len(SIMULATED_STUDENTS)}")
    print(f"  模式: {'自动' if auto_mode else '手动'}")
    print("=" * 60)

    for student in SIMULATED_STUDENTS:
        try:
            result = run_single_test(student, auto_mode=auto_mode)
            all_results.append(result)
        except Exception as e:
            print(f"\n  学生 {student['id']} 测试出错: {e}")
            all_results.append({
                "student_id": student["id"],
                "error": str(e),
            })

    # 汇总报告
    _print_summary(all_results)

    # 保存结果
    report_path = os.path.join(DATA_DIR, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n完整测试报告已保存至: {report_path}")

    # 保存打分记录到 JSONL 文件
    _save_scores_jsonl(all_results)

    return all_results


def _save_scores_jsonl(results: list[dict]) -> None:
    """将每个学生的打分记录追加保存到 JSONL 文件"""
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    jsonl_path = os.path.join(DATA_DIR, f"satisfaction_scores_{run_time}.jsonl")
    os.makedirs(DATA_DIR, exist_ok=True)
    saved_count = 0
    skipped_count = 0
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for r in results:
            scores = r.get("scores")
            if not scores:
                skipped_count += 1
                continue
            record = {
                "student_id": r.get("student_id", ""),
                "task_id": r.get("task_id", ""),
                "relevance": scores.get("relevance"),
                "coverage": scores.get("coverage"),
                "ranking_quality": scores.get("ranking_quality"),
                "overall_satisfaction": scores.get("overall_satisfaction"),
                "comments": scores.get("comments", ""),
            }
            if scores.get("error"):
                record["error"] = scores["error"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved_count += 1
    print(f"打分记录已保存至: {jsonl_path} (写入{saved_count}条, 跳过{skipped_count}条)")


def _print_summary(results: list[dict]) -> None:
    """打印汇总报告"""
    print(f"\n{'='*60}")
    print("  测试汇总报告")
    print(f"{'='*60}")

    valid_scores = [r["scores"] for r in results if r.get("scores") and not r["scores"].get("error")]

    if not valid_scores:
        print("  无有效评分数据。")
        return

    for dim in ["relevance", "coverage", "ranking_quality", "overall_satisfaction"]:
        scores = [s.get(dim, 0) for s in valid_scores if s.get(dim)]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {dim}: 平均 {avg:.2f}/5 (n={len(scores)})")

    overall = [s.get("overall_satisfaction", 0) for s in valid_scores if s.get("overall_satisfaction")]
    if overall:
        print(f"\n  整体满意度均值: {sum(overall)/len(overall):.2f}/5")


if __name__ == "__main__":
    import sys

    from logging_config import setup_logging

    # 日志级别可通过第二个参数指定，如 --single 0 DEBUG
    log_level = "INFO"
    if len(sys.argv) > 1 and sys.argv[-1] in ("DEBUG", "INFO", "WARNING"):
        log_level = sys.argv.pop()

    setup_logging(log_level)

    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        run_all_tests(auto_mode=False)
    elif len(sys.argv) > 1 and sys.argv[1] == "--single":
        idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        run_single_test(SIMULATED_STUDENTS[idx], auto_mode=True)
    else:
        run_all_tests(auto_mode=True)
