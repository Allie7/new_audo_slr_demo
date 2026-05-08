"""Test module - Simulate master's students conversing with the system using LLM

Simulated students use a separate LLM (DeepSeek), independent from the system LLM (GLM).
The prompts in test_start_prompt_en.txt serve as the simulated students' system prompts.
The simulated student LLM generates the initial query based on its persona and converses with the system.
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


# ─── Simulated Student Configuration ────────────────────────────────

def _load_students_from_file() -> list[dict]:
    """Load simulated student prompts from test_start_prompt_en.txt"""
    import re
    prompt_path = os.path.join(os.path.dirname(__file__), "test_start_prompt_en.txt")
    students = []
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            # Extract the leading number as user_id
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

# User message for generating the initial query
GENERATE_QUERY_MESSAGE = """你正在与一个自动文献综述系统对话。请根据你的人设，用简短自然的语言陈述你的文献综述需求。
你不需要重复整个人设——只需陈述你的核心需求，就像一个真实的学生那样。
直接输出你想对系统说的话，不要加任何前缀或说明。"""


# ─── Core Test Functions ────────────────────────────────────────

def run_single_test(student: dict, auto_mode: bool = True) -> dict:
    """Run a complete test for a single simulated student

    Args:
        student: Simulated student configuration, containing system_prompt
        auto_mode: True=automatically simulate user replies, False=manual input

    Returns:
        Test result dictionary
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
    print(f"Testing student: {student['id']}")
    print(f"Profile: {student['system_prompt'][:80]}...")
    print(f"{'='*60}")

    if auto_mode:
        _run_auto(coordinator, student, task_id)
    else:
        # Manual mode: use simulated student LLM to generate the initial query, then interact manually
        initial_query = _generate_student_query(student["system_prompt"])
        print(f"\n  [Simulated student initial query]: {initial_query}")
        coordinator.run(initial_query)

    # Evaluate results
    scored_papers = ms.read_scored_papers(task_id)
    if scored_papers:
        top_papers = _format_top_papers(scored_papers[:10])
        eval_result = _evaluate_results(student, top_papers)
        results["scores"] = eval_result
        print(f"\nEvaluation result: {json.dumps(eval_result, ensure_ascii=False, indent=2)}")

    return results


def _generate_student_query(system_prompt: str) -> str:
    """Have the simulated student LLM generate an initial query based on its persona"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": GENERATE_QUERY_MESSAGE},
    ]
    query = simulator_chat(messages)
    return query.strip()


def _simulate_student_reply(system_prompt: str, conversation_history: list[dict]) -> str:
    """Have the simulated student LLM reply to the system based on persona and conversation history

    Args:
        system_prompt: Student's system prompt (from test_start_prompt_en.txt)
        conversation_history: Conversation history [{"role": "user"/"assistant", "content": "..."}]
            where user is what the system said, and assistant is the student's previous replies
    """
    messages = [{"role": "system", "content": system_prompt}] + conversation_history
    reply = simulator_chat(messages)
    return reply.strip()


def _run_auto(coordinator: Coordinator, student: dict, task_id: str) -> None:
    """Auto mode: Use an independent LLM to simulate student conversation"""
    import memory_store as ms_inner

    assistant = coordinator.assistant
    executor = coordinator.executor
    contactor = coordinator.contactor
    user_id = student.get("user_id", "")

    student_system_prompt = student["system_prompt"]
    # Student's conversation history (for multi-turn context of the simulated student LLM)
    student_history: list[dict] = []

    # ── Phase 1: Generate initial query and extract information ──
    initial_query = _generate_student_query(student_system_prompt)
    student_history.append({"role": "assistant", "content": initial_query})
    print(f"  [Simulated student initial query]: {initial_query[:100]}...")
    logger.info(f"[User(Simulated)] Initial query: {initial_query}")

    assistant.extract_and_record(initial_query, task_id, user_id)
    ms_inner.append_timeline(task_id, "user_simulated", initial_query)

    # Check information sufficiency (simulate student answering follow-up questions)
    max_rounds = 3
    for _ in range(max_rounds):
        sufficiency = executor.check_query_sufficiency(task_id, user_id)
        if sufficiency.get("sufficient", False):
            break
        missing = sufficiency.get("missing_info", "")
        msg = contactor.ask_missing_info(missing, task_id, user_id)

        # Simulate student replying to the system's follow-up question
        student_history.append({"role": "user", "content": msg})
        reply = _simulate_student_reply(student_system_prompt, student_history)
        student_history.append({"role": "assistant", "content": reply})
        logger.info(f"[User(Simulated)] Reply to follow-up: {reply}")

        contactor.receive_user_input(reply)
        assistant.extract_and_record(reply, task_id, user_id)
        ms_inner.append_timeline(task_id, "user_simulated", reply)
        print(f"  [Simulated student reply to follow-up]: {reply[:80]}...")

    # ── Phase 2: Search ──
    ctx = assistant.provide_context("搜索相关论文", task_id, user_id)
    papers = executor.search_papers(task_id, ctx)
    print(f"  Found {len(papers)} papers")

    if not papers:
        print("  No papers found, skipping.")
        return

    # ── Phase 3: Sample screening loop ──
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

        # Present sample papers to the simulated student
        msg = contactor.present_sample_papers(included, task_id)
        student_history.append({"role": "user", "content": msg})

        for paper in included:
            # Have the simulated student judge this paper
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
            logger.info(f"[User(Simulated)] Paper feedback '{paper.get('title', '')[:50]}': {reply[:200]}")

            # Determine student's attitude
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
                print(f"  [Satisfied] {paper.get('title', '')[:50]}")
            else:
                paper["user_satisfied"] = False
                paper["rejection_reason"] = reply
                assistant.extract_and_record(
                    f"用户对论文 '{paper.get('title', '')}' 不满意: {reply[:100]}", task_id, user_id
                )
                coordinator._remove_rejected_paper(paper)
                print(f"  [Not satisfied] {paper.get('title', '')[:50]}: {reply[:50]}...")

            if len(satisfied) >= 10:
                break

    # ── Phase 4: Scoring ──
    assistant.generate_scoring_criteria(task_id, user_id)
    executor.score_papers(task_id)
    print(f"  Scoring complete, qualified papers: {len(satisfied)}")


def _format_top_papers(papers: list[dict]) -> str:
    """Format top papers for evaluation"""
    lines = []
    for i, p in enumerate(papers):
        lines.append(
            f"{i+1}. {p.get('title', 'N/A')} ({p.get('year', 'N/A')}) "
            f"- Total score: {p.get('total_score', 0)} - {p.get('brief_comment', '')}"
        )
    return "\n".join(lines)


def _evaluate_results(student: dict, top_papers: str) -> dict:
    """Have the simulated student LLM score the screening results"""
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
        logger.warning(f"Evaluation result JSON parsing failed, raw output:\n{result_str}")
        return {"error": "评估结果解析失败", "raw": result_str}
    return result


def run_all_tests(auto_mode: bool = True) -> list[dict]:
    """Run all simulated student tests

    Args:
        auto_mode: True=automatic simulation, False=manual

    Returns:
        List of all test results
    """
    all_results = []

    print("=" * 60)
    print("  Automatic Literature Review System - Batch Testing")
    print(f"  Number of simulated students: {len(SIMULATED_STUDENTS)}")
    print(f"  Mode: {'Auto' if auto_mode else 'Manual'}")
    print("=" * 60)

    for student in SIMULATED_STUDENTS:
        try:
            result = run_single_test(student, auto_mode=auto_mode)
            all_results.append(result)
        except Exception as e:
            print(f"\n  Student {student['id']} test error: {e}")
            all_results.append({
                "student_id": student["id"],
                "error": str(e),
            })

    # Summary report
    _print_summary(all_results)

    # Save results
    report_path = os.path.join(DATA_DIR, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nFull test report saved to: {report_path}")

    # Save score records to JSONL file
    _save_scores_jsonl(all_results)

    return all_results


def _save_scores_jsonl(results: list[dict]) -> None:
    """Save each student's score record to a JSONL file"""
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
    print(f"Score records saved to: {jsonl_path} (wrote {saved_count} records, skipped {skipped_count})")


def _print_summary(results: list[dict]) -> None:
    """Print summary report"""
    print(f"\n{'='*60}")
    print("  Test Summary Report")
    print(f"{'='*60}")

    valid_scores = [r["scores"] for r in results if r.get("scores") and not r["scores"].get("error")]

    if not valid_scores:
        print("  No valid score data.")
        return

    for dim in ["relevance", "coverage", "ranking_quality", "overall_satisfaction"]:
        scores = [s.get(dim, 0) for s in valid_scores if s.get(dim)]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {dim}: Average {avg:.2f}/5 (n={len(scores)})")

    overall = [s.get("overall_satisfaction", 0) for s in valid_scores if s.get("overall_satisfaction")]
    if overall:
        print(f"\n  Overall satisfaction mean: {sum(overall)/len(overall):.2f}/5")


if __name__ == "__main__":
    import sys

    from logging_config import setup_logging

    # Log level can be specified via the second argument, e.g. --single 0 DEBUG
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
