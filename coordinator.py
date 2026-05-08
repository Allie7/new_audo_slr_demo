"""Coordinator - 流程控制

按照以下流程协调各 agent 工作：
1. 用户 query -> assistant 抽取信息 -> executor 判断信息是否充分
2. 信息充分 -> executor 搜索论文 -> assistant 记录 -> executor 初步筛选
3. 展示样例 -> 用户反馈 -> 循环直到满意论文达标
4. assistant 生成打分标准 -> executor 对全量论文打分排序
"""
import json
import logging

import memory_store as ms
from assistant import Assistant
from executor import Executor
from contactor import Contactor
from config import MAX_SATISFIED_PAPERS

logger = logging.getLogger("auto_slr")


class Coordinator:
    """流程控制器：协调各 agent 按流程工作"""

    def __init__(self, task_id: str, user_id: str = ""):
        self.task_id = task_id
        self.user_id = user_id
        self.assistant = Assistant()
        self.executor = Executor()
        self.contactor = Contactor()

    def run(self, user_query: str) -> None:
        """运行完整的文献综述流程"""
        logger.info(f"========== 开始处理任务 {self.task_id} ==========")
        logger.info(f"[User] 输入: {user_query}")
        print(f"\n{'='*60}")
        print(f"开始处理任务 {self.task_id}")
        print(f"{'='*60}")

        # ─── 阶段 1: 信息抽取与充分性检查 ───
        logger.info("[阶段1] 信息抽取与充分性检查")
        print("\n[阶段1] 信息抽取与充分性检查")
        self.assistant.extract_and_record(user_query, self.task_id, self.user_id)
        ms.append_timeline(self.task_id, "coordinator", f"收到用户查询: {user_query[:80]}...")

        # 循环直到信息充分
        while True:
            sufficiency = self.executor.check_query_sufficiency(self.task_id, self.user_id)
            logger.info(f"充分性检查结果: {sufficiency}")
            if sufficiency.get("sufficient", False):
                logger.info("信息充分，可以开始搜索")
                print("  信息充分，可以开始搜索。")
                break

            missing = sufficiency.get("missing_info", "")
            logger.info(f"信息不充分，缺少: {missing}")
            print(f"  信息不充分，缺少: {missing}")
            msg = self.contactor.ask_missing_info(missing, self.task_id, self.user_id)
            print(f"\n  [Contactor]: {msg}")

            user_input = input("\n  [你的回答]: ")
            logger.info(f"[User] 回复追问: {user_input}")
            self.contactor.receive_user_input(user_input)
            self.assistant.extract_and_record(user_input, self.task_id, self.user_id)
            ms.append_timeline(self.task_id, "user", user_input)

        # ─── 阶段 2: 论文搜索 ───
        logger.info("[阶段2] 论文搜索")
        print("\n[阶段2] 论文搜索")
        assistant_context = self.assistant.provide_context("搜索相关论文", self.task_id, self.user_id)
        papers = self.executor.search_papers(self.task_id, assistant_context)
        logger.info(f"搜索完成，找到 {len(papers)} 篇论文")
        print(f"  搜索完成，找到 {len(papers)} 篇论文。")

        if not papers:
            print("  未找到任何论文，请调整搜索条件。")
            return

        # ─── 阶段 3: 样例筛选与用户确认循环 ───
        logger.info("[阶段3] 样例筛选与用户确认")
        print("\n[阶段3] 样例筛选与用户确认")
        satisfied_papers = ms.read_satisfied_papers(self.task_id)

        while len(satisfied_papers) < MAX_SATISFIED_PAPERS:
            # executor 初步筛选样例
            assistant_context = self.assistant.provide_context(
                "对论文进行初步筛选，根据摘要判断是否纳入", self.task_id, self.user_id
            )
            screening_result = self.executor.screen_papers_sample(self.task_id, assistant_context)
            included = screening_result["included"]

            if not included:
                print("  未找到更多符合要求的论文。")
                break

            # 展示样例论文给用户
            msg = self.contactor.present_sample_papers(included, self.task_id)
            print(f"\n  [Contactor]: {msg}")

            # 逐一确认每篇论文
            for paper in included:
                print(f"\n  论文: {paper.get('title', 'N/A')}")
                print(f"  年份: {paper.get('year', 'N/A')}")
                print(f"  筛选理由: {paper.get('include_reason', 'N/A')}")
                user_verdict = input("  这篇论文是否符合心意？(y/n/不确定): ").strip().lower()
                logger.info(f"[User] 论文判定 '{paper.get('title', '')}': {user_verdict}")

                if user_verdict in ("y", "yes", "是", "符合"):
                    satisfied_papers.append(paper)
                    ms.write_satisfied_papers(self.task_id, satisfied_papers)
                    self.assistant.extract_and_record(
                        f"用户对论文 '{paper.get('title', '')}' 感到满意。", self.task_id, self.user_id
                    )
                    print(f"  已记录！当前达标论文: {len(satisfied_papers)}/{MAX_SATISFIED_PAPERS}")
                elif user_verdict in ("n", "no", "否", "不符合"):
                    reason = input("  请说明不符合的原因: ").strip()
                    paper["user_satisfied"] = False
                    paper["rejection_reason"] = reason
                    self.assistant.extract_and_record(
                        f"用户对论文 '{paper.get('title', '')}' 不满意，原因: {reason}", self.task_id, self.user_id
                    )
                    # 从论文列表中移除被拒绝的论文，以便下一轮筛选新论文
                    self._remove_rejected_paper(paper)
                else:
                    paper["user_satisfied"] = "not_sure"
                    reason = input("  请说明你的疑虑: ").strip() if input("  是否有补充意见？(y/n): ").strip().lower() in ("y", "yes") else ""
                    if reason:
                        self.assistant.extract_and_record(
                            f"用户对论文 '{paper.get('title', '')}' 有疑虑: {reason}", self.task_id, self.user_id
                        )

                if len(satisfied_papers) >= MAX_SATISFIED_PAPERS:
                    break

            # 检查是否还有未筛选的论文
            remaining = self._count_unscreened_papers()
            if remaining == 0:
                print("  所有论文已筛选完毕。")
                break

        print(f"\n  样例确认阶段完成，共 {len(satisfied_papers)} 篇达标论文。")
        logger.info(f"样例确认阶段完成，达标论文: {len(satisfied_papers)} 篇")

        # ─── 阶段 4: 生成打分标准并打分 ───
        logger.info("[阶段4] 生成打分标准并对全量论文打分排序")
        print("\n[阶段4] 生成打分标准并对全量论文打分排序")
        criteria = self.assistant.generate_scoring_criteria(self.task_id, self.user_id)
        print(f"  打分标准已生成:\n{criteria[:200]}...")

        scored_papers = self.executor.score_papers(self.task_id)
        logger.info(f"打分完成，共 {len(scored_papers)} 篇论文已排序")
        print(f"  打分完成！共 {len(scored_papers)} 篇论文已排序。")

        # 输出 Top 10
        print(f"\n{'='*60}")
        print(f"最终结果 - Top {min(10, len(scored_papers))} 论文")
        print(f"{'='*60}")
        for i, paper in enumerate(scored_papers[:10]):
            print(f"\n  [{i+1}] {paper.get('title', 'N/A')}")
            print(f"      年份: {paper.get('year', 'N/A')} | 总分: {paper.get('total_score', 0)}")
            print(f"      评语: {paper.get('brief_comment', '')}")

        logger.info(f"任务 {self.task_id} 完成，结果已保存")
        print(f"\n完整结果已保存至 data/scored_papers_{self.task_id}.json")

    def _remove_rejected_paper(self, rejected_paper: dict) -> None:
        """从论文列表中移除被拒绝的论文"""
        papers = ms.read_paper_list(self.task_id)
        rejected_title = rejected_paper.get("title", "").lower().strip()
        papers = [p for p in papers if p.get("title", "").lower().strip() != rejected_title]
        ms.write_paper_list(self.task_id, papers)

    def _count_unscreened_papers(self) -> int:
        """统计未筛选的论文数"""
        papers = ms.read_paper_list(self.task_id)
        return len([p for p in papers if "screening_decision" not in p])
