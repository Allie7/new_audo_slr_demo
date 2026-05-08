"""Executor Agent - 搜索与筛选执行

职责：
1. 使用 Semantic Scholar API 和/或 arXiv 搜索论文
2. 根据用户要求对论文进行初步筛选（include/exclude/not sure）
3. 根据打分标准对全量论文列表进行打分排序
"""
import logging

import memory_store as ms
from llm_client import chat_with_system, extract_json
from config import MAX_PAPERS_TO_SEARCH, SAMPLE_INCLUDE_THRESHOLD, SEMANTIC_SCHOLAR_API_KEY, HTTP_PROXY, HTTPS_PROXY

logger = logging.getLogger("auto_slr")

SEARCH_SYSTEM_PROMPT = """你是一名学术文献搜索专家。根据给定的研究课题和要求，你需要为 Semantic Scholar API 和 arXiv 生成搜索查询关键词。

请输出一个 JSON 对象：
{{
  "semantic_scholar_query": "搜索关键词",
  "arxiv_query": "搜索关键词",
  "year_range": "起始年份-结束年份 或 空字符串",
  "max_results": 数字
}}

重要：两个 API 的查询语法不同，必须分别适配：

【Semantic Scholar 查询语法】
- 支持布尔运算符 AND / OR / NOT
- 支持引号精确匹配，如 "breast cancer"
- 示例："deep learning" AND "medical imaging" AND "breast cancer"

【arXiv 查询语法 — 务必严格遵守】
- arXiv 使用 Lucene 风格语法，支持字段前缀：ti:(标题)、abs:(摘要)、au:(作者)、cat:(分类)
- **禁止使用 all: 前缀**：all: 要求所有字段精确匹配完整短语，过于严格，会导致医学/临床类论文搜不到
- **推荐使用 abs: 前缀**：在摘要中搜索，如 abs:"deep learning" AND abs:"breast cancer"
- **避免 AND 条件过多**：最多 3 个 AND 子句。医学/临床类论文在 arXiv 上数量较少，条件过多会返回 0 结果
- 如果查询医学/临床/生物类主题，arXiv 查询要更宽松：用 2-3 个核心关键词即可，如 abs:"medical imaging" AND abs:"breast cancer"
- 示例（好）：abs:"medical imaging" AND abs:"deep learning"
- 示例（坏）：all:"deep learning" AND all:"medical imaging" AND all:"breast cancer" AND all:"early detection" AND all:"sensitivity"
- 对于医学/临床类主题（如 cancer, NSCLC, clinical trial, microbiota 等），arXiv 不是主要收录源，查询应尽量简短宽松

只输出 JSON，不要输出其他内容。"""

SCREEN_SYSTEM_PROMPT = """你是一名学术文献筛选专家。你将收到：
1. 用户的研究需求和偏好
2. 多篇论文的元数据和摘要

请对每篇论文判断是否应纳入文献综述，输出一个 JSON 数组：
[
  {{
    "index": 论文编号,
    "decision": "include" | "exclude" | "not_sure",
    "reason": "判断理由"
  }},
  ...
]

判断标准：
- include: 论文与研究方向高度相关，且满足用户的所有筛选要求
- exclude: 论文明显不相关，或明显不符合筛选要求
- not_sure: 仅凭摘要无法确定是否应纳入

只输出 JSON 数组，不要输出其他内容。"""

SCREEN_BATCH_SIZE = 10  # 每批筛选的论文数量

SCORE_SYSTEM_PROMPT = """你是一名学术文献评审专家。你将收到：
1. 一份固定的多维打分标准（JSON 格式，每个维度 1-5 分）
2. 多篇论文的元数据和摘要

请严格按照打分标准中的维度名称和评分标准对每篇论文打分。
每个维度的分数必须是 1 到 5 的整数。
总分 = 所有维度分数之和。

输出一个 JSON 数组：
[
  {{
    "index": 论文编号,
    "scores": {{
      "维度名称1": 分数(1-5),
      "维度名称2": 分数(1-5)
    }},
    "total_score": 所有维度分数之和,
    "brief_comment": "简要评语"
  }},
  ...
]

只输出 JSON 数组，不要输出其他内容。"""

SCORE_BATCH_SIZE = 5  # 每批打分的论文数量

CHECK_QUERY_SUFFICIENCY_PROMPT = """你是一名学术文献搜索评估专家。请判断用户提供的查询信息是否足够进行初步论文搜索。

至少应包含：
1. 明确的研究方向/主题
2. 可推断的时间范围（如果未明确指定，默认近5年可以接受）

用户信息：
{user_info}

研究任务：
{research_task}

请判断信息是否充分，输出 JSON：
{{
  "sufficient": true/false,
  "missing_info": "缺失信息的描述（如果充分则为空字符串）"
}}

只输出 JSON，不要输出其他内容。"""


def _search_semantic_scholar(query: str, year_range: str | None = None, limit: int = 50) -> list[dict]:
    """使用 Semantic Scholar API 搜索论文
    
    若 API Key 失效（403），自动回退匿名访问。
    """
    from semanticscholar import SemanticScholar

    # 代理配置
    proxy = HTTPS_PROXY or HTTP_PROXY
    timeout = 30
    api_key = SEMANTIC_SCHOLAR_API_KEY or None

    logger.info(f"[Semantic Scholar] 搜索: query={query[:80]}, year_range={year_range}, limit={limit}, proxy={proxy or 'none'}, api_key={'yes' if api_key else 'no'}")

    # 如果有代理，设置环境变量（semanticscholar 库底层使用 httpx）
    import os as _os
    old_http_proxy = _os.environ.get("HTTP_PROXY", "")
    old_https_proxy = _os.environ.get("HTTPS_PROXY", "")
    if proxy:
        _os.environ["HTTP_PROXY"] = proxy
        _os.environ["HTTPS_PROXY"] = proxy

    # 尝试顺序：带 API Key -> 匿名访问
    api_keys_to_try = [api_key, None] if api_key else [None]

    last_error = None
    try:
        for try_key in api_keys_to_try:
            try:
                sch_kwargs = {"api_key": try_key}
                if proxy:
                    sch_kwargs["timeout"] = timeout

                sch = SemanticScholar(**sch_kwargs)
                kwargs = {"query": query, "limit": limit, "fields": [
                    "title", "abstract", "year", "authors", "venue",
                    "citationCount", "url", "externalIds"
                ]}
                if year_range:
                    kwargs["year"] = year_range

                results = sch.search_paper(**kwargs)

                papers = []
                for paper in results:
                    authors = ", ".join(
                        a.name for a in (paper.authors or []) if hasattr(a, "name") and a.name
                    )
                    papers.append({
                        "title": paper.title or "",
                        "abstract": paper.abstract or "",
                        "year": paper.year,
                        "authors": authors,
                        "venue": paper.venue or "",
                        "citation_count": paper.citationCount or 0,
                        "url": paper.url or "",
                        "arxiv_id": (paper.externalIds or {}).get("ArXiv", ""),
                        "source": "semantic_scholar",
                    })
                logger.info(f"[Semantic Scholar] 找到 {len(papers)} 篇论文 (api_key={'yes' if try_key else 'anonymous'})")
                return papers

            except PermissionError as e:
                # 403 Forbidden — API Key 失效，回退匿名
                logger.warning(f"[Semantic Scholar] API Key 返回 403 Forbidden，回退匿名访问: {e}")
                last_error = e
                continue
            except Exception as e:
                logger.error(f"[Semantic Scholar] 搜索出错 (api_key={'yes' if try_key else 'anonymous'}): {type(e).__name__}: {e}")
                last_error = e
                continue
    finally:
        # 恢复原始环境变量（无论成功或失败都执行）
        if proxy:
            if old_http_proxy:
                _os.environ["HTTP_PROXY"] = old_http_proxy
            else:
                _os.environ.pop("HTTP_PROXY", None)
            if old_https_proxy:
                _os.environ["HTTPS_PROXY"] = old_https_proxy
            else:
                _os.environ.pop("HTTPS_PROXY", None)

    logger.error(f"[Semantic Scholar] 所有尝试均失败，最后一次错误: {type(last_error).__name__}: {last_error}")
    return []


def _enrich_arxiv_citations(papers: list[dict]) -> None:
    """为 citation_count=0 的 arXiv 论文通过 Semantic Scholar 批量补充引用数（就地修改）
    
    使用 SS 的 get_papers 批量 API，每批最多 500 篇，429 时自动退避。
    """
    import time as _time
    from semanticscholar import SemanticScholar

    proxy = HTTPS_PROXY or HTTP_PROXY
    api_key = SEMANTIC_SCHOLAR_API_KEY or None

    # 构建 ArXiv ID → paper 映射
    id_to_paper = {}
    for paper in papers:
        arxiv_id = paper.get("arxiv_id", "")
        if arxiv_id:
            id_to_paper[f"ArXiv:{arxiv_id}"] = paper

    if not id_to_paper:
        return

    import os as _os
    old_http_proxy = _os.environ.get("HTTP_PROXY", "")
    old_https_proxy = _os.environ.get("HTTPS_PROXY", "")
    if proxy:
        _os.environ["HTTP_PROXY"] = proxy
        _os.environ["HTTPS_PROXY"] = proxy

    try:
        api_keys_to_try = [api_key, None] if api_key else [None]
        for try_key in api_keys_to_try:
            try:
                sch_kwargs = {"api_key": try_key}
                if proxy:
                    sch_kwargs["timeout"] = 30
                sch = SemanticScholar(**sch_kwargs)

                enriched = 0
                batch_size = 500  # SS API 上限
                all_ids = list(id_to_paper.keys())

                for batch_start in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[batch_start:batch_start + batch_size]

                    # 429 重试
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            ss_papers = sch.get_papers(
                                batch_ids,
                                fields=["citationCount", "externalIds"]
                            )
                            for ss_paper in ss_papers:
                                if ss_paper is None:
                                    continue
                                # 找到对应的原始 paper
                                # SS 返回的 paper 有 paperId，需要通过 externalIds 匹配
                                ext_ids = ss_paper.externalIds or {} if hasattr(ss_paper, 'externalIds') and ss_paper.externalIds else {}
                                arxiv_id_val = ext_ids.get("ArXiv", "")
                                lookup_key = f"ArXiv:{arxiv_id_val}" if arxiv_id_val else ""
                                target_paper = id_to_paper.get(lookup_key)
                                if target_paper and ss_paper.citationCount is not None:
                                    target_paper["citation_count"] = ss_paper.citationCount
                                    enriched += 1
                            break  # 成功则跳出重试循环
                        except ConnectionRefusedError:
                            # 429 Too Many Requests
                            wait = 10 * (attempt + 1)
                            logger.warning(f"[Executor] 补充引用数: 429 限流，等待 {wait}s...")
                            _time.sleep(wait)
                        except Exception as e:
                            logger.warning(f"[Executor] 补充引用数批量出错: {type(e).__name__}: {e}")
                            if attempt < max_retries - 1:
                                _time.sleep(5)
                            else:
                                break

                    # 批次间短暂等待，避免触发限流
                    if batch_start + batch_size < len(all_ids):
                        _time.sleep(1)

                logger.info(f"[Executor] 补充引用数完成: {enriched}/{len(papers)} 篇 arXiv 论文获得了引用数 (api_key={'yes' if try_key else 'anonymous'})")
                return
            except PermissionError:
                logger.warning(f"[Executor] 补充引用数: API Key 403，回退匿名")
                continue
            except Exception as e:
                logger.warning(f"[Executor] 补充引用数出错 (api_key={'yes' if try_key else 'anonymous'}): {type(e).__name__}: {e}")
                continue
    finally:
        if proxy:
            if old_http_proxy:
                _os.environ["HTTP_PROXY"] = old_http_proxy
            else:
                _os.environ.pop("HTTP_PROXY", None)
            if old_https_proxy:
                _os.environ["HTTPS_PROXY"] = old_https_proxy
            else:
                _os.environ.pop("HTTPS_PROXY", None)


def _relax_arxiv_query(query: str) -> str:
    """将过于严格的 arXiv 查询放宽
    
    策略：
    1. 去掉 all: 前缀，替换为 abs:
    2. 如果 AND 条件超过 3 个，只保留前 3 个
    3. 去掉引号精确匹配（允许部分匹配）
    """
    import re
    relaxed = query
    # 去掉 all: 前缀 → abs:
    relaxed = re.sub(r'\ball:', 'abs:', relaxed)
    
    # 按 AND 分割，只保留前 3 个子句
    # 注意保留括号内的 OR 结构
    parts = re.split(r'\s+AND\s+', relaxed, flags=re.IGNORECASE)
    if len(parts) > 3:
        relaxed = ' AND '.join(parts[:3])
        logger.info(f"[arXiv] 查询放宽：AND 条件从 {len(parts)} 个减少到 3 个")
    
    return relaxed


def _search_arxiv(query: str, max_results: int = 50, year_range: str | None = None) -> list[dict]:
    """使用 arXiv API 搜索论文，带重试机制和查询回退"""
    import arxiv
    import time
    import re

    proxy = HTTPS_PROXY or HTTP_PROXY
    logger.info(f"[arXiv] 搜索: query={query[:80]}, max_results={max_results}, year_range={year_range}, proxy={proxy or 'none']}")

    # 如果有代理，设置环境变量（arxiv 底层使用 httpx/requests）
    import os as _os
    old_http_proxy = _os.environ.get("HTTP_PROXY", "")
    old_https_proxy = _os.environ.get("HTTPS_PROXY", "")
    if proxy:
        _os.environ["HTTP_PROXY"] = proxy
        _os.environ["HTTPS_PROXY"] = proxy

    # 如果有年份范围，在查询中加入 submittedDate 过滤
    # arXiv 语法: submittedDate:[YYYYMMDDTTTT TO YYYYMMDDTTTT]
    effective_query = query
    if year_range:
        try:
            parts = year_range.split("-")
            start_year = int(parts[0])
            end_year = int(parts[1]) if len(parts) > 1 else start_year
            date_filter = f'AND submittedDate:[{start_year}01010000 TO {end_year}12312359]'
            effective_query = f"({query}) {date_filter}"
            logger.info(f"[arXiv] 添加年份过滤: {date_filter}")
        except (ValueError, IndexError):
            logger.warning(f"[arXiv] 无效的 year_range: {year_range}，忽略")

    # 准备查询回退策略：原始查询 → 放宽查询 → 最简关键词
    queries_to_try = [effective_query]
    relaxed_query = _relax_arxiv_query(effective_query)
    if relaxed_query != effective_query:
        queries_to_try.append(relaxed_query)
    # 最简策略：提取引号中的核心关键词，用空格连接（arXiv 默认 OR 语义）
    core_terms = re.findall(r'"([^"]+)"', query)
    if core_terms:
        simple_query = " ".join(core_terms[:4])  # 最多 4 个核心词
        if year_range:
            try:
                parts = year_range.split("-")
                start_year = int(parts[0])
                end_year = int(parts[1]) if len(parts) > 1 else start_year
                simple_query = f'({simple_query}) AND submittedDate:[{start_year}01010000 TO {end_year}12312359]'
            except (ValueError, IndexError):
                pass
        if simple_query not in queries_to_try:
            queries_to_try.append(simple_query)

    max_retries = 3
    try:
        for query_idx, current_query in enumerate(queries_to_try):
            if query_idx > 0:
                logger.info(f"[arXiv] 查询回退 (第 {query_idx+1} 种): query={current_query[:80]}")
            for attempt in range(max_retries):
                try:
                    client = arxiv.Client()
                    search = arxiv.Search(query=current_query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
                    results = list(client.results(search))

                    papers = []
                    for result in results:
                        authors = ", ".join(a.name for a in result.authors)
                        papers.append({
                            "title": result.title or "",
                            "abstract": result.summary or "",
                            "year": result.published.year if result.published else None,
                            "authors": authors,
                            "venue": "",
                            "citation_count": 0,
                            "url": result.entry_id or "",
                            "arxiv_id": result.get_short_id() if hasattr(result, "get_short_id") else "",
                            "source": "arxiv",
                        })
                    logger.info(f"[arXiv] 找到 {len(papers)} 篇论文 (query={current_query[:60]})")
                    if papers:
                        return papers
                    # 0 结果 → 尝试下一个宽松查询（不重试当前查询）
                    logger.info(f"[arXiv] 当前查询返回 0 结果，尝试更宽松的查询")
                    break

                except Exception as e:
                    logger.warning(f"[arXiv] 搜索出错 (query#{query_idx+1}, 第 {attempt+1}/{max_retries} 次): {type(e).__name__}: {e}")
                    if attempt < max_retries - 1:
                        wait = 5 * (attempt + 1)  # 递增等待: 5s, 10s, 15s
                        logger.info(f"[arXiv] 等待 {wait}s 后重试...")
                        time.sleep(wait)
    finally:
        # 恢复原始环境变量（无论成功或失败都执行）
        if proxy:
            if old_http_proxy:
                _os.environ["HTTP_PROXY"] = old_http_proxy
            else:
                _os.environ.pop("HTTP_PROXY", None)
            if old_https_proxy:
                _os.environ["HTTPS_PROXY"] = old_https_proxy
            else:
                _os.environ.pop("HTTPS_PROXY", None)

    logger.error(f"[arXiv] 搜索失败，已重试 {max_retries} 次")
    return []


class Executor:
    """Executor Agent：论文搜索与筛选"""

    def check_query_sufficiency(self, task_id: str, user_id: str = "") -> dict:
        """检查用户查询是否包含足够信息进行论文搜索"""
        user_info = ms.read_user_info(user_id)
        research_task = ms.read_research_task(task_id)

        logger.info(f"[Executor] check_query_sufficiency 输入: user_info={user_info}, research_task={research_task}")

        result_str = chat_with_system(
            CHECK_QUERY_SUFFICIENCY_PROMPT.format(
                user_info=user_info or "(none)",
                research_task=research_task or "(none)",
            )
        )
        result = extract_json(result_str)
        if result is None:
            logger.warning(f"查询充分性检查 JSON 解析失败，原始返回:\n{result_str}")
            result = {"sufficient": False, "missing_info": "无法解析"}

        logger.info(f"[Executor] check_query_sufficiency 输出: {result}")
        ms.append_timeline(task_id, "executor", f"查询充分性检查: {result}")
        return result

    def search_papers(self, task_id: str, assistant_context: str) -> list[dict]:
        """执行论文搜索"""
        research_task = ms.read_research_task(task_id)

        logger.info(f"[Executor] search_papers 输入: research_task={research_task}, assistant_context={assistant_context[:200]}")

        # 生成搜索查询
        query_str = chat_with_system(
            SEARCH_SYSTEM_PROMPT,
            f"研究任务: {research_task}\n补充背景: {assistant_context}",
        )
        query_info = extract_json(query_str)
        if query_info is None:
            logger.warning(f"搜索查询 JSON 解析失败，原始返回:\n{query_str}")
            query_info = {
                "semantic_scholar_query": research_task,
                "arxiv_query": research_task,
                "year_range": "",
                "max_results": MAX_PAPERS_TO_SEARCH,
            }

        ms.append_timeline(task_id, "executor", f"生成搜索查询: {query_str}")

        # 执行搜索
        all_papers = []
        ss_query = query_info.get("semantic_scholar_query", "")
        arxiv_query = query_info.get("arxiv_query", "")
        year_range = query_info.get("year_range", "") or None
        # 强制使用 config 中的 MAX_PAPERS_TO_SEARCH，忽略 LLM 返回的 max_results
        max_results = MAX_PAPERS_TO_SEARCH

        # Semantic Scholar: 先用原始查询搜索，若 0 结果则放宽查询
        if ss_query:
            ss_papers = _search_semantic_scholar(ss_query, year_range, max_results)
            if not ss_papers:
                # 放宽策略：减少 AND 条件（只保留前 2-3 个核心子句）
                import re
                parts = re.split(r'\s+AND\s+', ss_query, flags=re.IGNORECASE)
                if len(parts) > 2:
                    relaxed_ss = ' AND '.join(parts[:2])
                    logger.info(f"[Semantic Scholar] 查询放宽: {ss_query[:60]} → {relaxed_ss[:60]}")
                    ss_papers = _search_semantic_scholar(relaxed_ss, year_range, max_results)
            all_papers.extend(ss_papers)

        # arXiv: 传入 year_range，内部已有查询回退逻辑
        if arxiv_query:
            all_papers.extend(_search_arxiv(arxiv_query, max_results, year_range=year_range))

        # 元数据过滤：在去重之前先排除不符合基本要求的论文
        filtered_papers = []
        filtered_out = 0
        for p in all_papers:
            # 排除无摘要的论文（无法进行后续筛选）
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
            # 排除超出年份范围的论文（API 层面可能漏掉，这里二次校验）
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
            logger.info(f"[Executor] 元数据过滤: 排除 {filtered_out} 篇不符合要求的论文 (无摘要/无标题/年份不符)")

        # 去重（按标题）— 优先保留有 citation_count 的版本
        seen_titles = {}
        unique_papers = []
        for p in filtered_papers:
            title_lower = p["title"].lower().strip()
            if title_lower not in seen_titles:
                seen_titles[title_lower] = len(unique_papers)
                unique_papers.append(p)
            else:
                # 已存在：如果新版本有 citation_count 而旧版本没有，替换
                idx = seen_titles[title_lower]
                old_cc = unique_papers[idx].get("citation_count", 0) or 0
                new_cc = p.get("citation_count", 0) or 0
                if new_cc > 0 and old_cc == 0:
                    logger.info(f"[去重] 替换论文引用数: '{title_lower[:50]}' 0 → {new_cc} (source: {p.get('source', '?')})")
                    unique_papers[idx] = p

        # 补充 arXiv 论文的 citation_count（通过 Semantic Scholar 批量查找）
        arxiv_only = [p for p in unique_papers if p.get("source") == "arxiv" and (p.get("citation_count") or 0) == 0 and p.get("arxiv_id")]
        if arxiv_only:
            logger.info(f"[Executor] 尝试为 {len(arxiv_only)} 篇 arXiv 论文补充引用数")
            _enrich_arxiv_citations(arxiv_only)

        # 保存论文列表
        ms.write_paper_list(task_id, unique_papers)
        ms.append_timeline(task_id, "executor", f"搜索完成，找到 {len(unique_papers)} 篇不重复论文")
        logger.info(f"[Executor] search_papers 输出: 找到 {len(unique_papers)} 篇不重复论文")

        return unique_papers

    def screen_papers_sample(self, task_id: str, assistant_context: str) -> dict:
        """对论文进行初步样例筛选（批量模式）"""
        logger.info(f"[Executor] screen_papers_sample 输入: assistant_context={assistant_context[:200]}")
        papers = ms.read_paper_list(task_id)
        included = []
        excluded = []
        not_sure = []

        # 筛选未处理的论文
        unscreened = [(i, p) for i, p in enumerate(papers) if "screening_decision" not in p]
        logger.info(f"初步筛选: 共 {len(papers)} 篇, 未筛选 {len(unscreened)} 篇")

        batch_idx = 0
        for batch_start in range(0, len(unscreened), SCREEN_BATCH_SIZE):
            if len(included) >= SAMPLE_INCLUDE_THRESHOLD:
                break

            batch = unscreened[batch_start:batch_start + SCREEN_BATCH_SIZE]
            batch_idx += 1
            logger.info(f"筛选第 {batch_idx} 批, 共 {len(batch)} 篇")

            # 构建批量论文信息
            papers_text = ""
            original_indices = []
            for j, (orig_idx, paper) in enumerate(batch):
                papers_text += f"\n--- 论文 {j+1} ---\n"
                papers_text += f"标题: {paper.get('title', '')}\n"
                papers_text += f"年份: {paper.get('year', '')}\n"
                papers_text += f"作者: {paper.get('authors', '')}\n"
                papers_text += f"摘要: {paper.get('abstract', '')[:500]}\n"
                original_indices.append(orig_idx)

            result_str = chat_with_system(
                SCREEN_SYSTEM_PROMPT,
                f"用户需求和偏好:\n{assistant_context}\n\n论文列表:{papers_text}",
            )
            result = extract_json(result_str)
            if result is None:
                logger.warning(f"批量筛选 JSON 解析失败，原始返回:\n{result_str}")
                # 回退：逐篇标记为 not_sure
                for orig_idx, paper in batch:
                    paper["screening_decision"] = "not_sure"
                    paper["screening_reason"] = "解析失败"
                    not_sure.append(paper)
                continue

            # result 应该是列表
            if isinstance(result, dict):
                result = [result]

            # 建立 index -> decision 映射
            decisions = {}
            for item in result:
                idx = item.get("index", 0) - 1  # 转为 0-based
                if 0 <= idx < len(batch):
                    decisions[idx] = item

            # 处理每篇论文
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

            logger.info(f"第 {batch_idx} 批筛选完成: included={len(included)}, excluded={len(excluded)}, not_sure={len(not_sure)}")

        # 保存筛选结果回 paper_list
        ms.write_paper_list(task_id, papers)

        ms.append_timeline(
            task_id, "executor",
            f"样例筛选完成: {len(included)} include, {len(excluded)} exclude, {len(not_sure)} not sure"
        )
        logger.info(f"[Executor] screen_papers_sample 输出: included={len(included)}, excluded={len(excluded)}, not_sure={len(not_sure)}")

        return {
            "included": included,
            "excluded": excluded,
            "not_sure": not_sure,
            "done": len(included) >= SAMPLE_INCLUDE_THRESHOLD,
        }

    def score_papers(self, task_id: str) -> list[dict]:
        """根据打分标准对全量论文列表进行打分排序（批量模式）"""
        criteria = ms.read_scoring_criteria(task_id)
        logger.info(f"[Executor] score_papers 输入: criteria={criteria[:200]}")
        papers = ms.read_paper_list(task_id)
        logger.info(f"开始打分，共 {len(papers)} 篇论文，每批 {SCORE_BATCH_SIZE} 篇")

        batch_idx = 0
        for batch_start in range(0, len(papers), SCORE_BATCH_SIZE):
            batch = papers[batch_start:batch_start + SCORE_BATCH_SIZE]
            batch_idx += 1
            logger.info(f"打分第 {batch_idx} 批, 共 {len(batch)} 篇")

            # 构建批量论文信息
            papers_text = ""
            for j, paper in enumerate(batch):
                papers_text += f"\n--- 论文 {j+1} ---\n"
                papers_text += f"标题: {paper.get('title', '')}\n"
                papers_text += f"年份: {paper.get('year', '')}\n"
                papers_text += f"摘要: {paper.get('abstract', '')[:500]}\n"

            result_str = chat_with_system(
                SCORE_SYSTEM_PROMPT,
                f"打分标准:\n{criteria}\n\n论文列表:{papers_text}",
            )
            result = extract_json(result_str)
            if result is None:
                logger.warning(f"批量打分 JSON 解析失败，原始返回:\n{result_str}")
                for paper in batch:
                    paper["scores"] = {}
                    paper["total_score"] = 0
                    paper["brief_comment"] = "解析失败"
                continue

            if isinstance(result, dict):
                result = [result]

            # 建立 index -> scores 映射
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

            logger.info(f"第 {batch_idx} 批打分完成")

        # 按总分降序排序
        papers.sort(key=lambda x: x.get("total_score", 0), reverse=True)

        ms.write_scored_papers(task_id, papers)
        ms.append_timeline(task_id, "executor", f"打分排序完成，共 {len(papers)} 篇论文")
        logger.info(f"[Executor] score_papers 输出: 共 {len(papers)} 篇，Top3 分数: {[p.get('total_score', 0) for p in papers[:3]]}")

        return papers
