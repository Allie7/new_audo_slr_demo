# Automatic Systematic Literature Review

基于 Multi-Agent 策略的自动系统文献综述系统，核心优化点在于 **Memory Management**。

## 设计理念

传统的自动 SLR 系统将所有信息塞入 LLM 的 context，导致超长 context 问题。本项目采用 **分角色记忆管理** 策略：

| Agent | 职责 | 记忆范围 |
|-------|------|----------|
| **Coordinator** | 流程控制 | 仅当前流程状态 |
| **Assistant** | 信息抽取与记录 | 当前轮对话 + 读取历史记录文件 |
| **Executor** | 搜索与筛选 | Coordinator 要求 + Assistant 补充信息 |
| **Contactor** | 用户交互 | 最近 3 轮对话 + 本轮额外信息 |

所有信息通过文件系统持久化，每个 agent 按需读取，避免不必要的 context 膨胀。

## 系统架构

```
用户 query
    │
    ▼
┌──────────┐   信息不充分   ┌──────────┐
│ Assistant │──────────────▶│ Contactor │──▶ 询问用户
│ (抽取信息) │◀──────────────│ (用户交互) │◀── 用户回答
└────┬─────┘                └──────────┘
     │ 信息充分
     ▼
┌──────────┐
│ Executor  │ 1. 搜索论文 (Semantic Scholar + arXiv + pubmed)
│ (搜索筛选) │ 
└────┬─────┘ 2. 迭代样例筛选 + 用户反馈循环
     │         4. 全量筛选剩余论文
     ▼         5. 生成打分标准 → 打分排序
┌──────────┐
│  结果输出 │ include / exclude / not_sure + 打分排序
└──────────┘
```

## 文件结构

```
new_auto_slr/
├── main.py                          # 交互式主入口
├── config.py                        # 配置文件（LLM、搜索参数等）
├── coordinator.py                    # 流程控制器
├── assistant.py                      # 信息抽取与记忆管理 Agent
├── executor.py                       # 搜索与筛选 Agent
├── contactor.py                      # 用户交互 Agent
├── memory_store.py                   # 文件记忆管理模块 + InMemoryStore 类
├── llm_client.py                    # LLM API 封装（系统 LLM + 模拟学生 LLM）
├── logging_config.py                 # 日志配置
├── build_papers.py                   # 从 CSMeD-FT splits 构建去重论文集
├── build_training_set.py             # 从 prompt 记录 + gold standard 构建训练集 JSONL
├── merge_gold_standard.py           # 合并 gold standard 数据
├── merge_reviews_json.py            # 合并 reviews metadata
├── test_csmed.py                    # CSMeD-FT 数据集完整测试脚本
├── test_csmed_export_prompt.py      # 导出 Stage 2.5 executor prompt（不执行筛选）
├── test_csmed_baseline.py           # 基线测试
├── requirements.txt                  # 依赖
├── .env                             # 环境变量
├── data/                            # 交互式运行时数据目录
├── data_csmed_test/                  # test_csmed.py 输出目录
├── data_csmed_prompt_export/         # test_csmed_export_prompt.py 输出目录
└── LitSearchRetrieval/              # LitSearch 检索索引
```

## 工作流程

### 交互式流程（`main.py` → `coordinator.py`）

```
1. 用户 query → Assistant 抽取信息 → Executor 判断信息是否充分
   ↓ (不充分)                    ↓ (充分)
   Contactor 询问用户          2. Executor 搜索论文
                                ↓
   3. Executor 迭代样例筛选 → Contactor 展示给用户 → 用户确认
      ↓ (不满意 → 记录原因)     ↓ (满意 → 记录达标)
      重复筛选                   检查是否达标 5 篇
                                ↓
   4. Assistant 生成打分标准 → Executor 对全量论文打分排序 → 输出结果
```

### 测试流程（`test_csmed.py`）

基于 CSMeD-FT 数据集的自动化评测，模拟用户反馈：

| 阶段 | 说明 |
|------|------|
| **Stage 1** | 信息抽取：review title → Assistant 抽取；逐步补充 criteria/abstract/search_strategy |
| **Stage 1.5** | Metadata 初筛（当前已跳过） |
| **Stage 1.8** | Title 筛选（当前已跳过） |
| **Stage 2** | 迭代样例筛选 + Gold Standard 模拟用户反馈 |
| **Stage 2.5** | 全量筛选剩余 unscreened 论文 |
| **评估** | 系统 include vs Gold included：Precision / Recall / F1 |

**Stage 2 核心逻辑**：
- 每轮找到 3 篇 include 即暂停，用 gold standard 模拟用户反馈
- 满意的论文记入 `satisfied_papers`，不满意的原因由 Assistant 学习
- 退出条件：本轮所有 include 用户都满意 / 累计满意 ≥ 5 / 达到 5 轮兜底
- **累计 2 篇 include 后，每轮筛选不超过 20 篇**（避免过度消耗 LLM token）

**Stage 2.5 核心逻辑**：
- 对 Stage 2 留下的所有 unscreened 论文做无门槛全量筛选
- 累计 ≥2 篇 include 时每轮最多处理 20 篇，分多轮直至筛完
- 未满 2 篇 include 时不限量，一次处理完即退出

### Prompt 导出流程（`test_csmed_export_prompt.py`）

运行到 Stage 2.5 时，不执行实际筛选，仅记录 executor 会收到的完整 prompt（system + user），用于后续微调训练集构建。

- 若 Stage 2 已覆盖所有论文（无 unscreened），则将全部已筛选论文当作 unscreened 生成 prompt，避免空转
- 输出 `executor_prompt_record.txt`（每行一条 JSON）

### 训练集构建流程

```
test_csmed_export_prompt.py → executor_prompt_record.txt
                                      ↓
                         build_training_set.py → training_set.jsonl
                                      ↑
                         CSMeD-FT-gold_standard.json (提供 decision label)
```

训练集格式（OpenAI ChatCompletion 风格）：
```json
{"messages": [
  {"role": "system", "content": "<SCREEN_SYSTEM_PROMPT + 用户需求偏好>"},
  {"role": "user", "content": "<单篇论文标题+摘要>"},
  {"role": "assistant", "content": "included" / "excluded"}
]}
```

## 论文筛选策略

### 筛选 Prompt（`SCREEN_SYSTEM_PROMPT`）

Executor 使用批量筛选 prompt，对每篇论文给出 include / exclude / not_sure 三分类：
- **include**：论文与研究方向高度相关，满足用户所有筛选要求
- **exclude**：论文明显不相关或不符合筛选要求
- **not_sure**：仅凭摘要无法确定是否应纳入

### 批量参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `SCREEN_BATCH_SIZE` | 10 | 每批筛选论文数 |
| `TITLE_SCREEN_BATCH_SIZE` | 20 | Title 初筛每批论文数 |
| `SCORE_BATCH_SIZE` | 5 | 每批打分论文数 |
| `SAMPLE_INCLUDE_THRESHOLD` | 3 | 每轮样例筛选 include 阈值 |
| `MAX_SATISFIED_PAPERS` | 5 | 达标论文最大数量 |

### 搜索策略（`Executor.search_papers`）

1. LLM 生成 Semantic Scholar + arXiv 搜索查询
2. 双源搜索，元数据过滤（无摘要/无标题/年份不符 → 排除）
3. 按标题去重，优先保留有引用数的版本
4. 为 arXiv 论文通过 Semantic Scholar 批量补充引用数
5. arXiv 查询自动放宽（`all:` → `abs:`、AND 条件 ≤3、核心关键词回退）

## 数据存储

### 交互模式（`memory_store.py` 模块函数）

文件存储在 `data/` 目录下：

| 文件 | 说明 |
|------|------|
| `user_info_{user_id}.txt` | 用户身份、学术背景、偏好 |
| `research_task_{task_id}.txt` | 研究任务信息 |
| `paper_list_{task_id}.json` | 全量论文列表 |
| `satisfied_papers_{task_id}.json` | 达标论文列表 |
| `scoring_criteria_{task_id}.txt` | 打分标准 |
| `scored_papers_{task_id}.json` | 打分排序后论文 |
| `timeline_{task_id}.log` | 时间线日志 |

### 测试模式（`InMemoryStore`）

每个 review 使用独立的 `InMemoryStore` 实例：
- 内存 + 实时文件写入（数据同时保存在内存和输出目录）
- 天然线程安全，支持多线程并发测试
- 输出目录：`data_csmed_test/{review_id}/`

## LLM 配置

系统使用两套 LLM：

| 用途 | 默认模型 | 配置项 |
|------|----------|--------|
| 系统 LLM（Assistant/Executor/Contactor） | GLM-5.1 | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` |
| 模拟学生 LLM | DeepSeek Chat | `SIMULATOR_API_KEY`, `SIMULATOR_BASE_URL`, `SIMULATOR_MODEL` |

## 快速开始

### 交互式运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY 等

# 3. 运行
python main.py
```

### CSMeD-FT 测试

```bash
# 默认：跑全部 213 个 review，3 线程
python test_csmed.py

# 只跑前 5 个 review
python test_csmed.py --num-reviews 5

# 从第 10 个开始跑 20 个
python test_csmed.py --offset 10 --num-reviews 20

# 5 线程并发
python test_csmed.py --num-reviews 5 --workers 5
```

**输出**（`data_csmed_test/`）：
- 每个 review 一个子目录（含筛选详情、gold 对比）
- `report_*.txt` — 文本报告
- `summary_*.json` — 详细 JSON 结果
- `test_csmed_*.log` — 运行日志

### 导出 Executor Prompt

```bash
# 同 test_csmed.py 的参数
python test_csmed_export_prompt.py --num-reviews 5
```

**输出**（`data_csmed_prompt_export/`）：
- `executor_prompt_record.txt` — 逐篇 prompt 记录（每行一个 JSON）
- `export_prompt_*.log` — 运行日志

### 构建训练集

```bash
python build_training_set.py
```

读取 `executor_prompt_record.txt` + `CSMeD-FT-gold_standard.json`，输出 `training_set.jsonl`。

## 配置说明

### 环境变量（`.env`）

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | 系统 LLM API 密钥 |
| `OPENAI_BASE_URL` | API 基础 URL（支持自定义端点） |
| `OPENAI_MODEL` | 使用的模型名称（默认 `glm-5.1`） |
| `SIMULATOR_API_KEY` | 模拟学生 LLM API 密钥 |
| `SIMULATOR_BASE_URL` | 模拟学生 LLM 基础 URL |
| `SIMULATOR_MODEL` | 模拟学生模型名称（默认 `deepseek-chat`） |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API Key（可选，不设则匿名访问） |
| `HTTP_PROXY` / `HTTPS_PROXY` | 代理配置（访问海外 API） |

### `config.py` 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_PAPERS_TO_SEARCH` | 100 | 每次搜索最大论文数 |
| `SAMPLE_INCLUDE_THRESHOLD` | 3 | 样例筛选 include 阈值 |
| `MAX_SATISFIED_PAPERS` | 5 | 达标论文最大数量 |
| `CONTRACTOR_CONTEXT_ROUNDS` | 3 | Contactor 记住的对话轮数 |

## CSMeD-FT 数据集

数据源路径：`CSMED_DATA_DIR`（硬编码在测试脚本中）

| 文件 | 说明 |
|------|------|
| `CSMeD-FT-all_reviews_metadata.json` | 213 个 review 的元数据 |
| `CSMeD-FT-papers.json` | 3333 篇去重论文 |
| `CSMeD-FT-gold_standard.json` | 3333 条 review-document 决策记录 |

测试时每个 review 仅筛选 gold_standard 中关联的论文（`gold_standard[rid].keys()`），而非全量 3333 篇，以提升运行效率。

## 评估指标

系统对每个 review 计算：

- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = 2 × P × R / (P + R)

其中 TP = 系统 include ∩ Gold included，报告按 review_type 分组汇总，并给出全局 TP/FP/FN 汇总。
