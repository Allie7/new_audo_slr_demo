# Automatic Systematic Literature Review 项目

## 项目概述

这是一个自动系统文献综述（Automatic Systematic Literature Review）项目，主要优化点在 **memory management**。项目采用 **multi-agent 策略**，为每个 agent 分配不同的角色，每个角色只保留与自身工作内容相关的 memory/context，避免超长 context 问题。

---

## Agent 角色定义

### 1. Coordinator（流程控制者）
- 判断当前应该调用哪个 agent 进行什么工作
- 可以是 LLM agent，现阶段也可以代码直接写死

### 2. Assistant/Observer（信息提取与记录者）
- 提取用户 query 和其他 agent 工作输出中的有效信息
- 分别归类记录在相关历史信息文件里
- 在 executor 执行任务前和 executor 进行交流，补充相关背景信息
- 只需记住当前轮对话信息并读取之前的记录文件

**assistant 应抽取和记录的信息示例：**

| 原始信息 | 抽取结果 |
|---------|---------|
| 用户 query: "I am a CS post-grad student, I want to do a research on llm agent memory management. Show me relevant papers in the recent two years." | User identity: post-grad student, User's academic field: CS (记录在 `user_info.txt`) |
| | Research Direction: llm agent memory management, time range requirement: past 2 years (记录在 `research_task_xx.txt`) |

**assistant 应抽取的信息包括但不限于：**
- 用户偏好
- 用户提供的学术背景信息
- Research 方向
- Papers 筛选要求
- 用户弃用某论文的原因等

> **注意**：由于用户提供的信息可能重复或前后矛盾，assistant 每次记录新信息前应读取完整原记录内容，和当前最新信息进行比对，修改并重写一个最新版本。

### 3. Executor（任务执行者）
- 接收到 coordinator 的任务后与 assistant 交流
- context/memory 只有 coordinator 的要求和 assistant 的补充信息

### 4. Contactor（用户交互者）
- 负责和用户进行直接交流
- 只需记住前三轮左右的 context 和 coordinator、assistant 或 executor 本轮提供的信息

### 5. Extractor（信息抽取者）
- 本地部署小模型，针对抽取论文（全文）关键数据和信息进行微调

### 6. Ranker（打分模型）
- 本地部署模型（bert，BioBERT等），针对论文相关性排序进行微调。

### 7. Ranker团队 (multi-agent打分团队)
- multi-agent讨论机制，在ranker和executor打分差距大时引入

---

## 调用工具与 API

### 1. 论文查找
- arxiv
- semanticscholar
- asta-paperfinder
- pubmed

### 2. PDF 处理
- PyMuPDF、fitz、pypdf、pdfplumber 等

### 3. 关键信息提取
- 本地小模型

### 4. 打分模型
- Bert, BioBert等本地模型

---

## 工作流程

### 第一步：Query 收集与验证
1. 用户提供一个做 literature review 相关的 query
2. assistant 将用户本轮对话中已表述的信息进行抽取和记录，并补充必要历史信息给 coordinator
3. coordinator 读取 query 和 assistant 提供的信息，调用 executor 判断是否包含做初步论文搜索的必要信息
4. 如果没有，委托 contactor 和用户继续对话，直到用户的 query 提供了必要信息
5. 如果包含必要信息，coordinator 将要求 executor 使用 Semantic Scholar API 或 arxiv 包，或直接使用 asta-paper-finder 的 skill 进行论文搜索

### 第二步：论文搜索
1. executor 执行搜索
2. assistant 记录 executor 的执行记录
3. executor 输出一个论文列表的 JSON 文件，包括论文的 metadata 和 abstract

### 第三步：样例初步筛选
1. assistant 提供给 executor 之前记录的历史信息（用户的历史偏好和当前要求等）
2. executor 读取论文列表文件，根据 abstract 对论文进行初步筛选
3. 将每篇论文标注为：`include`、`exclude`、`not sure`
4. 如果是 include，记录下来并标注 include 的理由
5. 当找满 **三篇 include 的文章** + **三篇 exclude 的文章** 时停止读取，提示 coordinator 样例 paper 已筛选完毕

### 第四步：用户确认与迭代
1. coordinator 确认 executor 完成初步筛选
2. 调用 contactor 和用户对话，要求用户查看样例 paper 的 abstract，确认标注是否符合心意
3. 如果有 paper 不符合，诱导用户给出理由
4. assistant 同时记录用户给出的理由
5. 重复第三步，直到满足以下任一条件：
   - 根据用户的当前 query 返回的所有 paper 用户都完全满意
   - 存储达标论文列表的文件中已记录了 **十篇及以上** 标注让用户满意的paper
   - **兜底设置: 1. 已经迭代10轮对话后结束, contactor引导用户回到第一步重新确认信息 2. 已screen部分paper数达到阈值，contactor引导用户回到第一步重新确认信息 
6. 注意：用户表达对标注满意的 paper 应该被记录到相应存储达标论文列表的文件里

### 第五步：全量screen
1. coordinator 调用executor进行全量screen。assistant提供已记录用户偏好和研究要求给executor，executor标注并给出标注理由
2. 标注为`include`、`exclude`、`not sure`的paper信息分别存放在三个不同的对应文档里，同时记录executor给出的标注理由

### 第六步：打分
1. 综合Ranker和executor（llm）对标为include和not_sure的论文进行打分，记录executor给出的打分理由。如果同一篇文章两者打分相差过大，引入Ranker团队进行重新打分
2. 根据打分进行排序

### ***第六步半：抽查
1. coordinator 要求 contactor 告知用户对结果进行抽查
2. 如果不满意，重复以上过程

### 第七步：论文下载与文字提取
- 按评分降序分批次下载论文全文，提取文字

### 第八步：筛选标准收集
1. coordinator 要求 contactor 告知用户提供详细筛选标准
2. 将用户的回答提供给 assistant
3. assistant 将用户的原始标准记录并拆解
4. 拆解出需要抽取的关键内容/信息（如力谱类型、被测分子类型等）
5. 传给小模型 extractor

### 第九步：关键信息抽取
- extractor 抽取关键信息
- 维度由user提供+assitant根据之前对话内容进行提取
- 整理为 JSONL 文件

### 第十步：最终判定与打分
1. assistant 根据原始标准对 extractor 抽取的信息进行进一步判定并打分
2. 输出最终打分文件

---

## 文件存储规范

- 所有用户和模型的输出内容（除已额外存储在单独文件中的，如候选论文列表）将按时间线记录在单独文件中
- 用户信息：`user_info_aa.txt` (aa是user代码)
- 研究任务：`research_task_xx.txt`（xx 是当前 task 的代码）
- 候选论文列表：`candidate_papers_xx.jsonl`    
- 已筛选论文列表：`selected_papers_xx.jsonl`
- 已下载论文全文：`downloaded_papers_xx.jsonl`
- 已抽取关键信息：`extracted_info_xx.jsonl`

