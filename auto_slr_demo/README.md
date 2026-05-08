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

## 工作流程

```
1. 用户 query → Assistant 抽取信息 → Executor 判断信息是否充分
   ↓ (不充分)                    ↓ (充分)
   Contactor 询问用户          2. Executor 搜索论文
                                ↓
   3. Executor 初步筛选样例 → Contactor 展示给用户 → 用户确认
      ↓ (不满意 → 记录原因)     ↓ (满意 → 记录达标)
      重复筛选                   检查是否达标 10 篇
                                ↓
   4. Assistant 生成打分标准 → Executor 对全量论文打分排序 → 输出结果
```

## 文件结构

```
new_auto_slr/
├── main.py              # 主入口
├── config.py            # 配置文件
├── coordinator.py       # 流程控制器
├── assistant.py         # 信息抽取与记忆管理 Agent
├── executor.py          # 搜索与筛选 Agent
├── contactor.py         # 用户交互 Agent
├── memory_store.py      # 文件记忆管理模块
├── llm_client.py        # LLM API 封装
├── requirements.txt     # 依赖
├── .env.example         # 环境变量示例
└── data/                # 运行时数据目录
    ├── user_info.txt              # 用户信息
    ├── research_task_{id}.txt     # 研究任务信息
    ├── paper_list_{id}.json       # 全量论文列表
    ├── satisfied_papers_{id}.json # 达标论文列表
    ├── scoring_criteria_{id}.txt  # 打分标准
    ├── scored_papers_{id}.json    # 打分排序后论文
    └── timeline_{id}.log          # 时间线日志
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 OPENAI_API_KEY

# 3. 运行
python main.py
```

## 配置说明

在 `.env` 中配置：

- `OPENAI_API_KEY`: OpenAI API 密钥
- `OPENAI_BASE_URL`: API 基础 URL（支持自定义端点）
- `OPENAI_MODEL`: 使用的模型名称（默认 gpt-4o）

在 `config.py` 中可调整：

- `MAX_PAPERS_TO_SEARCH`: 每次搜索最大论文数（默认 50）
- `SAMPLE_INCLUDE_THRESHOLD`: 样例筛选 include 阈值（默认 3）
- `MAX_SATISFIED_PAPERS`: 达标论文最大数量（默认 10）
- `CONTRACTOR_CONTEXT_ROUNDS`: Contactor 记住的对话轮数（默认 3）

## 测试方式

使用 LLM 模拟 10 个硕士学生与系统对话，每个学生有预设的 research question 和背景人设（在 system prompt 中），最终由学生对筛选结果打分评估。
