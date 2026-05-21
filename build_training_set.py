"""从 executor_prompt_record.txt, CSMeD-FT-gold_standard.json 构造训练集 JSONL 文件。

格式: {"messages": [
    {"role": "system", "content": <SCREEN_SYSTEM_PROMPT + 用户需求偏好>},
    {"role": "user", "content": <单篇论文标题+摘要>},
    {"role": "assistant", "content": "included"/"excluded"}
]}

通过 review_id + document_id 关联 gold_standard 的 decision。
跳过 system_prompt 或 user_prompt 为空的记录。
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSMED_DATA_DIR = r"D:\School\SAT404\pilot_study_code\systematic-review-datasets\data\CSMeD\CSMeD-FT\CSMeD-FT"

# ── 1. 读取 executor_prompt_record.txt ──
prompt_path = os.path.join(PROJECT_ROOT, "data_csmed_prompt_export", "executor_prompt_record.txt")
prompt_records = []
skip_empty = 0
total_lines = 0

with open(prompt_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total_lines += 1
        record = json.loads(line)
        sp = record.get("system_prompt", "")
        up = record.get("user_prompt", "")
        # 跳过旧格式（无 system_prompt/user_prompt 字段）或空内容
        if not sp or not up:
            skip_empty += 1
            continue
        prompt_records.append(record)

print(f"executor_prompt_record.txt: {total_lines} lines, {skip_empty} skipped (empty/old format), {len(prompt_records)} valid records")

# ── 2. 读取 CSMeD-FT-gold_standard.json ──
gs_path = os.path.join(CSMED_DATA_DIR, "CSMeD-FT-gold_standard.json")
with open(gs_path, "r", encoding="utf-8") as f:
    gold_list = json.load(f)

# Build lookup: (review_id, document_id) -> decision
gold_lookup = {}
for item in gold_list:
    key = (item["review_id"], item["document_id"])
    gold_lookup[key] = item.get("decision", "")

print(f"gold_standard: {len(gold_list)} records, {len(gold_lookup)} unique (review_id, document_id) pairs")

# ── 3. 构建训练集 ──
examples = []
no_decision = 0

for record in prompt_records:
    rid = record["review_id"]
    did = record.get("document_id", "")

    # 查找 gold standard decision
    decision = gold_lookup.get((rid, did))
    if not decision:
        no_decision += 1
        continue

    example = {
        "messages": [
            {"role": "system", "content": record["system_prompt"]},
            {"role": "user", "content": record["user_prompt"]},
            {"role": "assistant", "content": decision},
        ]
    }
    examples.append(example)

print(f"\n构建统计:")
print(f"  有效 prompt 记录: {len(prompt_records)}")
print(f"  无匹配 decision: {no_decision}")
print(f"  最终训练样本: {len(examples)}")

# ── 4. Decision 分布 ──
decision_counts = {}
for ex in examples:
    d = ex["messages"][2]["content"]
    decision_counts[d] = decision_counts.get(d, 0) + 1
print(f"\nDecision 分布:")
for d, c in sorted(decision_counts.items(), key=lambda x: -x[1]):
    print(f"  {d}: {c}")

review_ids_in_examples = set()
for record in prompt_records:
    if gold_lookup.get((record["review_id"], record.get("document_id", ""))):
        review_ids_in_examples.add(record["review_id"])
print(f"涉及的 review 数: {len(review_ids_in_examples)}")

# ── 5. 写入 JSONL ──
output_path = os.path.join(PROJECT_ROOT, "data_csmed_prompt_export", "training_set.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\n训练集已保存到: {output_path}")
print(f"共 {len(examples)} 条记录")
