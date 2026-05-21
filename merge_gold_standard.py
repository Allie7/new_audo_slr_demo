"""Merge all CSMeD-FT CSV splits, deduplicate, and build gold standard file."""
import pandas as pd
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = r'D:\School\SAT404\pilot_study_code\systematic-review-datasets\data\CSMeD\CSMeD-FT\CSMeD-FT'

splits = ['sample', 'dev', 'test', 'train']
dfs = []
for split in splits:
    path = os.path.join(DATA_DIR, f'CSMeD-FT-{split}.csv')
    df = pd.read_csv(path)
    print(f'{split}: {df.shape[0]} rows')
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
print(f'\nTotal before dedup: {all_df.shape[0]} rows')

# Deduplicate by (review_id, document_id) — same paper in same review should be unique
dedup_df = all_df.drop_duplicates(subset=['review_id', 'document_id'], keep='first')
dup_count = len(all_df) - len(dedup_df)
print(f'Duplicates removed: {dup_count}')
print(f'After dedup: {dedup_df.shape[0]} rows')

# Keep only gold standard columns
gold = dedup_df[['review_id', 'document_id', 'decision', 'reason_for_exclusion']].copy()
print(f'\nGold standard shape: {gold.shape}')
print(f'Columns: {list(gold.columns)}')

# Stats
print(f'\nDecision distribution:')
print(gold['decision'].value_counts().to_string())
print(f'\nUnique reviews: {gold["review_id"].nunique()}')
print(f'Unique documents: {gold["document_id"].nunique()}')

# Per-review stats
review_stats = gold.groupby('review_id')['decision'].value_counts().unstack(fill_value=0)
print(f'\nPer-review paper count:')
print(gold.groupby('review_id').size().describe().to_string())

# Save
out_csv = os.path.join(DATA_DIR, 'CSMeD-FT-gold_standard.csv')
gold.to_csv(out_csv, index=False, encoding='utf-8')
print(f'\nSaved CSV to: {out_csv}')

out_json = os.path.join(DATA_DIR, 'CSMeD-FT-gold_standard.json')
gold.to_json(out_json, orient='records', force_ascii=False, indent=2)
print(f'Saved JSON to: {out_json}')

# Show sample
print(f'\n=== Sample rows ===')
print(gold.head(10).to_string())
