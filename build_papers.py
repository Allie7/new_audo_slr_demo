"""Build deduplicated paper dataset from CSMeD-FT splits."""
import pandas as pd
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = r'D:\School\SAT404\pilot_study_code\systematic-review-datasets\data\CSMeD\CSMeD-FT\CSMeD-FT'

# Read all splits
dfs = []
for split in ['sample', 'dev', 'test', 'train']:
    path = os.path.join(DATA_DIR, f'CSMeD-FT-{split}.csv')
    df = pd.read_csv(path)
    n_unique = df['document_id'].nunique()
    print(f'{split}: {df.shape[0]} rows, {df.shape[1]} cols, unique document_ids={n_unique}')
    dfs.append(df)

# Merge
all_df = pd.concat(dfs, ignore_index=True)
print(f'\nTotal before dedup: {all_df.shape[0]} rows')
print(f'Unique document_ids: {all_df["document_id"].nunique()}')

# Deduplicate by document_id, keep first occurrence
dedup_df = all_df.drop_duplicates(subset='document_id', keep='first')
print(f'After dedup by document_id: {dedup_df.shape[0]} rows')

# Drop review-specific columns
drop_cols = ['review_id', 'decision', 'reason_for_exclusion']
paper_df = dedup_df.drop(columns=drop_cols)
print(f'Columns after dropping {drop_cols}: {list(paper_df.columns)}')
print(f'Final shape: {paper_df.shape}')

# Quick stats
print(f'\nNon-null counts:')
for col in paper_df.columns:
    n = paper_df[col].notna().sum()
    print(f'  {col}: {n}/{len(paper_df)}')

# Save JSON
out_json = os.path.join(DATA_DIR, 'CSMeD-FT-papers.json')
paper_df.to_json(out_json, orient='records', force_ascii=False, indent=2)
print(f'\nSaved JSON to: {out_json}')

# Save CSV
out_csv = os.path.join(DATA_DIR, 'CSMeD-FT-papers.csv')
paper_df.to_csv(out_csv, index=False, encoding='utf-8')
print(f'Saved CSV to: {out_csv}')

# Show sample
print(f'\n=== First paper (truncated) ===')
row = paper_df.iloc[0]
for col in paper_df.columns:
    val = str(row[col])[:150]
    print(f'  {col}: {val}')
