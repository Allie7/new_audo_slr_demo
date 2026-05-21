"""Merge all CSMeD-FT reviews_metadata.json files and deduplicate by review_id."""
import json
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = r'D:\School\SAT404\pilot_study_code\systematic-review-datasets\data\CSMeD\CSMeD-FT\CSMeD-FT'

splits = ['sample', 'dev', 'test', 'train']
merged = {}
stats = {}

for split in splits:
    path = os.path.join(DATA_DIR, f'CSMeD-FT-{split}_reviews_metadata.json')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        keys = list(data.keys())
        stats[split] = len(keys)
        # Track duplicates
        new_keys = [k for k in keys if k not in merged]
        dup_keys = [k for k in keys if k in merged]
        print(f'{split}: {len(keys)} reviews, {len(new_keys)} new, {len(dup_keys)} duplicates')
        
        for k in new_keys:
            merged[k] = data[k]
    elif isinstance(data, list):
        stats[split] = len(data)
        new_count = 0
        dup_count = 0
        for item in data:
            rid = item.get('review_id', '')
            if rid and rid not in merged:
                merged[rid] = item
                new_count += 1
            else:
                dup_count += 1
        print(f'{split}: {len(data)} reviews (list format), {new_count} new, {dup_count} duplicates')

print(f'\nTotal after merge & dedup: {len(merged)} unique reviews')
print(f'Per split totals: {stats}')

# Save
out_path = os.path.join(DATA_DIR, 'CSMeD-FT-all_reviews_metadata.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
print(f'\nSaved to: {out_path}')

# Show sample entry
first_key = list(merged.keys())[0]
print(f'\n=== Sample entry (key={first_key}) ===')
entry = merged[first_key]
print(f'Keys: {list(entry.keys())}')
for k, v in entry.items():
    val_str = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    print(f'  {k}: {val_str[:200]}')
