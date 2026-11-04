"""Verify one-to-one output completeness, distinct valid codes, and JSONL shape."""
import argparse
import json
from src.catalog import ICDCatalog

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--pred', required=True)
parser.add_argument('--mms', default='data/mms-2026-01.json')
args = parser.parse_args()
catalog = ICDCatalog(args.mms)

def load_jsonl(path):
    with open(path, encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]

inputs = load_jsonl(args.input)
outputs = load_jsonl(args.pred)
ids = [row['id'] for row in inputs]
output_ids = [row['id'] for row in outputs]
errors=[]
if len(set(ids)) != len(ids): errors.append('Input contains duplicate ids')
if len(set(output_ids)) != len(output_ids): errors.append('Output contains duplicate ids')
if set(ids) != set(output_ids): errors.append(f'Input and output ids differ, missing={sorted(set(ids) - set(output_ids))[:10]}')
for r in outputs:
    codes=r.get('codes')
    if not isinstance(codes,list) or not 1 <= len(codes) <= 3:
        errors.append(f'{r["id"]}: expected 1-3 codes')
        continue
    if len(codes) != len(set(codes)): errors.append(f'{r["id"]}: duplicate code')
    for c in codes:
        if not catalog.is_valid(c): errors.append(f'{r["id"]}: invalid code {c!r}')
print(f'Input notes: {len(inputs)}, output rows: {len(outputs)}, errors: {len(errors)}')
for error in errors[:20]: print('FAIL:',error)
if errors: raise SystemExit(1)
print('PASS: every input id has exactly one prediction with 1-3 distinct valid codes')
