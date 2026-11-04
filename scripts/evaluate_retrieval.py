"""Diagnostic-only Recall@K; outputs do not use gold during inference."""
import json
from collections import defaultdict
from src.catalog import ICDCatalog
from src.retrieval import LexicalRetriever

notes=[json.loads(s) for s in open('data/dev_notes.jsonl', encoding='utf-8') if s.strip()]
model=LexicalRetriever(ICDCatalog('data/mms-2026-01.json'))
metrics=defaultdict(int)
errors=[]
for n in notes:
    preds=model.rank(n['text'],100)
    for k in (1,3,10,30,50,100):
        metrics[k] += bool(set(n['gold']) & set(preds[:k]))
    if not set(n['gold']) & set(preds[:50]):
        errors.append((n['id'],n['gold'],preds[:3]))
print('Candidate recall:')
for k in (1,3,10,30,50,100):
    print(f'  @{k:>3}: {metrics[k]}/{len(notes)} = {metrics[k]/len(notes):.1%}')
print('Misses at @50:', errors)
