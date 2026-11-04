"""Generate a synthetic 600-item performance workload by repeating dev text with NEW IDs.
Performance only; NOT an independent evaluation set and not a hidden-score estimate.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', default='in/notes.jsonl')
parser.add_argument('--out', default='in/benchmark_600.jsonl')
parser.add_argument('--count', type=int, default=600)
args = parser.parse_args()
notes = [json.loads(x) for x in Path(args.source).read_text(encoding='utf-8').splitlines() if x.strip()]
assert len(notes) and args.count > 0
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, 'w', encoding='utf-8') as stream:
    for i in range(args.count):
        original = notes[i % len(notes)]
        stream.write(json.dumps({'id': f'bench-{i+1:04d}', 'text': original['text']}, ensure_ascii=False)+'\n')
print(f'Wrote {args.count} performance-only input notes to {args.out}')
