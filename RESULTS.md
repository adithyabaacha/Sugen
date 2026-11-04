
# Results

## Day 1 — Offline Baseline

I started with a retrieval-based approach using BM25 and word/character TF-IDF over the provided ICD-11 catalog. The idea was to get a working baseline without depending on an external LLM API.

### Development Results

Evaluated on the 60 provided consultation notes using `score.py`.

| Metric | Result |
|---|---|
| Score | 31.2 / 75 |
| Top-1 accuracy | 35% (21/60) |
| Top-3 accuracy | 55% (33/60) |
| Missing predictions | 0 |
| Invalid codes | 0 |

### Retrieval Performance

| Metric | Recall |
|---|---|
| Recall@10 | 66.7% |
| Recall@30 | 81.7% |
| Recall@50 | 90.0% |
| Recall@100 | 95.0% |

The interesting part is that the correct code is already among the top 50 candidates for 54 out of 60 notes, but only 21 are ranked first.

This suggests that retrieval is doing a reasonable job of finding relevant codes, while selecting the most appropriate one still needs improvement.

### Docker and Offline Testing

The Docker image built successfully, and the application processed all 60 notes using the required Docker Compose command.

I also tested it with networking completely disabled (`--network none`). All 60 predictions were generated, and output validation passed without errors.

The initial Python tests also passed (3/3).

### Next Steps

The main focus now is improving top-1 accuracy. I plan to try LLM-based reranking over the retrieved candidates and improve clinical diagnosis extraction, especially for negations, uncertain diagnoses, and code specificity.

I'll compare the changes against this baseline before deciding which ones to keep.
