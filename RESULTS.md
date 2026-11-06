# Results

This document summarizes the experiments I ran while developing the ICD-11 coding system.

All reported development scores were calculated using the supplied `score.py` and the 60 provided consultation notes. The development score is out of 75 points. These results do not establish performance on the 600 hidden consultations.

## Day 1 — Offline Baseline

I started with a retrieval-based approach using BM25 and word/character TF-IDF over the provided ICD-11 catalog.

The goal was to establish a working solution without depending on an external LLM API.

### Development Results

| Metric | Result |
|---|---:|
| Score | 31.2/75 |
| Top-1 accuracy | 35.0% (21/60) |
| Top-3 accuracy | 55.0% (33/60) |
| Missing predictions | 0 |
| Invalid codes | 0 |
| Duplicate IDs | 0 |

### Retrieval Performance

| Metric | Recall |
|---|---:|
| Recall@10 | 66.7% |
| Recall@30 | 81.7% |
| Recall@50 | 90.0% |
| Recall@100 | 95.0% |

The correct code was among the top 50 candidates for 54 of 60 notes and among the top 100 for 57 of 60 notes.

This suggested that candidate retrieval was reasonably effective, while choosing the best code remained the main problem.

### Docker and Offline Testing

The original Docker image built successfully and processed all 60 notes using Docker Compose.

I also tested the initial implementation with networking disabled using `--network none`. All predictions were generated, and output validation passed.

The initial test suite passed: **3/3 tests**.

## Day 2 — Cross-Encoder Experiment

I tested `cross-encoder/ms-marco-MiniLM-L6-v2` over the top 30 lexical candidates to see whether contextual relevance ranking would improve code selection.

| Method | Top-1 | Top-3 | Score |
|---|---:|---:|---:|
| Lexical baseline | 35.0% | 55.0% | 31.2/75 |
| Cross-encoder (0.25) | 33.3% | 50.0% | 29.2/75 |
| Cross-encoder (0.50) | 31.7% | 43.3% | 26.7/75 |
| Cross-encoder (0.75) | 28.3% | 40.0% | 24.2/75 |

All cross-encoder runs successfully processed 60 notes without inference errors.

Increasing its ranking weight consistently reduced accuracy.

I decided not to use this model as the primary reranker and moved to LLM-based clinical reasoning.

## Day 2 — Groq LLM Reranking

I added support for Groq through the OpenAI-compatible Python SDK so I could experiment with a free-tier LLM API.

The first runs fell back to lexical retrieval because Docker was still selecting OpenAI without an API key. After correcting the environment configuration, the five-note Groq test completed successfully.

### Full Development Run

Model: `openai/gpt-oss-20b`

| Metric | Result |
|---|---:|
| Score | 45.4/75 |
| Top-1 accuracy | 55.0% (33/60) |
| Top-3 accuracy | 71.7% (43/60) |
| Successful LLM requests | 29 |
| Failed requests | 16 |
| Notes processed after LLM disabled | 15 |
| Missing predictions | 0 |
| Invalid codes | 0 |
| Estimated API cost | $0.0048 |

The run encountered 15 rate-limit errors and one validation error. After repeated failures, the program disabled further LLM requests and used lexical fallback.

Despite this, the score improved substantially over the baseline.

The result demonstrated the value of LLM reranking, but the failed requests meant this was a mixed LLM-and-fallback evaluation, not a fully successful 60-note LLM run.

Groq was retained as a development experiment rather than a grading dependency.

## Day 2 — GPT-4.1 Mini, 50 Candidates

I switched the main reranker to OpenAI GPT-4.1 mini and supplied the top 50 lexical candidates for each note.

| Metric | Result |
|---|---:|
| Score | 61.7/75 |
| Top-1 accuracy | 80.0% (48/60) |
| Top-3 accuracy | 86.7% (52/60) |
| Successful LLM requests | 58 |
| Failed requests | 2 |
| Missing predictions | 0 |
| Invalid codes | 0 |
| Estimated API cost | $0.0407 |

The two failures were `ValueError` exceptions and were handled by lexical fallback.

This was the largest accuracy improvement during development. It confirmed that a stronger LLM could select appropriate diagnoses much more reliably than the original lexical ranking.

## Day 2 — GPT-4.1 Mini, 100 Candidates

I increased the candidate count from 50 to 100 because retrieval recall was higher at 100.

### Initial Run

| Metric | Result |
|---|---:|
| Score | 64.6/75 |
| Top-1 accuracy | 85.0% (51/60) |
| Top-3 accuracy | 88.3% (53/60) |
| Successful LLM requests | 59 |
| Failed requests | 1 |
| Estimated API cost | $0.0731 |

### Repeated Timed Run

I repeated the experiment to verify runtime and API reliability.

| Metric | Result |
|---|---:|
| Score | 65.0/75 |
| Top-1 accuracy | 85.0% (51/60) |
| Top-3 accuracy | 90.0% (54/60) |
| Successful LLM requests | 60/60 |
| Failed requests | 0 |
| Runtime | 58.156 seconds |
| Estimated API cost | $0.0731 |

The additional candidates improved accuracy over the 50-candidate version.

The small difference between the two 100-candidate runs also showed that LLM predictions may vary across repeated requests.

I selected **100 candidates** for subsequent experiments.

## Day 3 — Structured Output and Reliability Improvements

I hardened the LLM implementation to address occasional output validation failures.

Changes included strict JSON schema output for OpenAI, local validation of returned codes, specific failure categories, retries for recoverable errors, bounded backoff and lexical fallback.

### Test Results

| Check | Result |
|---|---:|
| Python tests | 21 passed |
| Docker build | Passed |
| Notes processed | 60 |
| API attempts | 61 |
| Successful note-level predictions | 60 |
| Retries | 1 |
| Failed notes | 0 |
| Missing predictions | 0 |
| Invalid codes | 0 |

### Development Accuracy

| Metric | Result |
|---|---:|
| Score | 64.6/75 |
| Top-1 accuracy | 85.0% (51/60) |
| Top-3 accuracy | 88.3% (53/60) |
| Estimated API cost | $0.0858 |

The retry mechanism worked: 61 API attempts produced successful results for all 60 notes.

The score was slightly below the earlier 65.0/75 run, so I consider this a reliability improvement rather than a demonstrated accuracy improvement.

## Day 3 — Diagnosis Normalization Experiment

I introduced an optional normalization stage to improve retrieval for non-English clinical terms, abbreviations and unfamiliar consultation wording.

The normalizer extracts a short English diagnosis search query. It does not replace the original consultation during final code selection.

The retriever combines original candidates with up to 20 additional candidates from the normalized query.

### Controlled Comparison

Both configurations used the same hardened code and 100-candidate limit.

| Metric | Normalization off | Normalization on |
|---|---:|---:|
| Score | 64.6/75 | **65.4/75** |
| Top-1 accuracy | 85.0% (51/60) | **86.7% (52/60)** |
| Top-3 accuracy | 88.3% (53/60) | 88.3% (53/60) |
| Missing predictions | 0 | 0 |
| Invalid codes | 0 | 0 |

The normalized pipeline completed 60 normalization requests and 60 final coding requests, all successfully.

### Error Analysis

I compared `predictions_hardened.jsonl` with `predictions_normalized.jsonl`.

Two notes changed their scoring outcomes:

| Note | Gold code | Normalization off | Normalization on |
|---|---|---|---|
| `dev-031` | `GA90` | Correct in top 3 | Missing from top 3 |
| `dev-060` | `5B81.01` | Missing from top 3 | Correct at rank 1 |

For `dev-031`, normalization removed a useful candidate.

For `dev-060`, the extra retrieval query brought the correct code into the first position.

These outcomes explain the overall improvement of **0.8 points compared with the hardened version without normalization**.

The experiment shows that normalization can help, but it can also displace valid codes. The gain is too small to prove better generalization on unseen data.

### Repeated Normalized Run

I scored the separate timed normalized output to check whether the result was reproducible.

| Metric | Result |
|---|---:|
| Score | 65.4/75 |
| Top-1 accuracy | 86.7% (52/60) |
| Top-3 accuracy | 88.3% (53/60) |
| Missing predictions | 0 |
| Invalid codes | 0 |

The repeated run produced the same aggregate score.

## Day 3 — Runtime and API Cost

I measured the complete application, including Docker startup and API execution.

| Metric | Without normalization | With normalization |
|---|---:|---:|
| Runtime for 60 notes | 58.156 seconds | 103.418 seconds |
| LLM requests | 60 | 120 |
| Estimated API cost | $0.0731 | $0.0919 |
| Estimated runtime for 600 similar notes | 9m 42s | 17m 14s |
| Estimated cost for 600 similar notes | $0.73 | $0.92 |

For the normalized configuration, the estimated 60-note cost consisted of:
- Code selection: **$0.0838**
- Diagnosis normalization: **$0.0081**
- Total: **$0.0919**

The 600-note figures are linear projections, not actual 600-note benchmarks.

The assessment allows 60 minutes and a total API budget of $6. The measured development runs suggest both configurations are feasible, but actual hidden-note length, API rate limits, retries and network latency may change the result.

The cost figures come from local token-based estimates, not confirmed billing totals.

## Day 3 — Slim Docker Image

I removed the unused PyTorch, Transformers and cross-encoder model-download steps from the final Docker build.

The image now contains the Python application, lexical retrieval dependencies, OpenAI SDK and the supplied ICD-11 catalog.

| Check | Result |
|---|---|
| Docker build | Passed |
| Build duration | Approximately 61 seconds |
| PyTorch dependency | Removed |
| Runtime cross-encoder download | Not required |
| Python tests | 21/21 passed |

The earlier cross-encoder image had taken approximately 229 seconds to build. The new build was substantially shorter in the observed runs, although the environments and caching conditions were not controlled for a direct speed comparison.

The smaller image reduces unnecessary dependencies and avoids downloading a model that did not improve accuracy.

## Offline and No-Key Fallback

I tested the final hardened application with `OPENAI_API_KEY` set to an empty value.

The LLM was disabled automatically and lexical retrieval handled the input.

```text
Wrote 60 predictions
LLM state=OPENAI_API_KEY missing
calls=0, success=0, failed=0
```

Output validation:

| Check | Result |
|---|---|
| Input notes | 60 |
| Output rows | 60 |
| Missing predictions | 0 |
| Invalid codes | 0 |
| Duplicate IDs | 0 |
| Validation | PASS |

The previously measured lexical fallback scored **31.2/75**.

The latest hardened no-key output was validated successfully but was not separately scored.

This is important because the system can still return valid predictions if the API key is missing or unavailable.

## Overall Experiment Comparison

| Approach | Top-1 | Top-3 | Score / 75 |
|---|---:|---:|---:|
| Lexical retrieval | 35.0% | 55.0% | 31.2 |
| Cross-encoder, weight 0.25 | 33.3% | 50.0% | 29.2 |
| Cross-encoder, weight 0.50 | 31.7% | 43.3% | 26.7 |
| Cross-encoder, weight 0.75 | 28.3% | 40.0% | 24.2 |
| Groq GPT-OSS-20B + fallback | 55.0% | 71.7% | 45.4 |
| GPT-4.1 mini, 50 candidates | 80.0% | 86.7% | 61.7 |
| GPT-4.1 mini, 100 candidates | 85.0% | 88.3% | 64.6 |
| GPT-4.1 mini, 100 candidates, successful repeat | 85.0% | 90.0% | 65.0 |
| Hardened GPT-4.1 mini, normalization off | 85.0% | 88.3% | 64.6 |
| **Hardened GPT-4.1 mini, normalization on** | **86.7%** | **88.3%** | **65.4** |

## Current Best Result

The highest measured development score is **65.4/75**, achieved with:

- GPT-4.1 mini
- BM25 and TF-IDF candidate retrieval
- 100 candidate codes
- Optional diagnosis normalization enabled
- Up to 20 candidates introduced from normalized retrieval
- Strict JSON schema and code validation
- Bounded retries and lexical fallback

This configuration achieved **52/60 top-1 correct** and **53/60 top-3 correct**, with no missing predictions or invalid codes.

Normalization produced a small improvement over the hardened configuration without normalization. The earlier non-hardened 100-candidate run achieved 65.0/75 and a slightly higher top-3 accuracy of 90%.

## Remaining Limitations

The strongest measured scores are still from the supplied 60-note development set.

There are several remaining uncertainties:

1. **Hidden distribution:** The assessment includes consultation notes outside the development distribution. Performance on those notes has not been measured.
2. **Normalization mistakes:** Generated search terms can help retrieval but can also displace correct candidates.
3. **API dependency:** The high-accuracy path depends on OpenAI being reachable. The offline fallback preserves output validity but is less accurate.
4. **Runtime extrapolation:** The full 600-note grading runtime has not been measured. The 60-note runs suggest sufficient headroom.
5. **Development-set tuning:** Multiple experiments were compared using the same 60 notes, so small score gains may not generalize.

## Final Status and Next Steps

Completed:
- Offline retrieval baseline.
- Cross-encoder comparison.
- Free-tier and OpenAI reranking experiments.
- Candidate-count evaluation.
- Structured response validation and retries.
- Optional diagnosis normalization.
- Python tests and Docker builds.
- No-key fallback validation.
- Timed 60-note API runs.

Before submission:
- Decide whether normalization should be enabled by default, ideally using independent unseen-style examples.
- Confirm the exact required grading command from a clean clone.
- Ensure the submitted repository contains no API credentials.
- Finalize `README.md`, `DECISIONS.md` and other required documentation.
- Create and verify the Git bundle.

The core pipeline is implemented and tested. Further changes should focus on reproducibility, generalization and submission reliability rather than optimizing repeatedly against the same development set.
