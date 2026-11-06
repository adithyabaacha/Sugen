# Engineering Decision Log

This log records the main decisions I made while developing the ICD-11 coding system, including the alternatives I tried, their results, and why I kept or rejected them.

I started with a fully offline baseline before adding LLM-based reranking. Every major change was evaluated against the 60 provided development notes using `score.py`. I also tested output validity, Docker execution, API reliability, and runtime.

The dates and commit references for the first two decisions are from my original log. I need to fill in the exact dates and commits for the remaining entries from Git history before submission.

## D01 — Starting with an offline baseline

**Date:** 04/11/2026  
**Commit:** `6dd646dabb619ed61ffea39658b555919541e564`

I decided to build a local retrieval-based solution first instead of starting directly with an LLM.

An LLM-only approach would have been quicker to prototype, but I wanted a working system that didn't depend on API availability, latency, or usage limits. The assessment also requires predictions when no API key is provided.

I used BM25 and word/character TF-IDF over the provided ICD-11 catalog, with clinical text cleaning and diagnosis-focused retrieval.

The initial implementation processed all 60 development notes and returned valid predictions for every input.

Results:
- Development score: **31.2/75**
- Top-1 accuracy: **35% (21/60)**
- Top-3 accuracy: **55% (33/60)**
- Missing predictions: **0**
- Invalid codes: **0**
- Recall@50: **90%**
- Recall@100: **95%**

I also tested Docker execution without networking. All 60 predictions were generated, and output validation passed.

The main limitation was ranking accuracy. Retrieval often found the correct code without ranking it first.

**Decision:** Keep lexical retrieval as the foundation and offline fallback, and investigate second-stage reranking.

## D02 — Testing local cross-encoder reranking

**Date:** 05/11/2026  
**Commit:** `734e180e6f57ab3b5ef0780ba8ad702e6b1c7408`

I tested `cross-encoder/ms-marco-MiniLM-L6-v2` as a local CPU-based reranker over the top 30 lexical candidates.

I wanted to see whether contextual relevance scoring would improve code selection without relying on an external API.

| Cross-encoder weight | Score |
|---|---:|
| 0.00 — lexical only | 31.2/75 |
| 0.25 | 29.2/75 |
| 0.50 | 26.7/75 |
| 0.75 | 24.2/75 |

The cross-encoder successfully reranked all 60 notes without inference errors, but every tested configuration performed worse than lexical retrieval.

Increasing the cross-encoder weight consistently reduced accuracy.

My hypothesis is that a general-purpose passage reranker is not well suited to distinguishing clinically specific ICD-11 codes using short category titles.

**Decision:** Reject the cross-encoder as the primary reranker. Keep the experiment in Git history and move to LLM-based clinical reasoning.

**Revisit:** A medically trained reranker with richer ICD-11 descriptions might be worth evaluating with more time.

## D03 — Using Groq to test LLM reranking without OpenAI credits

**Date/time:** 05/11/2026  
**Commit:** 0cfa16d66692ffb1e155697382571a37df0a29c3

I initially wanted to test LLM reranking without paying for API usage, so I added an OpenAI-compatible Groq provider.

The first attempts still selected OpenAI because the Docker container wasn't receiving the provider settings. The logs showed `OPENAI_API_KEY missing` and zero LLM calls. Checking environment variables inside the container helped identify the configuration problem.

After fixing `.env`, the five-note Groq test completed with **5 successful calls and no failures**.

I then tested all 60 development notes using `openai/gpt-oss-20b` on Groq.

Results:
- Top-1 accuracy: **55.0%**
- Top-3 accuracy: **71.7%**
- Score: **45.4/75**
- Successful LLM requests: **29**
- Failed requests: **16**
- Notes processed after the LLM was disabled: **15**

Most failures were rate limits. One was an output validation error. Lexical fallback still produced predictions for every note.

This was a useful improvement over 31.2/75, but the free API rate limits made it unsuitable as a dependable production path.

**Decision:** Keep provider compatibility for development, but use OpenAI for the assessed solution. Groq cannot be relied on in the grading environment because its endpoint is not on the permitted runtime list.

## D04 — Switching to GPT-4.1 mini for clinical reranking

**Date/time:** 05/11/2026  
**Commit:** 0cfa16d66692ffb1e155697382571a37df0a29c3

After the Groq experiment, I tested GPT-4.1 mini using OpenAI's API.

The pipeline remained the same:

1. Retrieve ICD-11 candidates using BM25 and TF-IDF.
2. Supply the consultation note and candidate codes to the LLM.
3. Ask the model to select up to three appropriate codes.
4. Validate the returned codes against the catalog.
5. Use lexical fallback if the API or response validation fails.

I started with 50 candidates.

Results:
- Top-1 accuracy: **80.0%**
- Top-3 accuracy: **86.7%**
- Score: **61.7/75**
- Successful requests: **58/60**
- Failed requests: **2/60**
- Estimated API cost: **$0.0407**

The two failures were `ValueError` exceptions during response processing. Both were handled by lexical fallback.

The improvement over Groq was substantial, although the models were not tested under identical API-success conditions.

**Decision:** Use GPT-4.1 mini as the main coding model and retain lexical retrieval as the candidate generator and fallback.

## D05 — Increasing candidate retrieval from 50 to 100

**Date/time:** 05/11/2026   
**Commit:** 914c792c07fa9523cecfdd898f061bc5b291cda3

The retrieval analysis showed that the correct code appeared among the top 50 candidates for 54 of 60 notes and among the top 100 for 57 of 60 notes.

Since the LLM can select only from the supplied candidate list, I tested whether providing more candidates would improve accuracy.

| Configuration | Top-1 | Top-3 | Score |
|---|---:|---:|---:|
| GPT-4.1 mini, 50 candidates | 80.0% | 86.7% | 61.7/75 |
| GPT-4.1 mini, 100 candidates | 85.0% | 88.3% | 64.6/75 |

The first 100-candidate run had 59 successful API requests and one validation failure.

I repeated the 100-candidate test. All 60 requests succeeded, and the score improved to **65.0/75**, with 85% top-1 and 90% top-3 accuracy.

The successful run took **58.156 seconds** for 60 notes and had an estimated API cost of **$0.0731**.

**Decision:** Increase the candidate limit to 100. It gave a measurable improvement while remaining inexpensive and fast enough for the assessment.

**Trade-off:** Larger prompts could increase cost and sometimes introduce more confusing alternatives. I stopped at 100 rather than continually tuning the candidate count on the same development notes.

## D06 — Keeping a no-key and API-failure fallback

**Date/time:** 05/11/2026   
**Commit:** 914c792c07fa9523cecfdd898f061bc5b291cda3

I kept the lexical retrieval path available even after GPT-4.1 mini became the primary model.

This matters because the system must still produce valid predictions when no API key is provided or when individual API requests fail.

I explicitly ran the LLM configuration with `OPENAI_API_KEY` empty.

The program made zero LLM calls and still generated predictions for all 60 development notes.

Output validation reported:
- Input notes: **60**
- Output rows: **60**
- Missing predictions: **0**
- Invalid codes: **0**
- Duplicate IDs: **0**

The original fallback configuration scored **31.2/75**.

**Decision:** Never make successful API access a requirement for producing an output. The fallback is less accurate but ensures the application continues to satisfy the output contract.

## D07 — Hardening LLM responses and retries

**Date/time:** 05/11/2026   
**Commit:** 914c792c07fa9523cecfdd898f061bc5b291cda3

The initial GPT-4.1 mini runs occasionally failed with `ValueError`, even when the API request itself succeeded.

I improved the reranker by adding:
- OpenAI Structured Outputs with a strict JSON schema.
- Validation that returned codes belong to the supplied candidates.
- Specific validation failure categories without logging clinical notes or API keys.
- One retry for malformed or unusable model responses.
- Bounded retries and backoff for temporary API failures.
- Immediate fallback or disabling for permanent authentication and configuration errors.
- A limit on estimated API spending and execution time.

After the change, I ran the updated test suite. All **21 tests passed**.

The hardened 60-note experiment produced:
- **61 physical API attempts**
- **60 successful note-level predictions**
- **1 retry**
- **0 failed notes**
- **64.6/75 development score**
- **$0.0858 estimated API cost**

The score was slightly below the earlier successful 65.0/75 run, so I cannot claim the schema change improves accuracy. Its main value is reliability and more understandable failure handling.

**Decision:** Keep the hardened error handling, while tracking any ranking-quality differences from the previous version.


## D08 — Treating development accuracy and hidden accuracy separately

**Date/time:** 05/11/2026   
**Commit:** 914c792c07fa9523cecfdd898f061bc5b291cda3

The best development score reached **65.4/75**, but it was measured on the same 60 provided notes used to compare candidate limits and normalization.

I don't want to assume that this number represents accuracy on the 600 hidden consultations.

Repeated experiments against one small development set can favor configurations that happen to work well on those examples.

The assessment also includes a different-distribution slice, so unseen terminology and consultation structure remain important risks.

**Decision:** Stop making major changes solely to improve the 60-note development score. Prefer independently checked examples, robust fallbacks and correct grading behavior over further prompt tuning.

## D09 — Final submission configuration and checks

**Date/time:** 05/11/2026   
**Commit:** 914c792c07fa9523cecfdd898f061bc5b291cda3

The intended submission architecture is:

**Original consultation → optional diagnosis normalization → lexical retrieval → GPT-4.1 mini reranking → code validation → JSONL output.**

The lexical fallback remains available when the OpenAI API cannot be used.

The measured configuration is:
- GPT-4.1 mini
- Up to 100 ICD-11 candidates
- Optional normalization, with up to 20 new candidates
- Strict output validation and bounded retries
- Offline lexical fallback

Completed checks include:
- 21 passing Python tests.
- Successful build of the smaller Docker image.
- 60 valid outputs with no API key.
- Successful OpenAI reranking of all 60 notes in the timed runs.
- Measured normalized runtime below the assessment limit when extrapolated to 600 similar notes.

**Still to verify before submission:**
- Exact grading command from a clean clone.
- Final committed normalization setting.
- `.env` and API keys excluded from version control.
- Required documentation completed.
- Git bundle created and verified.

**Decision:** Freeze the implementation once the final configuration passes these checks, rather than adding more features near the deadline.
