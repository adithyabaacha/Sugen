# Engineering Decision Log

This log records the main decisions I made while building and evaluating the ICD-11 coding system. I started by getting a working offline implementation in place before introducing any external model dependencies.

The Day 1 entries below document the decisions reflected in the initial implementation. I'll add new entries as I make changes and run further experiments.

## D01 — Starting with an offline baseline

**Date/time:**  04/11/2026  
**Commit:** 6dd646dabb619ed61ffea39658b555919541e564

I decided to build a local retrieval-based solution first instead of starting directly with an LLM.

An LLM-only approach would have been quicker to prototype, but I wanted a working system that didn't depend on API availability, latency, or usage limits. The assessment also requires predictions when no API key is provided.

The initial implementation runs completely offline. It processed all 60 development notes and returned valid predictions for every input.

The limitation is accuracy. The baseline scored 31.2/75, so I plan to keep it as a fallback while working on a better ranking approach.

## D02 — Testing local cross-encoder reranking

**Date/time:** 05/11/2026

**Commit:** 734e180e6f57ab3b5ef0780ba8ad702e6b1c7408

I tried `cross-encoder/ms-marco-MiniLM-L6-v2` to improve the ranking of the top 30 ICD-11 candidates returned by lexical retrieval.

The model ran successfully on CPU, but the results were worse than the original baseline.

| Cross-encoder weight | Score |
|---|---|
| 0.00 (lexical only) | 31.2/75 |
| 0.25 | 29.2/75 |
| 0.50 | 26.7/75 |
| 0.75 | 24.2/75 |

Lowering the weight helped, but none of the configurations outperformed lexical retrieval.

My conclusion is that the general-purpose cross-encoder isn't a good fit for selecting clinically specific ICD-11 diagnoses from short category titles.

I've kept lexical retrieval as the default and decided to explore LLM-based clinical reranking instead. With more time, I'd evaluate a medically trained reranker using richer ICD-11 descriptions.
