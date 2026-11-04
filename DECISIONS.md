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
