# ICD-11 Outpatient Coding

A Dockerized ICD-11 MMS coder that reads consultation notes and returns up to three valid diagnostic codes per note. It uses local lexical retrieval (BM25 plus word/character TF-IDF), optional LLM diagnosis normalization, and GPT-4.1 mini to select the most specific supported codes. Every result is checked against the supplied MMS catalog; API or validation failures fall back to lexical candidates.

## Reproduce

```bash
python -m scripts.make_dev_input
docker compose build
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
python -m scripts.verify_output --input in/notes.jsonl --pred out/predictions.jsonl
python score.py --gold data/dev_notes.jsonl --pred out/predictions.jsonl --mms data/mms-2026-01.json
```

The grader supplies `OPENAI_API_KEY` and `HTTPS_PROXY` as environment variables. **No `.env` file is required.** Without an API key, the same command returns valid but less accurate lexical predictions. See `DECISIONS.md` and `RESULTS.md` for detailed experiments.

## Results and estimate

**Best measured development score: 65.4/75** (60 provided notes; **52/60 top-1**, **53/60 top-3**), with GPT-4.1 mini, 100 retrieved candidates and normalization enabled. With normalization disabled, the hardened pipeline scored **64.6/75**. The no-key lexical baseline scored **31.2/75**. The normalized 60-note run took **103.4 seconds** with 120 successful API calls and an **estimated $0.0919** of token usage cost.

**My hidden-score estimate: approximately 70/100, with substantial uncertainty** (roughly 58/75 core plus 12/25 generalization). The 60-note development set is narrow; neither hidden accuracy nor multilingual accuracy has been measured. This is a judgment, not a benchmark result.

## Data, evaluation and AI-tooling error

I used the **assessment-supplied** ICD-11 MMS 2026-01 code/title catalog and development notes. I added a small manually written abbreviation/synonym glossary (`src/clinical.py`) and an optional runtime diagnosis-query normalizer (`src/note_normalizer.py`). **No external WHO terminology files, pretrained weights or additional labeled patient notes were added.**

Beyond development scoring, I ran **21 automated tests**, a successful Docker build, timed end-to-end API runs, and a no-key fallback test producing **60/60 valid outputs**. An earlier fully network-disabled baseline run also passed. These are reliability tests, **not independent labeled generalization accuracy**.

An AI-assisted proposal to use a generic MS MARCO cross-encoder sounded promising, but measured scores fell to **24.2/75** at weight 0.75 versus **31.2/75** for lexical retrieval. I caught this through controlled scoring, rejected that approach and removed its large runtime dependencies.

**Time spent:** 17 hours.
