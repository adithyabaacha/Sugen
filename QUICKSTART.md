
# Quickstart

This project reads consultation notes in JSONL format and returns up to three ICD-11 codes for each note.

The current baseline works entirely offline. It does not require an OpenAI or Anthropic API key.

## Requirements

- Python 3.11
- Docker with Docker Compose
- A Linux container environment

## Running Locally

Install the development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Prepare the development input file:

```bash
python scripts/make_dev_input.py
```

Run the tests:

```bash
python -m pytest -q
```

Generate predictions:

```bash
python -m src.main in/notes.jsonl out/predictions.jsonl
```

The output is written to `out/predictions.jsonl`.

## Running with Docker

Build the image:

```bash
docker compose build
```

Run the command required by the assessment:

```bash
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
```

The container reads the input from `/in` and writes predictions to `/out`. Both paths are mapped to local directories through Docker Compose.

## Checking the Output

Validate that every input note has a prediction and that all returned codes exist in the supplied ICD-11 catalog:

```bash
python -m scripts.verify_output \
  --input in/notes.jsonl \
  --pred out/predictions.jsonl
```

Run the official development scorer:

```bash
python score.py \
  --gold data/dev_notes.jsonl \
  --pred out/predictions.jsonl \
  --mms data/mms-2026-01.json
```

The initial offline baseline scored 31.2/75 on the 60 supplied development notes.

## Running Without Network Access

The baseline can also run with networking disabled:

```bash
docker run --rm --network none \
  --entrypoint python \
  -v "$PWD/in:/in:ro" \
  -v "$PWD/out:/out" \
  sugen-coder:latest \
  -m src.main \
  /in/notes.jsonl \
  /out/predictions_offline.jsonl
```

This command was tested successfully. It generated predictions for all 60 notes without network access.

Validate the offline output:

```bash
python -m scripts.verify_output \
  --input in/notes.jsonl \
  --pred out/predictions_offline.jsonl
```

## Additional Checks

To run the retrieval evaluation:

```bash
python -m scripts.evaluate_retrieval
```

To generate a 600-note performance workload:

```bash
python -m scripts.make_bulk_input --count 600
```

Then run:

```bash
docker compose run --rm coder \
  /in/benchmark_600.jsonl \
  /out/benchmark_600_predictions.jsonl
```

The 600-note workload repeats the development texts with different IDs. It is intended for performance testing, not accuracy evaluation.

## Notes

The `in/` and `out/` directories are used for local input and generated predictions. Their JSONL files are ignored by Git.

The original assessment data and ICD-11 catalog are kept in `data/`.

Before submission, the project should also be built and tested from a fresh Git clone.
