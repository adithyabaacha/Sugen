#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo '[1/6] Check Docker daemon and Compose'
docker info >/dev/null
docker compose version

echo '[2/6] Build image from Dockerfile'
docker compose build --no-cache coder

echo '[3/6] Prepare dev input (requires local Python; if unavailable, skip this line and provide in/notes.jsonl)'
if command -v python3 >/dev/null 2>&1; then
  python3 scripts/make_dev_input.py
fi
test -s in/notes.jsonl || { echo 'Missing in/notes.jsonl'; exit 1; }

echo '[4/6] Run the exact grader command WITHOUT model API keys'
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY docker compose run --rm coder /in/notes.jsonl /out/predictions.jsonl

echo '[5/6] Verify prediction completeness and official development score'
python3 -m scripts.verify_output --input in/notes.jsonl --pred out/predictions.jsonl
python3 score.py --gold data/dev_notes.jsonl --pred out/predictions.jsonl --mms data/mms-2026-01.json

echo '[6/6] Explicitly DISABLE networking (Linux only; docker run not docker compose)'
image="$(docker compose images -q coder)"
test -n "$image"
docker run --rm --network none --mount "type=bind,src=$(pwd)/in,dst=/in,readonly" --mount "type=bind,src=$(pwd)/out,dst=/out" "$image" /in/notes.jsonl /out/no_network_predictions.jsonl
python3 -m scripts.verify_output --input in/notes.jsonl --pred out/no_network_predictions.jsonl

echo 'PASS: Docker test and no-network offline fallback complete.'
