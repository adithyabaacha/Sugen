FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only PyTorch
RUN pip install --no-cache-dir torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download and save model during build
COPY scripts/download_cross_encoder.py \
     ./scripts/download_cross_encoder.py

RUN python scripts/download_cross_encoder.py

# Disable runtime downloads
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ICD_CE_MODEL_PATH=/app/models/cross_encoder

COPY src/ ./src/
COPY data/mms-2026-01.json ./data/mms-2026-01.json

ENTRYPOINT ["python", "-m", "src.main"]
