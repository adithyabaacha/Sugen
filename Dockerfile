FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The ICD-11 catalog ships with the image.
# No model downloads during grading.
COPY src/ ./src/
COPY data/mms-2026-01.json ./data/mms-2026-01.json

ENTRYPOINT ["python", "-m", "src.main"]
