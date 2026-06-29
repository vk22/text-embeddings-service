FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CLAP_CACHE_DIR=/app/.cache/clap
ENV HF_HOME=/app/.cache/clap
ENV TORCH_HOME=/app/.cache/clap
ENV XDG_CACHE_HOME=/app/.cache/clap
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV CLAP_TORCH_THREADS=1
ENV CLAP_TEXT_ONLY=1
ENV CLAP_PRUNE_AUDIO=1
ENV CLAP_DYNAMIC_QUANTIZE=0
ENV CLAP_MAX_TEXT_CHARS=1000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=600 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0 torchvision==0.26.0 \
 && pip install --no-cache-dir --default-timeout=600 -r requirements.txt

COPY app ./app

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1", "--limit-concurrency", "2", "--timeout-keep-alive", "5"]
