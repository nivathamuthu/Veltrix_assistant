FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OCR (tesseract) and PDF page rasterising (poppler) for scanned documents.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Models are fetched ONCE here, in their own layer (before the app code, so a
# code change does not re-download them). The running container loads them
# from these folders with no network access.
ENV EMBED_MODEL=BAAI/bge-small-en-v1.5 \
    EMBED_CACHE_DIR=/opt/models \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken
COPY scripts/download_models.py scripts/download_models.py
RUN python scripts/download_models.py

# From here on nothing may reach out to the model hub.
ENV HF_HUB_OFFLINE=1

COPY app ./app

# Writable location for the vector index (rebuilt at every startup).
RUN mkdir -p /app/index

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
