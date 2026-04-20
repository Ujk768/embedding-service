# Stage 1: Download and cache the model
FROM python:3.11-slim as model-downloader
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# This pre-downloads the specific model into the default FastEmbed cache
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding(model_name='BAAI/bge-small-en-v1.5').embed(['warmup']))"

# Stage 2: Final Image
FROM python:3.11-slim
WORKDIR /app

# Install libgomp1 (required for ONNX/FastEmbed runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages and the pre-downloaded model cache
COPY --from=model-downloader /usr/local /usr/local
# FastEmbed caches models in /tmp/fastembed_cache or ~/.cache/fastembed
# We copy the root cache to ensure it's available in the final image
COPY --from=model-downloader /root/.cache /root/.cache

COPY embeddingservice.py .

EXPOSE 7000

# Gunicorn settings: 
# Using 1 worker is CRITICAL on 1GB RAM to prevent multiple model instances from loading.
# Timeout is kept high to allow for larger PDF processing.
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "embeddingservice:app", "--bind", "0.0.0.0:7000", "--timeout", "150"]