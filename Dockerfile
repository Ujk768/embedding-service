# Stage 1: Builder
FROM python:3.11-slim as model-downloader
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y gcc g++

COPY requirements.txt .
# Install globally in the builder stage
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding(model_name='BAAI/bge-small-en-v1.5').embed(['warmup']))"

# Stage 2: Final Image
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy everything from /usr/local where pip installs by default
COPY --from=model-downloader /usr/local /usr/local
# Copy the model cache
COPY --from=model-downloader /root/.cache /root/.cache

COPY embeddingservice.py .

# Ensure /usr/local/bin is in the PATH (it usually is, but this is safe)
ENV PATH="/usr/local/bin:$PATH"

EXPOSE 7000

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "embeddingservice:app", "--bind", "0.0.0.0:7000", "--timeout", "150"]