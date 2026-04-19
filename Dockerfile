FROM python:3.11-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ libgomp1
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
# Copy cached model weights baked into the image
COPY --from=builder /root/.cache /root/.cache
COPY embeddingservice.py .
ENV PATH=/root/.local/bin:$PATH
EXPOSE 7000
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "embeddingservice:app", "--bind", "0.0.0.0:7000", "--timeout", "120"]