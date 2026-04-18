import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

DEVICE = "cpu"
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

# Batch size for the transformer forward pass.
# On a dedicated 1GB machine with no other load, 32 is safe and fast.
ENCODE_BATCH_SIZE = int(os.getenv("ENCODE_BATCH_SIZE", "32"))

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"[INFO] Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print("[INFO] Model ready.")
    yield
    print("[INFO] Shutting down.")


app = FastAPI(lifespan=lifespan, title="Embedding Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    count: int


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    if not request.texts:
        return EmbedResponse(embeddings=[], count=0)

    embeddings = model.encode(
        request.texts,
        batch_size=ENCODE_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )

    print(f"[INFO] Processed {len(request.texts)} texts into embeddings.")

    return EmbedResponse(
        embeddings=embeddings.tolist(),
        count=len(embeddings),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}