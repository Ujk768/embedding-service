import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# We use BGE-Small (384 dims) because it is highly optimized for CPU.
# FastEmbed handles the "DEVICE" and "CPU threads" internally very efficiently.
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-small-en-v1.5")

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"[INFO] Loading FastEmbed model: {MODEL_NAME}")
    # This library uses ONNX Runtime under the hood (No PyTorch!)
    model = TextEmbedding(model_name=MODEL_NAME)
    print("[INFO] Model ready.")
    yield
    print("[INFO] Shutting down.")

app = FastAPI(lifespan=lifespan, title="Fast Embedding Service")

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

    # FastEmbed's .embed returns an iterator of numpy arrays
    # It is optimized for CPU batching internally.
    embeddings_iter = model.embed(request.texts)
    embeddings_list = [e.tolist() for e in embeddings_iter]

    print(f"[INFO] Processed {len(request.texts)} texts into embeddings.")

    return EmbedResponse(
        embeddings=embeddings_list,
        count=len(embeddings_list),
    )

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}