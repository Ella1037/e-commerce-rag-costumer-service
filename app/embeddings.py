# app/embeddings.py
"""
Drop-in replacement for HuggingFaceEmbeddings using ONNX INT8 model.
Implements the same .embed_query() / .embed_documents() interface
so FAISS vectorstore can use it without changes.
"""
from email.mime import text

import torch
import numpy as np
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

QUANT_PATH = "models/gte-small-onnx-int8"

class ONNXEmbeddings:
    def __init__(self, model_path: str = QUANT_PATH):
        print(f"Loading ONNX INT8 embedding from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = ORTModelForFeatureExtraction.from_pretrained(model_path)
        print("  ✓ ONNX embedding ready")

    def _encode(self, texts: list[str]) -> list[list[float]]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # L2 normalize
        norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        embeddings = (embeddings / norms).numpy()
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)
    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)
# module-level singleton — loaded once at server startup
onnx_embeddings = ONNXEmbeddings()