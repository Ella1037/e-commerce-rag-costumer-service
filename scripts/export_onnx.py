# scripts/export_onnx.py
"""
Export thenlper/gte-small to ONNX and apply INT8 dynamic quantization.
Benchmark both versions to measure latency improvement.
"""
import time
import numpy as np
import statistics

# ── 1. Export to ONNX ─────────────────────────────────────────
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

MODEL_NAME = "thenlper/gte-small"
ONNX_PATH  = "models/gte-small-onnx"
QUANT_PATH = "models/gte-small-onnx-int8"

print("Step 1: Exporting to ONNX...")
ort_model = ORTModelForFeatureExtraction.from_pretrained(
    MODEL_NAME, export=True
)
ort_model.save_pretrained(ONNX_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(ONNX_PATH)
print(f"  ✓ Saved to {ONNX_PATH}")

# ── 2. INT8 Dynamic Quantization ──────────────────────────────
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

os.makedirs(QUANT_PATH, exist_ok=True)
quantize_dynamic(
    model_input=f"{ONNX_PATH}/model.onnx",
    model_output=f"{QUANT_PATH}/model.onnx",
    weight_type=QuantType.QInt8,
)
# 複製所有 config 檔案（包含 config.json，ORTModel 需要它來辨識 library）
import shutil
for f in os.listdir(ONNX_PATH):
    if f != "model.onnx":  # model.onnx 已經是 quantized 版本
        src = os.path.join(ONNX_PATH, f)
        dst = os.path.join(QUANT_PATH, f)
        if os.path.isfile(src):
            shutil.copy(src, dst)
print(f"  ✓ Quantized model saved to {QUANT_PATH}")

# ── 3. Model size comparison ──────────────────────────────────
orig_size = os.path.getsize(f"{ONNX_PATH}/model.onnx") / 1e6
quant_size = os.path.getsize(f"{QUANT_PATH}/model.onnx") / 1e6
print(f"\nModel size: {orig_size:.1f} MB → {quant_size:.1f} MB "
      f"({(1 - quant_size/orig_size)*100:.0f}% smaller)")

# ── 4. Latency benchmark ──────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import torch

TEST_QUERIES = [
    "How do I return an item?",
    "When will I receive my refund?",
    "How do I track my order?",
    "What payment methods are accepted?",
    "My order is delayed, what should I do?",
] * 4  # 20 runs

N = len(TEST_QUERIES)

def benchmark_hf(queries):
    """Original HuggingFace model"""
    model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        model.embed_query(q)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies

def benchmark_onnx(model_path, queries):
    """ONNX model via optimum"""
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = ORTModelForFeatureExtraction.from_pretrained(model_path)

    latencies = []
    for q in queries:
        inputs = tok(q, return_tensors="pt", padding=True, truncation=True, max_length=128)
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = mdl(**inputs)
        # mean pooling
        _ = outputs.last_hidden_state.mean(dim=1)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies

print("\nBenchmarking original HuggingFace model...")
hf_lat = benchmark_hf(TEST_QUERIES)

print("Benchmarking ONNX FP32...")
onnx_lat = benchmark_onnx(ONNX_PATH, TEST_QUERIES)

print("Benchmarking ONNX INT8...")
int8_lat = benchmark_onnx(QUANT_PATH, TEST_QUERIES)

def summary(name, latencies):
    s = sorted(latencies)
    print(f"  {name:<20} mean={statistics.mean(latencies):6.1f}ms  "
          f"p50={statistics.median(latencies):6.1f}ms  "
          f"p95={s[int(0.95*len(s))-1]:6.1f}ms")

print(f"\n{'='*55}")
print("EMBEDDING LATENCY COMPARISON")
print(f"{'='*55}")
summary("HuggingFace (orig)", hf_lat)
summary("ONNX FP32",          onnx_lat)
summary("ONNX INT8",          int8_lat)

improvement = statistics.mean(hf_lat) / statistics.mean(int8_lat)
print(f"\n→ Speedup (HF → ONNX INT8): {improvement:.1f}x")