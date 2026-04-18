# app/benchmark.py
import httpx
import time
import statistics
import json

BASE_URL  = "http://localhost:8000"
N_RUNS    = 10
METHODS   = ["baseline", "reranker", "hyde"]
QUESTIONS = [
    "How do I return an item?",
    "When will I receive my refund?",
    "How do I track my order?",
    "What payment methods are accepted?",
    "My order is delayed, what should I do?",
]

def run_benchmark():
    results = {}

    for method in METHODS:
        print(f"\n{'='*45}")
        print(f"Benchmarking: {method.upper()}")
        print(f"{'='*45}")

        # ── clear cache before each method ──
        httpx.delete(f"{BASE_URL}/cache")

        cold_latencies = []
        warm_latencies = []

        for i in range(N_RUNS):
            question = QUESTIONS[i % len(QUESTIONS)]

            resp = httpx.post(
                f"{BASE_URL}/query",
                json={"question": question, "method": method},
                timeout=30.0,
            )
            data      = resp.json()
            latency   = data["latency_ms"]
            cache_hit = data["cache_hit"]

            tag = "WARM (hit) " if cache_hit else "COLD (miss)"
            print(f"  Run {i+1:2d} [{tag}]: {latency:7.1f} ms")

            if cache_hit:
                warm_latencies.append(latency)
            else:
                cold_latencies.append(latency)

        results[method] = {
            "cold": {
                "mean":   statistics.mean(cold_latencies)   if cold_latencies else 0,
                "p95":    sorted(cold_latencies)[int(0.95 * len(cold_latencies)) - 1]
                          if len(cold_latencies) >= 2 else 0,
            },
            "warm": {
                "mean":   statistics.mean(warm_latencies)   if warm_latencies else 0,
                "p95":    sorted(warm_latencies)[int(0.95 * len(warm_latencies)) - 1]
                          if len(warm_latencies) >= 2 else 0,
            },
        }

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"{'BENCHMARK SUMMARY (with cache)':^60}")
    print(f"{'='*60}")
    print(f"{'Method':<12} {'Cold mean':>12} {'Cold p95':>10} {'Warm mean':>12} {'Warm p95':>10}")
    print(f"{'-'*60}")
    for m, s in results.items():
        print(f"{m:<12} {s['cold']['mean']:>10.0f}ms {s['cold']['p95']:>8.0f}ms "
              f"{s['warm']['mean']:>10.0f}ms {s['warm']['p95']:>8.0f}ms")

    with open("benchmark_with_cache.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved to benchmark_with_cache.json")

if __name__ == "__main__":
    run_benchmark()