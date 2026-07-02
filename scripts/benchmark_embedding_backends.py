#!/usr/bin/env python3
"""Benchmark Mimir embedding backends.

Run with:
    python scripts/benchmark_embedding_backends.py

Backends tested:
- sentence-transformer
- llama-server (if available at http://127.0.0.1:11435)
- fake (for baseline/reference only)
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any

import psutil
import torch

from mimir.application.factories import create_engine


@dataclass
class BackendResult:
    backend: str
    model: str
    output_dim: int
    cold_start_ms: float
    single_short_ms: float
    single_medium_ms: float
    single_long_ms: float
    batch_8_ms: float
    batch_32_ms: float
    batch_128_ms: float
    throughput_texts_per_sec: float
    memory_mb: float
    available: bool
    error: str | None = None


SHORT_TEXT = "hello world"
MEDIUM_TEXT = "Mimir is a local-first plastic memory layer for coding agents. " * 5
LONG_TEXT = "The quick brown fox jumps over the lazy dog. " * 50

BATCH_TEXTS = [f"sample text number {i} for benchmarking embedding backends" for i in range(128)]


def _measure_latency(fn, warmup: int = 1, runs: int = 10) -> float:
    """Return average latency in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        gc.collect()
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return sum(times) / len(times)


def benchmark_backend(backend: str, model: str = "all-MiniLM-L6-v2") -> BackendResult:
    """Benchmark a single backend."""
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    try:
        t0 = time.perf_counter()
        engine = create_engine(backend, model=model)
        _ = engine.encode([SHORT_TEXT])  # force model load / connection
        cold_start_ms = (time.perf_counter() - t0) * 1000
        output_dim = engine.output_dim
    except Exception as exc:
        return BackendResult(
            backend=backend,
            model=model,
            output_dim=0,
            cold_start_ms=0,
            single_short_ms=0,
            single_medium_ms=0,
            single_long_ms=0,
            batch_8_ms=0,
            batch_32_ms=0,
            batch_128_ms=0,
            throughput_texts_per_sec=0,
            memory_mb=0,
            available=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    single_short_ms = _measure_latency(lambda: engine.encode([SHORT_TEXT]))
    single_medium_ms = _measure_latency(lambda: engine.encode([MEDIUM_TEXT]))
    single_long_ms = _measure_latency(lambda: engine.encode([LONG_TEXT]))
    batch_8_ms = _measure_latency(lambda: engine.encode(BATCH_TEXTS[:8]))
    batch_32_ms = _measure_latency(lambda: engine.encode(BATCH_TEXTS[:32]))
    batch_128_ms = _measure_latency(lambda: engine.encode(BATCH_TEXTS[:128]))

    # throughput on 128 texts
    gc.collect()
    t0 = time.perf_counter()
    _ = engine.encode(BATCH_TEXTS)
    elapsed = time.perf_counter() - t0
    throughput_texts_per_sec = len(BATCH_TEXTS) / elapsed

    mem_after = process.memory_info().rss / 1024 / 1024
    memory_mb = mem_after - mem_before

    return BackendResult(
        backend=backend,
        model=model,
        output_dim=output_dim,
        cold_start_ms=cold_start_ms,
        single_short_ms=single_short_ms,
        single_medium_ms=single_medium_ms,
        single_long_ms=single_long_ms,
        batch_8_ms=batch_8_ms,
        batch_32_ms=batch_32_ms,
        batch_128_ms=batch_128_ms,
        throughput_texts_per_sec=throughput_texts_per_sec,
        memory_mb=memory_mb,
        available=True,
    )


def _fmt(value: float, unit: str = "") -> str:
    if value == 0:
        return "N/A"
    return f"{value:.2f}{unit}"


def print_results(results: list[BackendResult]) -> None:
    """Print results as Markdown table."""
    print("\n# Embedding Backend Benchmark\n")
    print("| Backend | Model | Dim | Available | Cold Start | Short | Medium | Long | Batch-8 | Batch-32 | Batch-128 | Throughput | Memory |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r.backend} | {r.model} | {r.output_dim} | "
            f"{'✅' if r.available else '❌'} | "
            f"{_fmt(r.cold_start_ms, 'ms')} | "
            f"{_fmt(r.single_short_ms, 'ms')} | "
            f"{_fmt(r.single_medium_ms, 'ms')} | "
            f"{_fmt(r.single_long_ms, 'ms')} | "
            f"{_fmt(r.batch_8_ms, 'ms')} | "
            f"{_fmt(r.batch_32_ms, 'ms')} | "
            f"{_fmt(r.batch_128_ms, 'ms')} | "
            f"{_fmt(r.throughput_texts_per_sec, '/s')} | "
            f"{_fmt(r.memory_mb, 'MB')} |"
        )
        if r.error:
            print(f"| | | | | | | | | | | | Error: {r.error} |")


def main() -> None:
    backends: list[dict[str, Any]] = [
        {"backend": "sentence-transformer", "model": "all-MiniLM-L6-v2"},
        {"backend": "llama-server", "model": "llama-server@11435"},
        {"backend": "fake", "model": "fake-dim-16"},
    ]

    results: list[BackendResult] = []
    for cfg in backends:
        print(f"Benchmarking {cfg['backend']} ...")
        results.append(benchmark_backend(cfg["backend"], cfg["model"]))
        gc.collect()

    print_results(results)


if __name__ == "__main__":
    main()
