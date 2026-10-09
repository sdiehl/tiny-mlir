"""Warm end-to-end operator latency, including Python dispatch and allocation."""

import argparse
import statistics
import time

import numpy as np

from tinymlir.compiler import CPU, Numpy

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("cpu", "cuda", "numpy"), default="cpu")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--cache", default=".mlir-cache")
    p.add_argument("--chip", default="sm_75")
    p.add_argument("--libdevice")
    args = p.parse_args()
    if min(args.rows, args.width, args.repeats) <= 0:
        p.error("Sizes and repeats must be positive")
    if args.backend == "cuda":
        from tinymlir.cuda_runtime import CUDA

        ops = CUDA(args.cache, args.chip, args.libdevice)
    else:
        ops = CPU(args.cache) if args.backend == "cpu" else Numpy()
    try:
        rng = np.random.default_rng(42)
        inputs = [
            ops.array(rng.normal(size=s).astype(np.float32))
            for s in [
                (args.rows, args.width),
                (args.width, 4 * args.width),
                (4 * args.width,),
            ]
        ]
        functions = {
            "linear + gelu": lambda: ops("gelu", ops("linear", *inputs)),
            "fused linear_gelu": lambda: ops("linear_gelu", *inputs),
        }
        for name, fn in functions.items():
            fn()
            ops.sync()  # Compile and warm before timing.
            samples = []
            for _ in range(args.repeats):
                ops.sync()
                start = time.perf_counter()
                result = fn()
                ops.sync()
                samples.append((time.perf_counter() - start) * 1000)
            print(
                f"{name}: median {statistics.median(samples):.4f} ms; "
                f"min {min(samples):.4f} ms; {args.repeats} runs; backend={args.backend}"
            )
    finally:
        ops.close()
