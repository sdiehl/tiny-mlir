# tiny-mlir

A small tensor compiler in Python using MLIR's builder interface. Python functions become typed tensor graphs, then `linalg` operations which MLIR fuses, bufferizes and lowers to native CPU code or NVIDIA GPU kernels. The example runs pretrained GPT-2.

## CPU

```bash
uv sync
uv run python fetch_model.py model
uv run python example.py "Alan Turing theorized that computers would one day become" --tokens 10
uv run python verify.py
uv run python emit.py gelu --stage lowered
uv run pytest
```

The compiler is five modules read in order: `expr.py` traces Python into a typed tensor graph, `builder.py` emits `linalg`, `jit.py` runs the passes and calls the result, `ops.py` holds the transformer formulas and `model.py` runs GPT-2 one compiled sublayer at a time.

Three modules build on that path without changing it. `blas.py` hands every bufferized `linalg.matmul` to the system BLAS (Accelerate on macOS, OpenBLAS on Linux), which is where NumPy gets its speed too; without a BLAS the matmuls lower to plain loops. `fused.py` traces the NumPy reference forward pass as written, `np.mean`, `np.split`, `np.triu` masks and nested parameter tuples included, into one kernel per sequence length; pass `--fused` to `example.py` or `verify.py` to use it. `gpu.py` maps the remaining loops to CUDA kernels and replaces matmuls with a shared-memory tiled kernel.

## GPU

Requires Linux x86_64, an NVIDIA driver, and a GPU of compute capability sm_75 (Turing) or newer.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdiehl/tiny-mlir/blob/main/Colab.ipynb)

[`Colab.ipynb`](Colab.ipynb) installs the CUDA build, inspects the outlined kernels and their PTX, generates on the GPU and runs the reference check on a T4.

```bash
uv sync
uv run python fetch_model.py model
uv run python example.py "Alan Turing theorized that computers would one day become" --tokens 10 --gpu
uv run python verify.py --gpu
uv run python emit.py gelu --stage lowered --gpu
uv run pytest
```

Verification downloads a pinned copy of the [NumPy reference](https://github.com/sdiehl/tiny-gpt2) and [tokenizer](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e).

## License

MIT Licensed. Copyright 2026 Stephen Diehl. See [LICENSE](LICENSE) for details.
