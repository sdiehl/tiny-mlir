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

## License

MIT Licensed. Copyright 2026 Stephen Diehl. See [LICENSE](LICENSE) for details.
