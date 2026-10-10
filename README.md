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

Every `linalg.matmul` is handed to the system BLAS (Accelerate on macOS, OpenBLAS on Linux) after bufferization, which is where NumPy gets its speed too. Without a BLAS the matmuls lower to plain loops.

Pass `--fused` to `example.py` or `verify.py` to compile the entire forward pass into one MLIR kernel. The traced function in `tinymlir/model.py` is the NumPy reference written as is: `np.mean`, `np.var`, `np.split`, `np.triu` masks, integer indexing and nested parameter tuples all trace, and captured arrays become trailing kernel arguments. Sequence lengths are padded to a power of two so decoding compiles a handful of kernels rather than one per token. The default splits the model into one kernel per sublayer, which compiles in well under a second each.

## GPU

Requires Linux x86_64, an NVIDIA driver, and a GPU of compute capability sm_75 (Turing) or newer.

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
