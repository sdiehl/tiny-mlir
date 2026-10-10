# tiny-mlir

A small tensor compiler in Python using MLIR's builder interface. Python functions become typed tensor graphs, then `linalg` operations which MLIR fuses, bufferizes and lowers to native CPU code or NVIDIA GPU kernels. The example runs pretrained GPT-2.

## Running

```bash
uv sync
uv run python fetch_model.py model
```

```console
$ uv run python example.py "Alan Turing theorized that computers would one day become" --tokens 10
Alan Turing theorized that computers would one day become the most powerful machines on the planet.
```

On the CPU every `linalg.matmul` is handed to the system BLAS (Accelerate on macOS, OpenBLAS on Linux) after bufferization, which is where NumPy gets its speed too; without a BLAS the matmuls lower to plain loops.

Pass `--fused` to compile the entire forward pass into one MLIR kernel. The traced function in `tinymlir/model.py` is the NumPy reference written as is: `np.mean`, `np.var`, `np.split`, `np.triu` masks, integer indexing and nested parameter tuples all trace, and captured arrays become trailing kernel arguments. Sequence lengths are padded to a power of two so decoding compiles a handful of kernels rather than one per token. The default splits the model into one kernel per sublayer, which compiles in well under a second each.

Pass `--gpu` to `example.py`, `verify.py` or `emit.py` to target an NVIDIA GPU instead of the CPU (the default, also `--cpu`). Parallel loops are tiled onto blocks and threads, outlined into `gpu.module` kernels and compiled to PTX, which the driver JIT compiles at load. Buffers and weights live in CUDA managed memory. This needs Linux x86_64 and an NVIDIA driver, where the locked dependencies already include the CUDA build of MLIR and libdevice.

```bash
# Compiler and numerical tests
uv run pytest

# Compare every decoding step with the original NumPy implementation
uv run python verify.py

# Inspect tensor IR, fusion, and LLVM lowering
uv run python emit.py gelu
uv run python emit.py gelu --stage optimized
uv run python emit.py gelu --stage lowered
```

Verification downloads a pinned copy of the [NumPy reference](https://github.com/sdiehl/tiny-gpt2) and [tokenizer](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e).

## License

MIT Licensed. Copyright 2026 Stephen Diehl. See [LICENSE](LICENSE) for details.
