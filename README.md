# tiny-mlir

A small tensor compiler in Python using MLIR's builder interface. Python functions become typed tensor graphs, then `linalg` operations which MLIR fuses, bufferizes and lowers to native CPU code. The example runs pretrained GPT-2.

## Running

```bash
uv sync
uv run python fetch_model.py model
```

```console
$ uv run python example.py "Alan Turing theorized that computers would one day become" --tokens 10
Alan Turing theorized that computers would one day become the most powerful machines on the planet.
```

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
