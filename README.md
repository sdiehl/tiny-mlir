# tiny-mlir

A small tensor compiler in Python using MLIR's builder interface. Python functions become typed tensor graphs, then `linalg` operations which MLIR fuses, bufferizes and lowers to native CPU code. The example runs pretrained GPT-2.

## Running

```bash
uv sync
uv run python fetch_model.py model
uv run python example.py
```

Completes “Alan Turing theorized that computers would one day become” with 10 tokens. Python 3.12 and the MLIR bindings are pinned; no separate LLVM installation is needed. The supplied wheels support macOS and Linux on ARM64 and x86-64.

```bash
# Compiler and numerical tests
uv run pytest

# Compare every decoding step with the original NumPy implementation
# Requires a working tiny-gpt2 checkout, including its encoder files.
uv run python verify.py --reference ../tiny-gpt2

# Inspect tensor IR, fusion, and LLVM lowering
uv run python emit.py gelu
uv run python emit.py gelu --stage optimized
uv run python emit.py gelu --stage lowered
```

## Development

Install [dprint](https://dprint.dev/install/) for Markdown, JSON and TOML formatting.

```bash
uv sync --locked
uv run pre-commit install
uv run black .
dprint fmt
uv run pre-commit run --all-files
uv run pytest
```

CI checks formatting, lint, the lockfile and pre-commit hooks, and runs tests and package builds on Linux and macOS.

## License

MIT.
