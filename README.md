# tiny-mlir

A small GPT-2 implementation in Python that compiles transformer operations through
MLIR to native code. Uses pretrained GPT-2 weights and greedy decoding. A teaching
project for the MLIR series.

## Running

Requires Python 3.12+, uv, and LLVM/MLIR 22.1.8 tools on `PATH` (`mlir-opt`,
`mlir-translate`, `clang`, `llc`, `llvm-link`, and `opt`).

```bash
uv sync
uv run python fetch_model.py model
uv run python example.py
```

The example completes “Alan Turing theorized that computers would one day become”
with 10 new tokens, running the model through MLIR on the CPU.

```bash
# Compare compiled operations against NumPy
uv run python check.py

# Compare the pretrained model against NumPy
uv run python -m tinymlir.cli --model-dir model --verify \
  --prompt "Alan Turing theorized that computers would one day become" --tokens 10
```

On macOS with Homebrew LLVM, first run `export PATH="$(brew --prefix llvm)/bin:$PATH"`.

## CUDA

Requires Linux, an NVIDIA GPU and the CUDA toolkit.

```bash
export LIBDEVICE=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc
uv run python -m tinymlir.cli --backend cuda --model-dir model \
  --prompt "Alan Turing theorized that computers would one day become" --tokens 10
```

## License

MIT.
