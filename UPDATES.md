# Updates

The [series](https://www.stephendiehl.com/posts/mlir_introduction/) was finalized on 2026-10-09 against the code at `ca99486`. The repository has moved since. Each entry names the passage that no longer matches, so the posts can be brought back in line.

## Part 9, A Python Frontend

`Expr` now implements `__array_ufunc__` and `__array_function__`. Calls such as `np.mean`, `np.var`, `np.sqrt`, `np.split` and `np.triu` on a traced value build graph nodes, so NumPy code traces without being rewritten against the library's own functions. NumPy arrays captured from the enclosing scope become constants, which the JIT then lifts to trailing kernel arguments. The post describes only the operator overloads and the library functions.

## Part 11, Matrix Multiplication and Attention

The post says reshapes lower to `tensor.collapse_shape` and `tensor.expand_shape`. The builder now emits `tensor.reshape` with a shape operand. Elementwise fusion folds collapse and expand into an adjacent `linalg.matmul` and generalizes it, which hides the matmul from the substitutions below.

## Part 12, Fusion, Bufferization and Execution

The quoted `LOWERING_PIPELINE` is now three pipelines. Fusion runs on tensors, then `BUFFERIZATION_PIPELINE` assigns and releases storage once, then each target substitutes its own matmul before lowering the rest. On the CPU, `blas.py` replaces row-major `linalg.matmul` and `linalg.batch_matmul` with `cblas_sgemm` calls into Accelerate or OpenBLAS when one is present, and falls back to loops otherwise. On the GPU, `gpu.py` replaces them with a hand-written shared-memory tiled kernel.

The cache key is no longer a flat tuple of tensor types. Arguments are flattened as trees of tuples, named tuples, dataclasses and dicts, and the key is that structure with the leaf types, plus the static keyword arguments and the target.

The closing section says the companion runs only on the CPU. It now has an NVIDIA target, selected with `--gpu`, that maps the remaining parallel loops to CUDA kernels over managed memory. PTX is validated with `ptxas` in CI, and `verify.py --gpu` runs the reference comparison on hardware.

## Part 13, Running GPT-2

The model has a second mode, `--fused`, in which the NumPy reference forward pass from `tinygpt2/gpt2_minimal.py` is traced as written into one kernel per padded sequence length. Sequence lengths are padded to a power of two so decoding compiles a handful of kernels rather than one per token. In this mode only the final logits are compared against the reference; the per-block residual checks listed in the post apply to the per-sublayer path.

The statement that the companion runs on the CPU and that a GPU target would be future work is out of date, as above.

## Owed

A chapter on the GPU target, the tiled matmul kernel and the fused model has not been written.
