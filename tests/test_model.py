"""The NumPy forward pass traces into one kernel that agrees with the per-operation kernels."""

import numpy as np
from test_compiler import names

from tinymlir import blas
from tinymlir.fused import gpt2
from tinymlir.model import Model

WIDTH, HEADS, LAYERS, CONTEXT, VOCABULARY = 8, 2, 2, 8, 16


def random_checkpoint():
    rng = np.random.default_rng(7)

    def parameters(prefix, shape):
        return {
            f"{prefix}.weight": (rng.normal(size=shape) * 0.3).astype(np.float32),
            f"{prefix}.bias": (rng.normal(size=shape[-1:]) * 0.1).astype(np.float32),
        }

    weights = {
        "wte.weight": rng.normal(size=(VOCABULARY, WIDTH)).astype(np.float32),
        "wpe.weight": rng.normal(size=(CONTEXT, WIDTH)).astype(np.float32),
        **parameters("ln_f", (WIDTH,)),
    }
    for index in range(LAYERS):
        for name, shape in [
            ("ln_1", (WIDTH,)),
            ("attn.c_attn", (WIDTH, 3 * WIDTH)),
            ("attn.c_proj", (WIDTH, WIDTH)),
            ("ln_2", (WIDTH,)),
            ("mlp.c_fc", (WIDTH, 4 * WIDTH)),
            ("mlp.c_proj", (4 * WIDTH, WIDTH)),
        ]:
            weights.update(parameters(f"h.{index}.{name}", shape))
    config = {
        "n_embd": WIDTH,
        "n_head": HEADS,
        "n_layer": LAYERS,
        "n_positions": CONTEXT,
        "vocab_size": VOCABULARY,
    }
    return weights, config


def test_whole_model_is_one_kernel():
    weights, config = random_checkpoint()
    ids = [3, 1, 4, 1, 5]
    fused = Model(weights, config, fused=True)
    np.testing.assert_allclose(fused(ids), Model(weights, config)(ids), rtol=1e-6, atol=1e-6)

    compiled = gpt2.compile(np.asarray(ids, np.int32), fused.weights, n_head=HEADS)
    operations = names(compiled.module)
    assert operations.count("func.func") == 1
    assert operations.count("linalg.matmul") == 4 * LAYERS + 1
    assert operations.count("linalg.batch_matmul") == 2 * LAYERS
    # One captured causal mask per block rides along as a trailing argument.
    assert len(compiled.constants) == LAYERS
    assert not [name for name in names(compiled.lowered) if name.startswith("linalg.")]
    if blas.LIBRARY:
        assert str(compiled.lowered).count("llvm.call @cblas_") == 6 * LAYERS + 1
