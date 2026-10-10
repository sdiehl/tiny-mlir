import numpy as np
import pytest
from test_compiler import names

from tinymlir import gather, gpu, jit
from tinymlir.jit import GPU
from tinymlir.ops import attention, gelu, layer_norm, softmax

rng = np.random.default_rng(5)
x = rng.normal(size=(5, 8)).astype(np.float32)
CASES = {
    "gelu": (gelu, [x], {}),
    "softmax": (softmax, [x], {}),
    "layer_norm": (layer_norm, [x, *rng.normal(size=(2, 8)).astype(np.float32)], {}),
    "matmul": (lambda a, b: a @ b, [x, x.T.copy()], {}),
    "gather": (gather, [x, np.array([4, 0, 2], np.int32)], {}),
    "attention": (
        attention,
        [x, *(rng.normal(size=s).astype(np.float32) for s in [(8, 24), (24,), (8, 8), (8,)])],
        {"heads": 2},
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_every_loop_becomes_a_kernel(case):
    function, args, static = CASES[case]
    compiled = jit(function).compile(*args, **static)
    with compiled.context:
        gpu.outline(compiled.optimized)
    operations = names(compiled.optimized)
    host = operations[: operations.index("gpu.module")]
    assert "gpu.launch_func" in host
    assert not [op for op in host if op.startswith(("scf.", "linalg.", "memref.alloc"))]


@pytest.mark.skipif(not gpu.available(), reason="needs an NVIDIA GPU")
@pytest.mark.parametrize("case", CASES)
def test_gpu_matches_cpu(case):
    function, args, static = CASES[case]
    function = jit(function)
    expected = function(*args, **static)
    actual = function(*args, target=GPU, **static)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    # Results live in managed memory and feed later calls without a copy.
    assert gpu.device(actual) is actual


@pytest.mark.skipif(not gpu.RUNTIME.exists(), reason="needs the CUDA build of MLIR")
@pytest.mark.parametrize("case", CASES)
def test_kernels_compile_to_ptx(case):
    function, args, static = CASES[case]
    compiled = jit(function).compile(*args, **static)
    with compiled.context:
        gpu.lower(compiled.optimized)
    assert ".entry kernel_kernel" in str(compiled.optimized)
