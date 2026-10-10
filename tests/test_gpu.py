import subprocess

import numpy as np
import pytest
from mlir import ir
from mlir.dialects.gpu import ObjectAttr
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
    functions = [
        op for op in compiled.optimized.body.operations if op.operation.name == "func.func"
    ]
    host = [name for op in functions for name in names(op)]
    assert "gpu.launch_func" in host
    assert not [op for op in host if op.startswith(("scf.", "linalg.", "memref.alloc"))]


@pytest.mark.parametrize("case", ["matmul", "attention"])
def test_matmuls_stage_tiles_in_shared_memory(case):
    function, args, static = CASES[case]
    compiled = jit(function).compile(*args, **static)
    with compiled.context:
        gpu.outline(compiled.optimized)
    assert "#gpu.address_space<workgroup>" in str(compiled.optimized)
    assert "gpu.barrier" in names(compiled.optimized)


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


@pytest.mark.skipif(not gpu.toolkit(), reason="needs the CUDA build of MLIR and ptxas")
@pytest.mark.parametrize("case", CASES)
def test_ptxas_accepts_kernels(case, tmp_path):
    function, args, static = CASES[case]
    compiled = jit(function).compile(*args, **static)
    with compiled.context:
        gpu.lower(compiled.optimized)
        binaries = []
        compiled.optimized.operation.walk(
            lambda op: (op.name == "gpu.binary" and binaries.append(op)) or ir.WalkResult.ADVANCE
        )
        modules = [ObjectAttr(o).object for op in binaries for o in op.attributes["objects"]]
    assert modules
    for index, ptx in enumerate(modules):
        source = tmp_path / f"{index}.ptx"
        source.write_bytes(ptx)
        ptxas = gpu.toolkit() / "bin" / "ptxas"
        subprocess.run(
            [ptxas, f"--gpu-name={gpu.CHIP}", source, "-o", tmp_path / f"{index}.cubin"],
            check=True,
        )
