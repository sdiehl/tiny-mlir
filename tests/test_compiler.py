import os
import shutil

import numpy as np
import pytest

from tinymlir import CPU, Model, Numpy
from tinymlir.model import load_checkpoint, random_model


@pytest.fixture(scope="module")
def cpu(tmp_path_factory):
    if not shutil.which("mlir-opt"):
        pytest.skip("LLVM/MLIR tools are required")
    ops = CPU(tmp_path_factory.mktemp("compiled"))
    yield ops
    ops.close()


def compare_model(weights, config, ops, ids):
    actual, expected = {}, {}
    Model(weights, config, ops)(ids, actual)
    Model(weights, config, Numpy())(ids, expected)
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("ids", [[1], [1, 2, 3], [1, 2, 3, 4, 5]])
def test_transformer_blocks(cpu, ids):
    weights, config = random_model()
    compare_model(weights, config, cpu, ids)


def test_causal_attention(cpu):
    qkv = np.random.default_rng(8).normal(size=(5, 48)).astype(np.float32)

    def attention(x):
        return cpu("context", x, cpu("softmax", cpu("scores", x, heads=2)), heads=2)

    before = attention(qkv)
    qkv[3:] += 10
    np.testing.assert_allclose(before[:3], attention(qkv)[:3], atol=1e-6)


@pytest.mark.parametrize("ids", [[], [-1], [41], [1] * 33, [1.5]])
def test_invalid_tokens(cpu, ids):
    weights, config = random_model()
    with pytest.raises(ValueError):
        Model(weights, config, cpu)(ids)


def test_pretrained(cpu):
    directory = os.getenv("TINYMLIR_CHECKPOINT")
    if not directory:
        pytest.skip("Set TINYMLIR_CHECKPOINT to test pretrained GPT-2")
    weights, config, tokenizer = load_checkpoint(directory)
    compare_model(
        weights,
        config,
        cpu,
        tokenizer.encode("Alan Turing theorized that computers would one day become").ids,
    )


def test_cuda(tmp_path):
    if os.getenv("TINYMLIR_TEST_CUDA") != "1":
        pytest.skip("Set TINYMLIR_TEST_CUDA=1 on a CUDA host")
    from tinymlir.cuda_runtime import CUDA

    ops = CUDA(tmp_path)
    try:
        weights, config = random_model()
        compare_model(weights, config, ops, [1, 2, 3])
    finally:
        ops.close()
