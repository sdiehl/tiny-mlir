import numpy as np
import pytest

from tinymlir import gather, jit
from tinymlir.jit import Compiled
from tinymlir.ops import attention, gelu, layer_norm, softmax


def names(module):
    def walk(operation):
        yield operation.name
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    yield from walk(child.operation)

    return list(walk(module.operation))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_broadcast_and_fusion(dtype):
    @jit
    def function(x, gain, bias):
        return (x * gain + bias) * 0.5

    x = np.arange(21, dtype=dtype).reshape(3, 7)
    gain = np.arange(7, dtype=dtype)
    bias = np.ones((3, 1), dtype=dtype)
    compiled = function.compile(x, gain, bias)
    np.testing.assert_allclose(compiled(x, gain, bias), (x * gain + bias) * 0.5)
    assert names(compiled.module).count("linalg.generic") == 3
    assert names(compiled.optimized).count("linalg.generic") == 1
    assert all(not name.startswith("linalg.") for name in names(compiled.lowered))
    unfused = Compiled(function.function, compiled.types, {}, optimize=False)
    np.testing.assert_array_equal(compiled(x, gain, bias), unfused(x, gain, bias))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("batch", [False, True])
def test_matmul(dtype, batch):
    rng = np.random.default_rng(1)
    prefix = (2,) if batch else ()
    a = rng.normal(size=prefix + (3, 5)).astype(dtype)
    b = rng.normal(size=prefix + (5, 7)).astype(dtype)
    function = jit(lambda a, b: a @ b)
    compiled = function.compile(a, b)
    assert ("linalg.batch_matmul" if batch else "linalg.matmul") in names(compiled.module)
    np.testing.assert_allclose(compiled(a, b), a @ b, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_reductions(dtype, axis, keepdims):
    x = np.random.default_rng(2).normal(size=(3, 7)).astype(dtype)
    function = jit(lambda x: x.mean(axis, keepdims) + x.max(axis, keepdims))
    expected = x.mean(axis=axis, keepdims=keepdims) + x.max(axis=axis, keepdims=keepdims)
    np.testing.assert_allclose(function(x), expected, rtol=2e-6, atol=2e-6)


def test_softmax_extremes():
    x = np.array(
        [[10000, 9999, -10000], [-10000, -9999, -10001], [0, -np.inf, -np.inf]], np.float32
    )
    y = jit(softmax)(x)
    expected = np.exp(x - x.max(axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(y, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(y.sum(axis=-1), 1, atol=1e-6)


def test_layer_norm_constant_and_random():
    x = np.random.default_rng(3).normal(size=(4, 7)).astype(np.float32)
    x[0] = 1000
    gain = np.arange(7, dtype=np.float32)
    bias = np.ones(7, np.float32)
    y = jit(layer_norm)(x, gain, bias)
    expected = (
        gain * (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-5)
        + bias
    )
    np.testing.assert_allclose(y, expected, rtol=2e-6, atol=2e-6)


def test_gelu_promotion():
    x = np.linspace(-5, 5, 31, dtype=np.float32)
    y = jit(gelu)(x)
    expected = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    assert y.dtype == expected.dtype == np.float64
    np.testing.assert_allclose(y, expected, rtol=2e-6, atol=2e-6)


def test_views():
    x = np.arange(60, dtype=np.float32).reshape(5, 12)
    function = jit(lambda x: x[1:4, 4:12].reshape((3, 2, 4)).transpose((1, 0, 2)))
    np.testing.assert_array_equal(function(x), x[1:4, 4:12].reshape(3, 2, 4).transpose(1, 0, 2))


def test_gather():
    table = np.arange(35, dtype=np.float32).reshape(7, 5)
    ids = np.array([0, 6, 2], np.int32)
    function = jit(gather)
    np.testing.assert_array_equal(function(table, ids), table[ids])
    with pytest.raises(ValueError, match="outside"):
        function(table, np.array([0, 7, 2], np.int32))


def test_causal_attention():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5, 8)).astype(np.float32)
    weights = [
        rng.normal(size=shape).astype(np.float32) for shape in [(8, 24), (24,), (8, 8), (8,)]
    ]
    function = jit(attention)
    before = function(x, *weights, heads=2)
    changed = x.copy()
    changed[3:] += 10
    np.testing.assert_allclose(before[:3], function(changed, *weights, heads=2)[:3], atol=1e-6)


def test_rejects_invalid_programs():
    x = np.ones((2, 3), np.float32)
    with pytest.raises(ValueError):
        jit(lambda x, y: x + y)(x, np.ones(4, np.float32))
    with pytest.raises(ValueError):
        jit(lambda x, y: x @ y)(x, x)
    with pytest.raises(ValueError):
        jit(lambda x: x.reshape((7,)))(x)
    with pytest.raises(ValueError):
        jit(lambda x: x.sum(axis=2))(x)
    with pytest.raises(TypeError, match="control flow"):
        jit(lambda x: x if x else x * 2)(x)
    with pytest.raises(TypeError):
        jit(lambda x: x + 1)(np.ones(3, np.int32))


def test_aliases_and_specialization():
    function = jit(lambda x, y: x + y)
    x = np.arange(9, dtype=np.float32)
    np.testing.assert_array_equal(function(x[:6], x[3:]), x[:6] + x[3:])
    compiled = function.compile(x[:6], x[3:])
    assert function.compile(x[:6], x[3:]) is compiled
    with pytest.raises(TypeError):
        compiled(x, x)
