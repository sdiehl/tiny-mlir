"""Numerical checks, model checks, or GPU cross-compilation checks."""

import argparse

import numpy as np

from tinymlir.compiler import CPU, Compiler, Numpy, spec
from tinymlir.kernels import generate
from tinymlir.model import Model, random_model


def cases():
    rng = np.random.default_rng(17)

    def a(shape):
        return rng.normal(size=shape).astype(np.float32)

    x = a((3, 7))
    yield "gelu", (np.array([[-20, -5, -1, -0.01, 0, 0.01, 1, 5, 20]], np.float32),), {}
    yield "add", (x, a(x.shape)), {}
    yield "softmax", (np.array([[10000, 9999, -10000], [-10000, -9999, -10001]], np.float32),), {}
    yield "softmax", (np.array([[0, -np.inf, -np.inf], [2, 3, -np.inf]], np.float32),), {}
    yield "norm", (x, a((7,)), a((7,))), {}
    yield "norm", (np.full((2, 7), 1000, np.float32), a((7,)), a((7,))), {}
    yield "matmul", (a((3, 5)), a((5, 7))), {}
    yield "matmul", (a((3, 5)), a((7, 5))), {"transpose": True}
    yield "linear", (a((3, 5)), a((5, 7)), a((7,))), {}
    yield "linear_gelu", (a((3, 5)), a((5, 7)), a((7,))), {}
    yield "scores", (a((5, 48)),), {"heads": 2}
    qkv = a((5, 48))
    ref = Numpy()
    probs = ref("softmax", ref("scores", qkv, heads=2))
    yield "context", (qkv, probs), {"heads": 2}
    yield "embedding", (np.array([0, 6, 2], np.int32), a((7, 16)), a((8, 16))), {}
    yield "logits", (a((5, 16)), a((41, 16))), {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--ptx-only", action="store_true")
    p.add_argument("--cache", default=".mlir-cache")
    p.add_argument("--libdevice")
    p.add_argument("--chip", default="sm_75")
    args = p.parse_args()
    if args.ptx_only:
        compiler = Compiler(args.cache, args.chip, args.libdevice)
        for name, inputs, options in cases():
            source, _, _ = generate(name, tuple(spec(a) for a in inputs), "cuda", **options)
            path = compiler.build(source, "cuda")
            assert ".visible .entry kernel(" in path.read_text()
            print(name, "PTX compiled and device symbols resolved")
        print("Cross-compilation passed. This does not test GPU execution.")
        return
    if args.backend == "cuda":
        from tinymlir.cuda_runtime import CUDA

        ops = CUDA(args.cache, args.chip, args.libdevice)
    else:
        ops = CPU(args.cache)
    ref = Numpy()
    try:
        for name, inputs, options in cases():
            expected = ref(name, *inputs, **options)
            actual = ops.numpy(ops(name, *(ops.array(a) for a in inputs), **options))
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5, err_msg=name)
            if name == "softmax":
                np.testing.assert_allclose(actual.sum(axis=-1), 1, atol=1e-6)
            print(name, "numerical check passed")
        qkv = np.random.default_rng(8).normal(size=(5, 48)).astype(np.float32)

        def attention(x):
            x = ops.array(x)
            p = ops("softmax", ops("scores", x, heads=2))
            return ops.numpy(ops("context", x, p, heads=2))

        original = attention(qkv)
        changed = qkv.copy()
        changed[3:] += 10
        np.testing.assert_allclose(original[:3], attention(changed)[:3], atol=1e-6)
        weights, config = random_model()
        compiled = Model(weights, config, ops)
        reference = Model(weights, config, ref)
        for ids in ([1], [1, 2, 3], [1, 2, 3, 4, 5]):
            actual_trace, expected_trace = {}, {}
            compiled(ids, actual_trace)
            reference(ids, expected_trace)
            for name, expected in expected_trace.items():
                np.testing.assert_allclose(
                    actual_trace[name], expected, rtol=3e-4, atol=3e-5, err_msg=name
                )
        unfused = Model(weights, config, ops, fused=False)
        np.testing.assert_allclose(
            ops.numpy(compiled([1, 2, 3])),
            ops.numpy(unfused([1, 2, 3])),
            rtol=3e-4,
            atol=3e-5,
        )
        for ids in ([], [-1], [config["vocab_size"]], [1] * 33):
            try:
                compiled(ids)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid tokens accepted: {ids}")
        for name, types, options in [
            (
                "norm",
                (
                    spec(np.zeros((2, 7), np.float32)),
                    spec(np.zeros(6, np.float32)),
                    spec(np.zeros(7, np.float32)),
                ),
                {},
            ),
            (
                "matmul",
                (
                    spec(np.zeros((2, 3), np.float32)),
                    spec(np.zeros((4, 5), np.float32)),
                ),
                {},
            ),
            ("scores", (spec(np.zeros((2, 48), np.float32)),), {"heads": 3}),
        ]:
            try:
                generate(name, types, **options)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid shapes accepted")
        print(
            "Causal masking, multiple sequence lengths, block outputs, logits, fusion and invalid inputs passed."
        )
    finally:
        ops.close()


if __name__ == "__main__":
    main()
