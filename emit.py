"""Print a complete MLIR kernel for inspection or command-line lowering."""

import argparse

from tinymlir.kernels import Tensor, generate

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("op", choices=("gelu", "softmax", "norm", "linear", "linear_gelu", "scores"))
    p.add_argument("--target", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--width", type=int, default=8)
    a = p.parse_args()
    x = Tensor((a.rows, a.width))
    options = {}
    if a.op == "norm":
        inputs = [x, Tensor((a.width,)), Tensor((a.width,))]
    elif a.op.startswith("linear"):
        inputs = [x, Tensor((a.width, 4 * a.width)), Tensor((4 * a.width,))]
    elif a.op == "scores":
        inputs = [Tensor((a.rows, 3 * a.width))]
        options = {"heads": 2}
    else:
        inputs = [x]
    print(generate(a.op, inputs, a.target, **options)[0], end="")
