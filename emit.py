"""Inspect the MLIR modules for a compiled tensor function."""

import argparse

import numpy as np

from tinymlir import jit
from tinymlir.ops import gelu, softmax

p = argparse.ArgumentParser()
p.add_argument("operation", choices=["gelu", "softmax"])
p.add_argument("--stage", choices=["module", "optimized", "lowered"], default="module")
args = p.parse_args()
function = jit({"gelu": gelu, "softmax": softmax}[args.operation])
compiled = function.compile(np.zeros((2, 8), np.float32))
print(getattr(compiled, args.stage))
