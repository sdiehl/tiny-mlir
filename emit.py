"""Inspect the MLIR modules for a compiled tensor function."""

import argparse

import numpy as np

from tinymlir import jit
from tinymlir.jit import add_target_flags
from tinymlir.ops import gelu, softmax

OPERATIONS = {"gelu": gelu, "softmax": softmax}
STAGES = ("module", "optimized", "lowered")
EXAMPLE_SHAPE = (2, 8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=OPERATIONS)
    parser.add_argument("--stage", choices=STAGES, default=STAGES[0])
    add_target_flags(parser)
    args = parser.parse_args()
    function = jit(OPERATIONS[args.operation])
    compiled = function.compile(np.zeros(EXAMPLE_SHAPE, np.float32), target=args.target)
    print(getattr(compiled, args.stage))


if __name__ == "__main__":
    main()
