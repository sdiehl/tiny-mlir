"""Shape-specialized tracing, MLIR passes, and native execution."""

import argparse
import ctypes
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from functools import update_wrapper
from pathlib import Path

import mlir._mlir_libs
import numpy as np
from mlir import ir
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
from mlir.runtime import get_ranked_memref_descriptor

from . import blas, gpu
from .builder import KERNEL_NAME, Builder
from .expr import Expr, Op, TensorType, as_array

CPU = "cpu"
GPU = "gpu"
TARGETS = (CPU, GPU)

OPTIMIZATION_PIPELINE = (
    "builtin.module(canonicalize,cse,linalg-fuse-elementwise-ops,canonicalize,cse)"
)
BUFFERIZATION_PIPELINE = (
    "builtin.module("
    "one-shot-bufferize{bufferize-function-boundaries},buffer-deallocation-pipeline)"
)
LOWERING_PIPELINE = (
    "builtin.module("
    "convert-linalg-to-loops,"
    "expand-strided-metadata,lower-affine,"
    "convert-scf-to-cf,convert-math-to-libm,convert-math-to-llvm,"
    "convert-arith-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,"
    "finalize-memref-to-llvm,reconcile-unrealized-casts)"
)

LLVM_OPTIMIZATION_LEVEL = 3
# Strided memref copies call into this library.
RUNNER_UTILS = next(Path(mlir._mlir_libs.__file__).parent.glob("libmlir_c_runner_utils.*"))


def flatten(tree):
    """Split nested tuples, dataclasses and dicts into array leaves and a hashable structure."""
    if isinstance(tree, Expr):
        return [tree], tree.type
    if isinstance(tree, dict):
        keys, children = tuple(tree), tree.values()
    elif is_dataclass(tree):
        keys = tuple(field.name for field in fields(tree))
        children = (getattr(tree, key) for key in keys)
    elif isinstance(tree, tuple):
        keys, children = None, tree
    else:
        array = as_array(tree)
        return [array], TensorType(array.shape, array.dtype)
    parts = [flatten(child) for child in children]
    leaves = [leaf for part, _ in parts for leaf in part]
    return leaves, (type(tree), keys, tuple(structure for _, structure in parts))


def unflatten(structure, leaves):
    if isinstance(structure, TensorType):
        return next(leaves)
    typ, keys, children = structure
    values = [unflatten(child, leaves) for child in children]
    if keys is None:
        return typ(*values) if hasattr(typ, "_fields") else typ(values)
    pairs = zip(keys, values, strict=True)
    return typ(**dict(pairs)) if is_dataclass(typ) else typ(pairs)


def leaf_types(structure):
    if isinstance(structure, TensorType):
        yield structure
    else:
        for child in structure[2]:
            yield from leaf_types(child)


def clone_module(source: ir.Module) -> ir.Module:
    result = ir.Module.create()
    with ir.InsertionPoint(result.body):
        for operation in source.body.operations:
            operation.clone()
    return result


class Compiled:
    def __init__(self, function: Callable, structure, static: dict, optimize=True, target=CPU):
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}")
        self.structure = structure
        self.target = target
        types = tuple(leaf_types(structure))
        inputs = [Expr(Op.INPUT, typ, attr=i) for i, typ in enumerate(types)]
        output = function(*unflatten(structure, iter(inputs)), **static)
        if not isinstance(output, Expr):
            raise TypeError("A compiled function must return one tensor expression")
        self.result_type = output.type
        self.constants = []
        self.gathers = []
        visited = set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for arg in node.args:
                visit(arg)
            if node.op is Op.CONSTANT and isinstance(node.attr, np.ndarray):
                # Captured arrays become trailing kernel arguments.
                self.constants.append(node.attr)
                node.op, node.attr = Op.INPUT, len(inputs)
                inputs.append(node)
            if node.op is Op.GATHER:
                self.gathers.append((node.args[1].attr, node.args[0].shape[0]))

        visit(output)
        self.context = ir.Context()
        with self.context, ir.Location.unknown():
            self.module = Builder().build(inputs, output)
            # Preserve both forms for inspecting what the compiler changed.
            self.optimized = clone_module(self.module)
            if optimize:
                PassManager.parse(OPTIMIZATION_PIPELINE).run(self.optimized.operation)
            self.optimized.operation.verify()
            self.lowered = clone_module(self.optimized)
            # Storage is assigned and released once here. Matmuls keep their structure through
            # it, so each target substitutes its own implementation before the rest lowers.
            PassManager.parse(BUFFERIZATION_PIPELINE).run(self.lowered.operation)
            if target == GPU:
                gpu.lower(self.lowered)
            else:
                blas.rewrite(self.lowered)
                PassManager.parse(LOWERING_PIPELINE).run(self.lowered.operation)
            self.lowered.operation.verify()
            self.engine = ExecutionEngine(
                self.lowered,
                opt_level=LLVM_OPTIMIZATION_LEVEL,
                shared_libs=[str(RUNNER_UTILS), *([str(gpu.RUNTIME)] if target == GPU else [])],
            )
            # Global constructors load the embedded PTX modules.
            self.engine.initialize()

    def __call__(self, *args):
        leaves, structure = flatten(args)
        if structure != self.structure:
            raise TypeError("Arguments do not match this compiled specialization")
        arrays = [np.ascontiguousarray(a) for a in [*leaves, *self.constants]]
        for index, bound in self.gathers:
            if np.any(arrays[index] < 0) or np.any(arrays[index] >= bound):
                raise ValueError("Gather index outside the table")
        # restrict on to_tensor requires distinct input storage. Duplicate/overlapping
        # inputs are copied at the boundary, leaving the function itself purely functional.
        for i, array in enumerate(arrays):
            if any(np.may_share_memory(array, previous) for previous in arrays[:i]):
                arrays[i] = array.copy()
        if self.target == GPU:
            arrays = [gpu.device(a) for a in arrays]
            output = gpu.empty(self.result_type.shape, self.result_type.dtype)
        else:
            output = np.empty(self.result_type.shape, self.result_type.dtype)
        pointers = [
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(a)))
            for a in [*arrays, output]
        ]
        self.engine.invoke(KERNEL_NAME, *pointers)
        return output


class Jit:
    def __init__(self, function: Callable):
        update_wrapper(self, function)
        self.function = function
        self.specializations = {}

    def compile(self, *args, target=CPU, **static) -> Compiled:
        _, structure = flatten(args)
        key = (structure, tuple(sorted(static.items())), target)
        if key not in self.specializations:
            self.specializations[key] = Compiled(self.function, structure, static, target=target)
        return self.specializations[key]

    def __call__(self, *args, target=CPU, **static):
        if any(isinstance(leaf, Expr) for leaf in flatten(args)[0]):
            return self.function(*args, **static)
        return self.compile(*args, target=target, **static)(*args)


def add_target_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpu", dest="target", action="store_const", const=CPU, default=CPU)
    group.add_argument("--gpu", dest="target", action="store_const", const=GPU, help="NVIDIA")
    parser.add_argument(
        "--fused", action="store_true", help="compile the whole forward pass as one kernel"
    )


def jit(function: Callable) -> Jit:
    """Compile a pure tensor function, specializing on argument types and static keywords."""
    return Jit(function)
