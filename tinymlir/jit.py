"""Shape-specialized tracing, MLIR passes, and native execution."""

import ctypes
from functools import update_wrapper
import numpy as np
from mlir import ir
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
from mlir.runtime import get_ranked_memref_descriptor
from .builder import Builder
from .expr import Expr, TensorType


OPTIMIZE = 'builtin.module(canonicalize,cse,linalg-fuse-elementwise-ops,canonicalize,cse)'
LOWER = (
    'builtin.module('
    'one-shot-bufferize{bufferize-function-boundaries},'
    'convert-linalg-to-loops,'
    'buffer-deallocation-pipeline,'
    'expand-strided-metadata,lower-affine,'
    'convert-scf-to-cf,convert-math-to-libm,convert-math-to-llvm,'
    'convert-arith-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,'
    'finalize-memref-to-llvm,reconcile-unrealized-casts)'
)


def clone_module(source):
    result = ir.Module.create()
    with ir.InsertionPoint(result.body):
        for operation in source.body.operations:
            operation.clone()
    return result


class Compiled:
    def __init__(self, function, types, static, optimize=True):
        self.types = types
        inputs = tuple(Expr('input', typ, attr=i) for i, typ in enumerate(types))
        output = function(*inputs, **static)
        if not isinstance(output, Expr):
            raise TypeError('A compiled function must return one tensor expression')
        self.result_type = output.type
        self.context = ir.Context()
        with self.context, ir.Location.unknown():
            self.module = Builder().build(inputs, output)
            # Preserve both forms for inspecting what the compiler changed.
            self.optimized = clone_module(self.module)
            if optimize:
                PassManager.parse(OPTIMIZE).run(self.optimized.operation)
            self.optimized.operation.verify()
            self.lowered = clone_module(self.optimized)
            PassManager.parse(LOWER).run(self.lowered.operation)
            self.lowered.operation.verify()
            self.engine = ExecutionEngine(self.lowered, opt_level=3)

    def __call__(self, *args):
        arrays = [np.ascontiguousarray(a) for a in args]
        if tuple(TensorType(a.shape, a.dtype) for a in arrays) != self.types:
            raise TypeError('Arguments do not match this compiled specialization')
        # restrict on to_tensor requires distinct input storage. Duplicate/overlapping
        # inputs are copied at the boundary, leaving the function itself purely functional.
        for i, array in enumerate(arrays):
            if any(np.shares_memory(array, previous) for previous in arrays[:i]):
                arrays[i] = array.copy()
        output = np.empty(self.result_type.shape, self.result_type.dtype)
        pointers = [ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(a)))
                    for a in [*arrays, output]]
        self.engine.invoke('kernel', *pointers)
        return output


class Jit:
    def __init__(self, function):
        update_wrapper(self, function)
        self.function = function
        self.specializations = {}

    def compile(self, *args, **static):
        types = tuple(TensorType(a.shape, a.dtype) for a in args)
        key = (types, tuple(sorted(static.items())))
        if key not in self.specializations:
            self.specializations[key] = Compiled(self.function, types, static)
        return self.specializations[key]

    def __call__(self, *args, **static):
        if any(isinstance(a, Expr) for a in args):
            return self.function(*args, **static)
        return self.compile(*args, **static)(*args)


jit = Jit
