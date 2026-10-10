"""NVIDIA lowering: parallel loops become CUDA kernels over managed memory."""

import ctypes
from functools import cache
from importlib.util import find_spec
from pathlib import Path

import mlir._mlir_libs
import numpy as np
from mlir import ir
from mlir.dialects import gpu
from mlir.passmanager import PassManager

CHIP = "sm_75"
TILE_SIZES = "16,16"
RUNTIME = Path(mlir._mlir_libs.__file__).with_name("libmlir_cuda_runtime.so")
DRIVER = "libcuda.so.1"

KERNEL_PIPELINE = (
    "builtin.module("
    "one-shot-bufferize{bufferize-function-boundaries},"
    "convert-linalg-to-parallel-loops,"
    "buffer-deallocation-pipeline,"
    f"func.func(scf-parallel-loop-tiling{{parallel-loop-tile-sizes={TILE_SIZES} "
    "no-min-max-bounds=true}),"
    "func.func(gpu-map-parallel-loops),"
    "convert-parallel-loops-to-gpu,"
    "gpu-kernel-outlining,canonicalize)"
)


def toolkit() -> Path | None:
    # libdevice supplies exp, tanh and friends; NVIDIA's pip package ships it with ptxas.
    spec = find_spec("nvidia") and find_spec("nvidia.cuda_nvcc")
    return Path(spec.submodule_search_locations[0]) if spec else None


def llvm_pipeline():
    path = toolkit()
    option = f" toolkit={path}" if path else ""
    return (
        "builtin.module("
        "expand-strided-metadata,lower-affine,convert-scf-to-cf,"
        f"nvvm-attach-target{{chip={CHIP} O=3}},"
        "gpu.module(convert-gpu-to-nvvm),"
        "gpu-to-llvm,finalize-memref-to-llvm,"
        "convert-arith-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,convert-index-to-llvm,"
        "reconcile-unrealized-casts,"
        f"gpu-module-to-binary{{format=isa{option}}})"
    )


def managed_allocations(module: ir.Module) -> None:
    """Host-shared buffers are visible to kernels and to the host code between launches."""
    operations = []
    module.operation.walk(lambda op: operations.append(op) or ir.WalkResult.ADVANCE)
    for op in operations:
        if op.name not in ("memref.alloc", "memref.dealloc"):
            continue
        with ir.InsertionPoint(op), op.location:
            if op.name == "memref.alloc":
                alloc = gpu.AllocOp(op.results[0].type, None, [], [], [], hostShared=True)
                op.results[0].replace_all_uses_with(alloc.result)
            else:
                # Only the asynchronous form of gpu.dealloc lowers to the runtime.
                token = gpu.AsyncTokenType.get()
                stream = gpu.WaitOp(token, []).asyncToken
                gpu.WaitOp(None, [gpu.DeallocOp(token, [stream], op.operands[0]).asyncToken])
        op.erase()


def outline(module: ir.Module) -> None:
    PassManager.parse(KERNEL_PIPELINE).run(module.operation)
    managed_allocations(module)


def lower(module: ir.Module) -> None:
    if not RUNTIME.exists():
        raise RuntimeError("The GPU target needs the CUDA build of MLIR (Linux x86_64)")
    outline(module)
    PassManager.parse(llvm_pipeline()).run(module.operation)


@cache
def available() -> bool:
    """Whether this machine can both compile for and run on an NVIDIA GPU."""
    if not RUNTIME.exists():
        return False
    try:
        driver = ctypes.CDLL(DRIVER)
    except OSError:
        return False
    count = ctypes.c_int()
    return (
        driver.cuInit(0) == 0
        and driver.cuDeviceGetCount(ctypes.byref(count)) == 0
        and count.value > 0
    )


@cache
def runtime() -> ctypes.CDLL:
    library = ctypes.CDLL(str(RUNTIME))
    library.mgpuMemAlloc.restype = ctypes.c_void_p
    library.mgpuMemAlloc.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_bool]
    library.mgpuMemFree.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return library


class Managed:
    """CUDA unified memory, exposed to NumPy through the array interface."""

    def __init__(self, shape, dtype):
        dtype = np.dtype(dtype)
        self.pointer = runtime().mgpuMemAlloc(
            max(int(np.prod(shape)), 1) * dtype.itemsize, None, True
        )
        self.__array_interface__ = {
            "shape": tuple(shape),
            "typestr": dtype.str,
            "data": (self.pointer, False),
            "version": 3,
        }

    def __del__(self):
        runtime().mgpuMemFree(self.pointer, None)


def empty(shape, dtype) -> np.ndarray:
    return np.asarray(Managed(shape, dtype))


def device(array) -> np.ndarray:
    """Place an array in managed memory, leaving arrays already there untouched."""
    array = np.asarray(array)
    if isinstance(array.base, Managed) and array.flags.c_contiguous:
        return array
    result = empty(array.shape, array.dtype)
    result[...] = array
    return result
