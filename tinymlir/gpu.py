"""NVIDIA lowering: parallel loops become CUDA kernels over managed memory."""

import ctypes
from functools import cache
from importlib.util import find_spec
from pathlib import Path

import mlir._mlir_libs
import numpy as np
from mlir import ir
from mlir.dialects import func, gpu, memref
from mlir.passmanager import PassManager

CHIP = "sm_75"
TILE_SIZES = "16,16"
RUNTIME = Path(mlir._mlir_libs.__file__).with_name("libmlir_cuda_runtime.so")
DRIVER = "libcuda.so.1"

TILE = 16

KERNEL_PIPELINE = (
    "builtin.module("
    "convert-linalg-to-parallel-loops,"
    f"func.func(scf-parallel-loop-tiling{{parallel-loop-tile-sizes={TILE_SIZES} "
    "no-min-max-bounds=true}),"
    "func.func(gpu-map-parallel-loops),"
    "convert-parallel-loops-to-gpu,"
    "gpu-kernel-outlining,canonicalize)"
)


# Each block stages a TILE x TILE slice of A and B in shared memory per step along K,
# so every global element is loaded once per block rather than once per thread.
MATMUL = """
func.func private @{name}(%a: {view}, %b: {view}, %out: {view}) {{
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %i2 = arith.constant 2 : index
  %tile = arith.constant {tile} : index
  %zero = arith.constant 0.0 : {t}
  %m = memref.dim %a, %i{row} : {view}
  %k = memref.dim %a, %i{col} : {view}
  %n = memref.dim %b, %i{col} : {view}
  %batch = memref.dim %a, %i0 : {view}
  %gx = arith.ceildivui %n, %tile : index
  %gy = arith.ceildivui %m, %tile : index
  gpu.launch blocks(%bx, %by, %bz) in (%sx = %gx, %sy = %gy, %sz = {batch})
             threads(%tx, %ty, %tz) in (%ux = %tile, %uy = %tile, %uz = %i1)
             workgroup(%sa : {shared}, %sb : {shared}) {{
    %r = arith.muli %by, %tile : index
    %row = arith.addi %r, %ty : index
    %c = arith.muli %bx, %tile : index
    %col = arith.addi %c, %tx : index
    %rowok = arith.cmpi ult, %row, %m : index
    %colok = arith.cmpi ult, %col, %n : index
    %sum = scf.for %k0 = %i0 to %k step %tile iter_args(%acc = %zero) -> ({t}) {{
      %ka = arith.addi %k0, %tx : index
      %kaok = arith.cmpi ult, %ka, %k : index
      %aok = arith.andi %rowok, %kaok : i1
      %av = scf.if %aok -> ({t}) {{
        %v = memref.load %a[{z}%row, %ka] : {view}
        scf.yield %v : {t}
      }} else {{
        scf.yield %zero : {t}
      }}
      memref.store %av, %sa[%ty, %tx] : {shared}
      %kb = arith.addi %k0, %ty : index
      %kbok = arith.cmpi ult, %kb, %k : index
      %bok = arith.andi %kbok, %colok : i1
      %bv = scf.if %bok -> ({t}) {{
        %v = memref.load %b[{z}%kb, %col] : {view}
        scf.yield %v : {t}
      }} else {{
        scf.yield %zero : {t}
      }}
      memref.store %bv, %sb[%ty, %tx] : {shared}
      gpu.barrier
      %part = scf.for %kk = %i0 to %tile step %i1 iter_args(%s = %acc) -> ({t}) {{
        %x = memref.load %sa[%ty, %kk] : {shared}
        %y = memref.load %sb[%kk, %tx] : {shared}
        %p = arith.mulf %x, %y : {t}
        %next = arith.addf %s, %p : {t}
        scf.yield %next : {t}
      }}
      gpu.barrier
      scf.yield %part : {t}
    }}
    %ok = arith.andi %rowok, %colok : i1
    scf.if %ok {{
      %old = memref.load %out[{z}%row, %col] : {view}
      %new = arith.addf %old, %sum : {t}
      memref.store %new, %out[{z}%row, %col] : {view}
    }}
    gpu.terminator
  }}
  return
}}
"""


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


def matmul_kernel(module: ir.Module, rank: int, element: ir.Type) -> ir.Operation:
    name = f"tiled_matmul_{rank}d_{element}"
    for op in module.body.operations:
        if op.operation.name == "func.func" and op.sym_name.value == name:
            return op
    source = MATMUL.format(
        name=name,
        t=element,
        tile=TILE,
        view=f"memref<{'?x' * rank}{element}, strided<[{', '.join('?' * rank)}], offset: ?>>",
        shared=f"memref<{TILE}x{TILE}x{element}, #gpu.address_space<workgroup>>",
        row=rank - 2,
        col=rank - 1,
        batch="%batch" if rank == 3 else "%i1",
        z="%bz, " if rank == 3 else "",
    )
    parsed = ir.Module.parse(source)
    kernel = parsed.body.operations[0]
    kernel.move_before(module.body.operations[0])
    return kernel


def tiled_matmuls(module: ir.Module) -> None:
    """Replace floating-point matmuls with calls to the shared-memory kernel."""
    operations = []
    module.operation.walk(lambda op: operations.append(op) or ir.WalkResult.ADVANCE)
    for op in operations:
        if op.name not in ("linalg.matmul", "linalg.batch_matmul"):
            continue
        view = ir.MemRefType(op.operands[0].type)
        if not ir.FloatType.isinstance(view.element_type):
            continue
        kernel = matmul_kernel(module, view.rank, view.element_type)
        with ir.InsertionPoint(op), op.location:
            operands = [
                memref.CastOp(t, v).result
                for t, v in zip(kernel.type.inputs, op.operands, strict=True)
            ]
            func.CallOp([], kernel.sym_name.value, operands)
        op.erase()


def outline(module: ir.Module) -> None:
    """Lower a bufferized module: matmuls become the tiled kernel, other loops become kernels."""
    tiled_matmuls(module)
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
