"""Hand bufferized matmuls to the system BLAS instead of lowering them to scalar loops."""

import ctypes
import ctypes.util

from mlir import ir
from mlir.dialects import arith, func, linalg, llvm, memref, scf

ROW_MAJOR, NO_TRANSPOSE = 101, 111
GEMM = {"f32": "cblas_sgemm", "f64": "cblas_dgemm"}
CANDIDATES = ("/System/Library/Frameworks/Accelerate.framework/Accelerate", "openblas", "blas")


def locate():
    """Load the first BLAS found into the process, where the JIT resolves symbols from."""
    for name in CANDIDATES:
        path = name if name.startswith("/") else ctypes.util.find_library(name)
        try:
            library = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL) if path else None
        except OSError:
            continue
        if hasattr(library, "cblas_sgemm"):
            return library
    return None


LIBRARY = locate()


def i32(value):
    typ = ir.IntegerType.get_signless(32)
    return (
        arith.constant(typ, ir.IntegerAttr.get(typ, value))
        if isinstance(value, int)
        else arith.index_cast(typ, value)
    )


def index(value):
    typ = ir.IndexType.get()
    return arith.constant(typ, ir.IntegerAttr.get(typ, value))


def pointer(source, offset):
    """Address of element `offset` within a memref's aligned buffer, as an opaque pointer."""
    itemsize = ir.MemRefType(source.type).element_type.width // 8
    scaled = arith.muli(offset, index(itemsize))
    address = arith.addi(memref.extract_aligned_pointer_as_index(source), scaled)
    i64 = ir.IntegerType.get_signless(64)
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr"), arith.index_cast(i64, address))


def row_major(value):
    strides, _ = ir.MemRefType(value.type).get_strides_and_offset()
    return strides[-1] == 1


def declare(module, name, element):
    for operation in module.body.operations:
        if isinstance(operation, func.FuncOp) and operation.name.value == name:
            return
    i32, ptr = ir.IntegerType.get_signless(32), ir.Type.parse("!llvm.ptr")
    types = [i32] * 6 + [element, ptr, i32, ptr, i32, element, ptr, i32]
    with ir.InsertionPoint(module.body):
        func.FuncOp(name, (types, []), visibility="private")


def gemm(name, operands, batch=None):
    """Emit C += A @ B for one batch index, or for the whole matrices when batch is None."""
    metadata = [memref.ExtractStridedMetadataOp(v) for v in operands]
    offsets = [m.offset for m in metadata]
    if batch is not None:
        offsets = [arith.addi(m.offset, arith.muli(batch, m.strides[0])) for m in metadata]
    a, b, c = (pointer(v, offset) for v, offset in zip(operands, offsets, strict=True))
    lda, ldb, ldc = (i32(m.strides[-2]) for m in metadata)
    m, k = metadata[0].sizes[-2:]
    n = metadata[1].sizes[-1]
    element = ir.MemRefType(operands[0].type).element_type
    one = arith.constant(element, ir.FloatAttr.get(element, 1.0))
    layout = [i32(ROW_MAJOR), i32(NO_TRANSPOSE), i32(NO_TRANSPOSE), i32(m), i32(n), i32(k)]
    func.CallOp([], ir.FlatSymbolRefAttr.get(name), [*layout, one, a, lda, b, ldb, one, c, ldc])


def rewrite(module):
    """Replace every row-major linalg.matmul and batch_matmul with a cblas gemm call."""
    matmuls = []

    def collect(operation):
        if isinstance(operation.opview, (linalg.MatmulOp, linalg.BatchMatmulOp)):
            matmuls.append(operation.opview)
        return ir.WalkResult.ADVANCE

    module.operation.walk(collect)
    for op in matmuls:
        operands = [*op.inputs, *op.outputs]
        if not all(row_major(v) for v in operands):
            continue
        element = ir.MemRefType(operands[0].type).element_type
        name = GEMM[str(element)]
        declare(module, name, element)
        with ir.InsertionPoint(op):
            if isinstance(op, linalg.MatmulOp):
                gemm(name, operands)
            else:
                zero, batches, one = (index(v) for v in (0, operands[0].type.shape[0], 1))
                loop = scf.ForOp(zero, batches, one)
                with ir.InsertionPoint(loop.body):
                    gemm(name, operands, loop.induction_variable)
                    scf.YieldOp([])
        op.operation.erase()
