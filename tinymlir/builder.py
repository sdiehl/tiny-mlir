"""Lower typed expressions to tensor and linalg operations using MLIR builders."""

from math import prod
from mlir import ir
from mlir.dialects import arith, bufferization, func, linalg, math, tensor


def element_type(dtype):
    return {'float32': ir.F32Type.get, 'float64': ir.F64Type.get,
            'int32': lambda: ir.IntegerType.get_signless(32)}[str(dtype)]()


def tensor_type(typ):
    return ir.RankedTensorType.get(typ.shape, element_type(typ.dtype))


def scalar(value, typ):
    attr = ir.FloatAttr.get(typ, float(value)) if ir.FloatType.isinstance(typ) else ir.IntegerAttr.get(typ, int(value))
    return arith.ConstantOp(typ, attr).result


def empty(typ):
    return tensor.EmptyOp(typ.shape, element_type(typ.dtype)).result


def affine_map(rank, results):
    return ir.AffineMap.get(rank, 0, results)


class Builder:
    def __init__(self):
        self.values = {}

    def generic(self, inputs, output, maps, body):
        rank = len(output.type.shape)
        op = linalg.GenericOp([output.type], inputs, [output], maps, ['parallel'] * rank)
        block = op.regions[0].blocks.append(*(v.type.element_type for v in [*inputs, output]))
        with ir.InsertionPoint(block):
            linalg.YieldOp([body(*block.arguments[:-1])])
        return op.result

    def elementwise(self, expr, operands):
        rank = len(expr.shape)
        dims = [ir.AffineDimExpr.get(i) for i in range(rank)]
        maps = []
        for arg in expr.args:
            offset = rank - len(arg.shape)
            indices = [ir.AffineConstantExpr.get(0) if size == 1 else dims[offset+i]
                       for i, size in enumerate(arg.shape)]
            maps.append(affine_map(rank, indices))
        maps.append(affine_map(rank, dims))

        def body(*args):
            match expr.op:
                case 'add': return arith.AddFOp(*args).result
                case 'sub': return arith.SubFOp(*args).result
                case 'mul': return arith.MulFOp(*args).result
                case 'div': return arith.DivFOp(*args).result
                case 'pow': return math.PowFOp(*args).result
                case 'exp': return math.ExpOp(*args).result
                case 'tanh': return math.TanhOp(*args).result
                case 'sqrt': return math.SqrtOp(*args).result
                case 'cast':
                    ctor = arith.ExtFOp if expr.dtype.itemsize > expr.args[0].dtype.itemsize else arith.TruncFOp
                    return ctor(element_type(expr.dtype), args[0]).result
                case _: raise ValueError(expr.op)

        return self.generic(operands, empty(expr.type), maps, body)

    def reduction(self, expr, operand):
        typ = element_type(expr.dtype)
        init = scalar(0 if expr.op == 'sum' else float('-inf'), typ)
        filled = linalg.fill(init, outs=[empty(expr.type)])
        op = linalg.ReduceOp([tensor_type(expr.type)], [operand], [filled], [expr.attr])
        block = op.regions[0].blocks.append(typ, typ)
        with ir.InsertionPoint(block):
            ctor = arith.AddFOp if expr.op == 'sum' else arith.MaximumFOp
            linalg.YieldOp([ctor(*block.arguments).result])
        return op.result

    def reshape(self, expr, operand):
        source_shape = expr.args[0].shape
        typ = element_type(expr.dtype)
        if not source_shape or not expr.shape:
            # Rank-zero tensors and unit tensors have the same single element.
            rank = len(expr.shape)
            maps = [affine_map(rank, [ir.AffineConstantExpr.get(0)] * len(source_shape)),
                    ir.AffineMap.get_identity(rank)]
            return self.generic([operand], empty(expr.type), maps, lambda x: x)
        if len(source_shape) > 1:
            flat = ir.RankedTensorType.get([prod(source_shape)], typ)
            operand = tensor.CollapseShapeOp(flat, operand, [list(range(len(source_shape)))]).result
        if len(expr.shape) > 1:
            operand = tensor.ExpandShapeOp(tensor_type(expr.type), operand,
                [list(range(len(expr.shape)))], [], expr.shape).result
        return operand

    def lower(self, expr):
        if expr in self.values:
            return self.values[expr]
        args = [self.lower(arg) for arg in expr.args]
        match expr.op:
            case 'constant':
                value = scalar(expr.attr, element_type(expr.dtype))
                result = tensor.FromElementsOp(tensor_type(expr.type), [value]).result
            case 'add' | 'sub' | 'mul' | 'div' | 'pow' | 'exp' | 'tanh' | 'sqrt' | 'cast':
                result = self.elementwise(expr, args)
            case 'sum' | 'max':
                result = self.reduction(expr, args[0])
            case 'matmul':
                init = linalg.fill(scalar(0, element_type(expr.dtype)), outs=[empty(expr.type)])
                operation = linalg.matmul if len(expr.shape) == 2 else linalg.batch_matmul
                result = operation(*args, outs=[init])
            case 'reshape':
                result = self.reshape(expr, args[0])
            case 'transpose':
                result = linalg.transpose(args[0], outs=[empty(expr.type)], permutation=expr.attr)
            case 'slice':
                result = tensor.ExtractSliceOp(tensor_type(expr.type), args[0], [], [], [],
                    expr.attr, expr.shape, [1] * len(expr.shape)).result
            case 'gather':
                # The row lookup is data-dependent; the column dimension remains affine.
                maps = [affine_map(2, [ir.AffineDimExpr.get(0)]), ir.AffineMap.get_identity(2)]
                def lookup(index):
                    row = arith.IndexCastOp(ir.IndexType.get(), index).result
                    col = linalg.IndexOp(1).result
                    return tensor.ExtractOp(args[0], [row, col]).result
                result = self.generic([args[1]], empty(expr.type), maps, lookup)
            case 'mask':
                def mask():
                    row, col = linalg.IndexOp(0).result, linalg.IndexOp(1).result
                    allowed = arith.CmpIOp(arith.CmpIPredicate.sge, row, col).result
                    typ = element_type(expr.dtype)
                    return arith.SelectOp(allowed, scalar(0, typ), scalar(-1e10, typ)).result
                result = self.generic([], empty(expr.type), [ir.AffineMap.get_identity(2)], mask)
            case _:
                raise ValueError(f'Unknown expression: {expr.op}')
        self.values[expr] = result
        return result

    def build(self, inputs, output, name='kernel'):
        module = ir.Module.create()
        types = [ir.MemRefType.get(e.shape, element_type(e.dtype)) for e in (*inputs, output)]
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(name, (types, []))
            function.attributes['llvm.emit_c_interface'] = ir.UnitAttr.get()
            block = function.add_entry_block()
            with ir.InsertionPoint(block):
                for expr, argument in zip(inputs, block.arguments):
                    self.values[expr] = bufferization.ToTensorOp(tensor_type(expr.type), argument,
                        restrict=True).result
                result = self.lower(output)
                bufferization.MaterializeInDestinationOp(None, result, block.arguments[-1], writable=True)
                func.ReturnOp([])
        module.operation.verify()
        return module
