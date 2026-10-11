"""A typed tensor graph. Shapes and dtypes are resolved before MLIR construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from math import prod

import numpy as np

FLOAT32 = np.dtype(np.float32)
FLOAT64 = np.dtype(np.float64)
INT32 = np.dtype(np.int32)
SUPPORTED_DTYPES = frozenset((FLOAT32, FLOAT64, INT32))
MASKED_SCORE = -1e10


class Op(StrEnum):
    """Operations understood by the tensor frontend and MLIR lowering."""

    INPUT = auto()
    CONSTANT = auto()
    CAST = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()
    EXP = auto()
    TANH = auto()
    SQRT = auto()
    SUM = auto()
    MAX = auto()
    MATMUL = auto()
    RESHAPE = auto()
    TRANSPOSE = auto()
    SLICE = auto()
    GATHER = auto()
    MASK = auto()


UFUNCS = {
    np.add: Op.ADD,
    np.subtract: Op.SUB,
    np.multiply: Op.MUL,
    np.divide: Op.DIV,
    np.power: Op.POW,
    np.exp: Op.EXP,
    np.tanh: Op.TANH,
    np.sqrt: Op.SQRT,
    np.matmul: Op.MATMUL,
}
FUNCTIONS = {
    np.mean: "mean",
    np.var: "var",
    np.sum: "sum",
    np.max: "max",
    np.amax: "max",
    np.split: "split",
    np.reshape: "reshape",
    np.transpose: "transpose",
}


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]
    dtype: np.dtype

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        if any(d <= 0 for d in self.shape):
            raise ValueError("Tensor dimensions must be positive")
        if self.dtype not in SUPPORTED_DTYPES:
            raise TypeError(f"Unsupported dtype: {self.dtype}")


def dims(args):
    """Accept both NumPy spellings: f(a, b, c) and f((a, b, c))."""
    return tuple(args[0]) if len(args) == 1 and not isinstance(args[0], int) else tuple(args)


@dataclass(eq=False)
class Expr:
    op: Op
    type: TensorType
    args: tuple[Expr, ...] = ()
    attr: object = None

    @property
    def shape(self):
        return self.type.shape

    @property
    def dtype(self):
        return self.type.dtype

    def __bool__(self):
        raise TypeError("Data-dependent Python control flow cannot be traced")

    def __len__(self):
        return self.shape[0]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs or ufunc not in UFUNCS:
            return NotImplemented
        dtype = next(i.dtype for i in inputs if isinstance(i, Expr))
        first, *rest = (as_expr(i, dtype) for i in inputs)
        if not rest:
            return first.unary(UFUNCS[ufunc])
        return first @ rest[0] if ufunc is np.matmul else first.binary(UFUNCS[ufunc], rest[0])

    def __array_function__(self, func, types, args, kwargs):
        if func not in FUNCTIONS:
            return NotImplemented
        x, *rest = args
        return getattr(x, FUNCTIONS[func])(*rest, **kwargs)

    def astype(self, dtype):
        dtype = np.dtype(dtype)
        if dtype == self.dtype:
            return self
        if not np.issubdtype(self.dtype, np.floating) or not np.issubdtype(dtype, np.floating):
            raise TypeError("Only floating-point casts are supported")
        return Expr(Op.CAST, TensorType(self.shape, dtype), (self,))

    def binary(self, op, other):
        other = as_expr(other, self.dtype)
        dtype = np.result_type(self.dtype, other.dtype)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("Arithmetic requires floating-point tensors")
        shape = np.broadcast_shapes(self.shape, other.shape)
        return Expr(op, TensorType(shape, dtype), (self.astype(dtype), other.astype(dtype)))

    def __add__(self, other):
        return self.binary(Op.ADD, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self.binary(Op.SUB, other)

    def __rsub__(self, other):
        return as_expr(other, self.dtype) - self

    def __mul__(self, other):
        return self.binary(Op.MUL, other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self.binary(Op.DIV, other)

    def __rtruediv__(self, other):
        return as_expr(other, self.dtype) / self

    def __pow__(self, other):
        return self.binary(Op.POW, other)

    def __neg__(self):
        return self * -1

    def __matmul__(self, other):
        other = as_expr(other, self.dtype)
        if len(self.shape) not in (2, 3):
            raise ValueError("Matmul accepts matrices or equally batched matrices")
        if not np.issubdtype(self.dtype, np.floating) or not np.issubdtype(
            other.dtype, np.floating
        ):
            raise TypeError("Matmul requires floating-point tensors")
        if len(other.shape) != len(self.shape) or self.shape[:-2] != other.shape[:-2]:
            raise ValueError("Matmul batch dimensions must agree")
        if self.shape[-1] != other.shape[-2]:
            raise ValueError("Matmul reduction dimensions must agree")
        dtype = np.result_type(self.dtype, other.dtype)
        shape = self.shape[:-1] + (other.shape[-1],)
        return Expr(Op.MATMUL, TensorType(shape, dtype), (self.astype(dtype), other.astype(dtype)))

    def unary(self, name):
        if not np.issubdtype(self.dtype, np.floating):
            raise TypeError("Math operations require floating-point tensors")
        return Expr(name, self.type, (self,))

    def reduce(self, name, axis=-1, keepdims=False):
        rank = len(self.shape)
        if not isinstance(axis, int) or not -rank <= axis < rank:
            raise ValueError("Reduction axis is outside the tensor rank")
        axis %= rank
        shape = self.shape[:axis] + self.shape[axis + 1 :]
        result = Expr(name, TensorType(shape, self.dtype), (self,), axis)
        if keepdims:
            result = result.reshape(self.shape[:axis] + (1,) + self.shape[axis + 1 :])
        return result

    def sum(self, axis=-1, keepdims=False):
        return self.reduce(Op.SUM, axis, keepdims)

    def max(self, axis=-1, keepdims=False):
        return self.reduce(Op.MAX, axis, keepdims)

    def mean(self, axis=-1, keepdims=False):
        return self.sum(axis, keepdims) / self.shape[axis]

    def var(self, axis=-1, keepdims=False):
        centered = self - self.mean(axis, keepdims=True)
        return (centered * centered).mean(axis, keepdims)

    def reshape(self, *shape):
        shape = dims(shape)
        if prod(shape) != prod(self.shape) or any(d <= 0 for d in shape):
            raise ValueError("Reshape must preserve the number of elements")
        return (
            self
            if shape == self.shape
            else Expr(Op.RESHAPE, TensorType(shape, self.dtype), (self,))
        )

    def transpose(self, *axes):
        axes = tuple(reversed(range(len(self.shape)))) if not axes else dims(axes)
        if sorted(axes) != list(range(len(self.shape))):
            raise ValueError("Transpose requires a permutation of the axes")
        return Expr(
            Op.TRANSPOSE, TensorType(tuple(self.shape[i] for i in axes), self.dtype), (self,), axes
        )

    @property
    def T(self):
        return self.transpose()

    def split(self, sections, axis=-1):
        axis %= len(self.shape)
        size, remainder = divmod(self.shape[axis], sections)
        if remainder:
            raise ValueError("Split must divide the axis evenly")
        return [
            self[(slice(None),) * axis + (slice(i * size, (i + 1) * size),)]
            for i in range(sections)
        ]

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        keys = tuple(slice(k.start, k.stop, k.step) if isinstance(k, range) else k for k in keys)
        if len(keys) == 1 and not isinstance(keys[0], slice):
            return gather(self, keys[0])
        keys += (slice(None),) * (len(self.shape) - len(keys))
        if len(keys) != len(self.shape) or any(not isinstance(k, slice) for k in keys):
            raise TypeError("Index with static slices or a single int32 row vector")
        bounds = tuple(k.indices(d) for k, d in zip(keys, self.shape, strict=True))
        if any(step != 1 or stop <= start for start, stop, step in bounds):
            raise ValueError("Slices must be nonempty and have unit stride")
        shape = tuple(stop - start for start, stop, _ in bounds)
        return Expr(
            Op.SLICE, TensorType(shape, self.dtype), (self,), tuple(a for a, _, _ in bounds)
        )


def as_array(value):
    """Integer data becomes int32, the only integer type the kernels accept."""
    array = np.asarray(value)
    integer = np.issubdtype(array.dtype, np.integer) and array.dtype != INT32
    return array.astype(INT32) if integer else array


def as_expr(value, dtype):
    if isinstance(value, Expr):
        return value
    if np.isscalar(value):
        dtype = np.asarray(value).dtype if isinstance(value, np.generic) else np.dtype(dtype)
        return Expr(Op.CONSTANT, TensorType((), dtype), attr=value)
    array = as_array(value)
    return Expr(Op.CONSTANT, TensorType(array.shape, array.dtype), attr=array)


def exp(x):
    return x.unary(Op.EXP)


def tanh(x):
    return x.unary(Op.TANH)


def sqrt(x):
    return x.unary(Op.SQRT)


def gather(table, ids):
    ids = as_expr(ids, INT32)
    if ids.op not in (Op.INPUT, Op.CONSTANT):
        raise ValueError("Gather indices must be a function argument or a constant")
    if len(table.shape) != 2 or len(ids.shape) != 1 or ids.dtype != INT32:
        raise ValueError("Gather requires a matrix and an int32 index vector")
    return Expr(Op.GATHER, TensorType((ids.shape[0], table.shape[1]), table.dtype), (table, ids))


def causal_mask(size, dtype):
    return Expr(Op.MASK, TensorType((size, size), dtype))
