"""A typed tensor graph. Shapes and dtypes are resolved before MLIR construction."""

from dataclasses import dataclass
from math import prod

import numpy as np


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]
    dtype: np.dtype

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        if any(d <= 0 for d in self.shape):
            raise ValueError("Tensor dimensions must be positive")
        if self.dtype not in (np.dtype("float32"), np.dtype("float64"), np.dtype("int32")):
            raise TypeError(f"Unsupported dtype: {self.dtype}")


@dataclass(eq=False)
class Expr:
    op: str
    type: TensorType
    args: tuple = ()
    attr: object = None

    __array_priority__ = 1000

    @property
    def shape(self):
        return self.type.shape

    @property
    def dtype(self):
        return self.type.dtype

    def __bool__(self):
        raise TypeError("Data-dependent Python control flow cannot be traced")

    def astype(self, dtype):
        dtype = np.dtype(dtype)
        if dtype == self.dtype:
            return self
        if self.dtype.kind != "f" or dtype.kind != "f":
            raise TypeError("Only floating-point casts are supported")
        return Expr("cast", TensorType(self.shape, dtype), (self,))

    def binary(self, op, other):
        other = as_expr(other, self.dtype)
        dtype = np.result_type(self.dtype, other.dtype)
        if dtype.kind != "f":
            raise TypeError("Arithmetic requires floating-point tensors")
        shape = np.broadcast_shapes(self.shape, other.shape)
        return Expr(op, TensorType(shape, dtype), (self.astype(dtype), other.astype(dtype)))

    def __add__(self, other):
        return self.binary("add", other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self.binary("sub", other)

    def __rsub__(self, other):
        return as_expr(other, self.dtype) - self

    def __mul__(self, other):
        return self.binary("mul", other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self.binary("div", other)

    def __rtruediv__(self, other):
        return as_expr(other, self.dtype) / self

    def __pow__(self, other):
        return self.binary("pow", other)

    def __neg__(self):
        return self * -1

    def __matmul__(self, other):
        if not isinstance(other, Expr) or len(self.shape) not in (2, 3):
            raise ValueError("Matmul accepts matrices or equally batched matrices")
        if self.dtype.kind != "f" or other.dtype.kind != "f":
            raise TypeError("Matmul requires floating-point tensors")
        if len(other.shape) != len(self.shape) or self.shape[:-2] != other.shape[:-2]:
            raise ValueError("Matmul batch dimensions must agree")
        if self.shape[-1] != other.shape[-2]:
            raise ValueError("Matmul reduction dimensions must agree")
        dtype = np.result_type(self.dtype, other.dtype)
        shape = self.shape[:-1] + (other.shape[-1],)
        return Expr("matmul", TensorType(shape, dtype), (self.astype(dtype), other.astype(dtype)))

    def unary(self, name):
        if self.dtype.kind != "f":
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
        return self.reduce("sum", axis, keepdims)

    def max(self, axis=-1, keepdims=False):
        return self.reduce("max", axis, keepdims)

    def mean(self, axis=-1, keepdims=False):
        return self.sum(axis, keepdims) / self.shape[axis]

    def reshape(self, shape):
        shape = tuple(shape)
        if prod(shape) != prod(self.shape) or any(d <= 0 for d in shape):
            raise ValueError("Reshape must preserve the number of elements")
        return (
            self if shape == self.shape else Expr("reshape", TensorType(shape, self.dtype), (self,))
        )

    def transpose(self, axes=None):
        axes = tuple(reversed(range(len(self.shape)))) if axes is None else tuple(axes)
        if sorted(axes) != list(range(len(self.shape))):
            raise ValueError("Transpose requires a permutation of the axes")
        return Expr(
            "transpose", TensorType(tuple(self.shape[i] for i in axes), self.dtype), (self,), axes
        )

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        keys += (slice(None),) * (len(self.shape) - len(keys))
        if len(keys) != len(self.shape) or any(not isinstance(k, slice) for k in keys):
            raise TypeError("Use static slices; gather() handles tensor indices")
        bounds = tuple(k.indices(d) for k, d in zip(keys, self.shape, strict=True))
        if any(step != 1 or stop <= start for start, stop, step in bounds):
            raise ValueError("Slices must be nonempty and have unit stride")
        shape = tuple(stop - start for start, stop, _ in bounds)
        return Expr("slice", TensorType(shape, self.dtype), (self,), tuple(a for a, _, _ in bounds))


def as_expr(value, dtype):
    if isinstance(value, Expr):
        return value
    if not np.isscalar(value):
        raise TypeError("Pass arrays as function arguments, not captured constants")
    dtype = np.asarray(value).dtype if isinstance(value, np.generic) else np.dtype(dtype)
    return Expr("constant", TensorType((), dtype), attr=value)


def exp(x):
    return x.unary("exp")


def tanh(x):
    return x.unary("tanh")


def sqrt(x):
    return x.unary("sqrt")


def gather(table, ids):
    if ids.op != "input":
        raise ValueError("Gather indices must be a function argument")
    if len(table.shape) != 2 or len(ids.shape) != 1 or ids.dtype != np.dtype("int32"):
        raise ValueError("Gather requires a matrix and an int32 index vector")
    return Expr("gather", TensorType((ids.shape[0], table.shape[1]), table.dtype), (table, ids))


def causal_mask(size, dtype):
    return Expr("mask", TensorType((size, size), dtype))
