"""A small MLIR compiler for transformer kernels."""

from .compiler import CPU, Numpy
from .kernels import Tensor, generate
from .model import Model

__all__ = ["CPU", "Model", "Numpy", "Tensor", "generate"]
