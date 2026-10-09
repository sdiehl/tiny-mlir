"""Shape-specialized MLIR kernels for Parts 10–13. LLVM/MLIR 22.1.8.

Each work item owns a disjoint output element or row. The CPU backend loops
through work items; the CUDA backend maps them to individual GPU threads.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]
    dtype: str = "f32"

    @property
    def size(self):
        return prod(self.shape)

    @property
    def mlir(self):
        return f"memref<{self.size}x{self.dtype}>"


class Builder:
    def __init__(self, specs):
        self.specs = specs
        self.header, self.lines = [], []
        self.constants = {}
        self.serial = 0
        self.depth = 0

    def name(self):
        self.serial += 1
        return f"%v{self.serial}"

    def line(self, text):
        self.lines.append("  " * self.depth + text)

    def value(self, operation):
        name = self.name()
        self.line(f"{name} = {operation}")
        return name

    def constant(self, value, typ="index"):
        key = (value, typ)
        if key not in self.constants:
            name = self.name()
            if typ == "f32":
                literal = "0xFF800000" if value == float("-inf") else f"{value:.9e}"
            else:
                literal = str(value)
            self.header.append(f"{name} = arith.constant {literal} : {typ}")
            self.constants[key] = name
        return self.constants[key]

    def binary(self, op, a, b, typ="f32"):
        return self.value(f"arith.{op} {a}, {b} : {typ}")

    def index(self, op, a, b):
        a = self.constant(a) if isinstance(a, int) else a
        b = self.constant(b) if isinstance(b, int) else b
        return self.binary(op, a, b, "index")

    def at(self, row, width, col):
        return self.index("addi", self.index("muli", row, width), col)

    def load(self, arg, index):
        return self.value(f"memref.load %a{arg}[{index}] : {self.specs[arg].mlir}")

    def store(self, value, index):
        arg = len(self.specs) - 1
        self.line(f"memref.store {value}, %a{arg}[{index}] : {self.specs[arg].mlir}")

    def math(self, op, value):
        return self.value(f"math.{op} {value} : f32")

    @contextmanager
    def loop(self, length):
        i = self.name()
        self.line(
            f"scf.for {i} = {self.constant(0)} to {self.constant(length)} step {self.constant(1)} {{"
        )
        self.depth += 1
        yield i
        self.depth -= 1
        self.line("}")

    def reduce(self, length, initial, step):
        result, i, acc = self.name(), self.name(), self.name()
        self.line(
            f"{result} = scf.for {i} = {self.constant(0)} to {self.constant(length)} step {self.constant(1)} iter_args({acc} = {initial}) -> (f32) {{"
        )
        self.depth += 1
        value = step(i, acc)
        self.line(f"scf.yield {value} : f32")
        self.depth -= 1
        self.line("}")
        return result

    def choose(self, condition, then, otherwise):
        result = self.name()
        self.line(f"{result} = scf.if {condition} -> (f32) {{")
        self.depth += 1
        value = then()
        self.line(f"scf.yield {value} : f32")
        self.depth -= 1
        self.line("} else {")
        self.depth += 1
        value = otherwise()
        self.line(f"scf.yield {value} : f32")
        self.depth -= 1
        self.line("}")
        return result


def gelu_scalar(b, x):
    # GPT-2's tanh approximation, with the cubic term retained.
    square = b.binary("mulf", x, x)
    cube = b.binary("mulf", square, x)
    cubic = b.binary("mulf", b.constant(0.044715, "f32"), cube)
    inner = b.binary("addf", x, cubic)
    inner = b.binary("mulf", b.constant(0.7978845608028654, "f32"), inner)
    t = b.math("tanh", inner)
    plus_one = b.binary("addf", b.constant(1.0, "f32"), t)
    half_x = b.binary("mulf", b.constant(0.5, "f32"), x)
    return b.binary("mulf", half_x, plus_one)


def generate(op, inputs, target="cpu", **options):
    """Return (module text, output type, number of independent work items)."""
    if target not in ("cpu", "cuda"):
        raise ValueError(target)
    inputs = tuple(inputs)
    if not inputs or any(not t.shape or any(d <= 0 for d in t.shape) for t in inputs):
        raise ValueError("Expected nonempty tensors with positive static shapes")
    if any(t.dtype != "f32" for t in inputs[(1 if op == "embedding" else 0) :]):
        raise TypeError("Numerical inputs must be f32")
    x = inputs[0].shape
    if op in ("gelu", "add"):
        if len(inputs) != (2 if op == "add" else 1) or any(t.shape != x for t in inputs):
            raise ValueError("Elementwise operands must have the same shape")
        output, work = Tensor(x), prod(x)
    elif op in ("softmax", "norm"):
        if len(x) != 2 or len(inputs) != (3 if op == "norm" else 1):
            raise ValueError("Expected a matrix and, for norm, gain and bias")
        rows, cols = x
        if op == "norm" and any(t.shape != (cols,) for t in inputs[1:]):
            raise ValueError("Normalization parameters must match the last axis")
        output, work = Tensor(x), rows
    elif op in ("matmul", "linear", "linear_gelu", "logits"):
        if len(x) != 2 or len(inputs) != (3 if op.startswith("linear") else 2):
            raise ValueError("Expected two matrices and an optional bias")
        m, k = x
        if len(inputs[1].shape) != 2:
            raise ValueError("Expected a matrix weight")
        transpose = op == "logits" or options.get("transpose", False)
        if transpose:
            n, wk = inputs[1].shape
        else:
            wk, n = inputs[1].shape
        if k != wk or op.startswith("linear") and inputs[2].shape != (n,):
            raise ValueError("Incompatible matrix or bias shape")
        output = Tensor((1 if op == "logits" else m, n))
        work = output.size
    elif op in ("scores", "context"):
        heads = options["heads"]
        if not isinstance(heads, int) or heads <= 0 or len(x) != 2 or x[1] % (3 * heads):
            raise ValueError("Packed QKV width must be divisible by 3 * heads")
        tokens, packed = x
        channels, width = packed // 3, packed // (3 * heads)
        if op == "scores":
            if len(inputs) != 1:
                raise ValueError("Scores expects packed QKV")
            output = Tensor((heads * tokens, tokens))
        else:
            if len(inputs) != 2 or inputs[1].shape != (heads * tokens, tokens):
                raise ValueError("Invalid attention probability shape")
            output = Tensor((tokens, channels))
        work = output.size
    elif op == "embedding":
        if len(inputs) != 3 or inputs[0].dtype != "i32" or len(x) != 1:
            raise ValueError("Embedding expects i32 token IDs and two weight matrices")
        tokens = x[0]
        if any(len(t.shape) != 2 for t in inputs[1:]):
            raise ValueError("Embedding weights must be matrices")
        channels = inputs[1].shape[1]
        if inputs[2].shape[1] != channels or tokens > inputs[2].shape[0]:
            raise ValueError("Invalid position embedding shape")
        output, work = Tensor((tokens, channels)), tokens * channels
    else:
        raise ValueError(f"Unknown operation: {op}")

    b = Builder((*inputs, output))
    zero = b.constant(0.0, "f32")
    if target == "cpu":
        b.line(f"scf.for %work = {b.constant(0)} to {b.constant(work)} step {b.constant(1)} {{")
    else:
        block = b.value("gpu.block_id x")
        block_size = b.value("gpu.block_dim x")
        thread = b.value("gpu.thread_id x")
        base = b.index("muli", block, block_size)
        b.line(f"%work = arith.addi {base}, {thread} : index")
        valid = b.value(f"arith.cmpi ult, %work, {b.constant(work)} : index")
        b.line(f"scf.if {valid} {{")
    b.depth += 1
    i = "%work"
    if op in ("gelu", "add"):
        value = b.load(0, i)
        result = gelu_scalar(b, value) if op == "gelu" else b.binary("addf", value, b.load(1, i))
        b.store(result, i)
    elif op == "softmax":
        maximum = b.reduce(
            cols,
            b.constant(float("-inf"), "f32"),
            lambda j, acc: b.binary("maximumf", acc, b.load(0, b.at(i, cols, j))),
        )

        def sum_exp(j, acc):
            offset = b.at(i, cols, j)
            shifted = b.binary("subf", b.load(0, offset), maximum)
            e = b.math("exp", shifted)
            b.store(e, offset)
            return b.binary("addf", acc, e)

        total = b.reduce(cols, zero, sum_exp)
        with b.loop(cols) as j:
            offset = b.at(i, cols, j)
            b.store(b.binary("divf", b.load(len(inputs), offset), total), offset)
    elif op == "norm":
        count = b.constant(float(cols), "f32")
        total = b.reduce(
            cols,
            zero,
            lambda j, acc: b.binary("addf", acc, b.load(0, b.at(i, cols, j))),
        )
        mean = b.binary("divf", total, count)

        def variance_step(j, acc):
            centered = b.binary("subf", b.load(0, b.at(i, cols, j)), mean)
            return b.binary("addf", acc, b.binary("mulf", centered, centered))

        squared = b.reduce(cols, zero, variance_step)
        variance = b.binary("divf", squared, count)
        denom = b.math("sqrt", b.binary("addf", variance, b.constant(1e-5, "f32")))
        with b.loop(cols) as j:
            offset = b.at(i, cols, j)
            centered = b.binary("subf", b.load(0, offset), mean)
            scaled = b.binary("mulf", b.binary("divf", centered, denom), b.load(1, j))
            b.store(b.binary("addf", scaled, b.load(2, j)), offset)
    elif op in ("matmul", "linear", "linear_gelu", "logits"):
        row = b.constant(m - 1) if op == "logits" else b.index("divui", i, n)
        col = b.index("remui", i, n)

        def dot(j, acc):
            left = b.load(0, b.at(row, k, j))
            right_offset = b.at(col, k, j) if transpose else b.at(j, n, col)
            return b.binary("addf", acc, b.binary("mulf", left, b.load(1, right_offset)))

        result = b.reduce(k, zero, dot)
        if op.startswith("linear"):
            result = b.binary("addf", result, b.load(2, col))
        if op == "linear_gelu":
            result = gelu_scalar(b, result)
        b.store(result, i)
    elif op == "scores":
        col = b.index("remui", i, tokens)
        row_head = b.index("divui", i, tokens)
        row = b.index("remui", row_head, tokens)
        head = b.index("divui", row_head, tokens)
        head_offset = b.index("muli", head, width)
        future = b.value(f"arith.cmpi ugt, {col}, {row} : index")

        def score():
            def dot(j, acc):
                dim = b.index("addi", head_offset, j)
                q = b.load(0, b.at(row, packed, dim))
                key_dim = b.index("addi", channels, dim)
                key = b.load(0, b.at(col, packed, key_dim))
                return b.binary("addf", acc, b.binary("mulf", q, key))

            value = b.reduce(width, zero, dot)
            return b.binary("mulf", value, b.constant(width**-0.5, "f32"))

        b.store(b.choose(future, lambda: b.constant(float("-inf"), "f32"), score), i)
    elif op == "context":
        row, dim = b.index("divui", i, channels), b.index("remui", i, channels)
        head = b.index("divui", dim, width)
        probability_row = b.at(head, tokens, row)

        def weighted_value(j, acc):
            p = b.load(1, b.at(probability_row, tokens, j))
            v = b.load(0, b.at(j, packed, b.index("addi", 2 * channels, dim)))
            return b.binary("addf", acc, b.binary("mulf", p, v))

        b.store(b.reduce(tokens, zero, weighted_value), i)
    elif op == "embedding":
        row, col = b.index("divui", i, channels), b.index("remui", i, channels)
        token = b.value(f"arith.index_cast {b.load(0, row)} : i32 to index")
        word = b.load(1, b.at(token, channels, col))
        position = b.load(2, b.at(row, channels, col))
        b.store(b.binary("addf", word, position), i)
    b.depth -= 1
    b.line("}")
    body = "\n".join("    " + line for line in b.header + b.lines)
    arguments = ", ".join(f"%a{i}: {t.mlir}" for i, t in enumerate(b.specs))
    if target == "cpu":
        module = f"module {{\n  func.func @kernel({arguments}) attributes {{llvm.emit_c_interface}} {{\n{body}\n    return\n  }}\n}}\n"
    else:
        module = f"module {{\n  gpu.module @device {{\n    gpu.func @kernel({arguments}) kernel {{\n{body}\n      gpu.return\n    }}\n  }}\n}}\n"
    return module, output, work
