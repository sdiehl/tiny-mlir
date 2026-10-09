import ast
from typing import ClassVar
import math
import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class Expr:
    op: str
    shape: tuple[int, ...]
    args: tuple = ()
    value: object = None


def tensor_shape(node):
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "f32"
    ):
        raise TypeError("Expected an annotation such as f32[8]")
    dims = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    if not dims or any(
        not isinstance(d, ast.Constant) or type(d.value) is not int or d.value <= 0 for d in dims
    ):
        raise TypeError("Tensor dimensions must be positive integer literals")
    return tuple(d.value for d in dims)


class Frontend(ast.NodeVisitor):
    ops: ClassVar[dict] = {ast.Add: "addf", ast.Sub: "subf", ast.Mult: "mulf"}

    def __init__(self, env):
        self.env = dict(env)

    def visit_Name(self, node):
        if node.id not in self.env:
            raise TypeError(f"Unknown variable: {node.id}")
        return self.env[node.id]

    def visit_Constant(self, node):
        if type(node.value) not in (int, float):
            raise TypeError("Only numeric constants are supported")
        value = struct.unpack("f", struct.pack("f", float(node.value)))[0]
        if not math.isfinite(value):
            raise TypeError("Expected a finite f32 constant")
        return Expr("const", (), value=value)

    def visit_BinOp(self, node):
        if type(node.op) not in self.ops:
            raise TypeError(f"Unsupported operator: {type(node.op).__name__}")
        left, right = self.visit(node.left), self.visit(node.right)
        if left.shape and right.shape and left.shape != right.shape:
            raise TypeError(f"Shape mismatch: {left.shape} and {right.shape}")
        return Expr(self.ops[type(node.op)], left.shape or right.shape, (left, right))

    def visit_UnaryOp(self, node):
        if not isinstance(node.op, ast.USub):
            return self.generic_visit(node)
        value = self.visit(node.operand)
        return Expr("negf", value.shape, (value,))

    def generic_visit(self, node):
        raise TypeError(f"Unsupported syntax: {type(node).__name__}")


def specialize(source):
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise TypeError("Expected one function definition")
    fn = tree.body[0]
    a = fn.args
    if (
        fn.decorator_list
        or getattr(fn, "type_params", [])
        or a.posonlyargs
        or a.kwonlyargs
        or a.vararg
        or a.kwarg
        or a.defaults
        or not a.args
    ):
        raise TypeError("Expected plain positional arguments without defaults")
    names = [arg.arg for arg in a.args]
    if len(set(names)) != len(names):
        raise TypeError("Duplicate argument name")
    shapes = [tensor_shape(arg.annotation) for arg in a.args]
    shape = shapes[0]
    if any(s != shape for s in shapes) or tensor_shape(fn.returns) != shape:
        raise TypeError("All arguments and the result must have the same shape")
    inputs = [Expr("arg", shape, value=i) for i in range(len(names))]
    frontend = Frontend(dict(zip(names, inputs)))
    if not fn.body or not isinstance(fn.body[-1], ast.Return):
        raise TypeError("Expected a final return")
    for stmt in fn.body[:-1]:
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            raise TypeError("Only local assignments may precede the return")
        frontend.env[stmt.targets[0].id] = frontend.visit(stmt.value)
    result = frontend.visit(fn.body[-1].value)
    if result.shape != shape:
        raise TypeError("The return expression must have the declared shape")
    return inputs, result


def emit_mlir(inputs, result):
    shape = result.shape
    dims = "x".join(map(str, shape))
    tensor = f"tensor<{dims}xf32>"
    memref = f"memref<{dims}xf32>"
    indices = ", ".join(f"d{i}" for i in range(len(shape)))
    identity = f"affine_map<({indices}) -> ({indices})>"
    maps = ", ".join([identity] * (len(inputs) + 1))
    iters = ", ".join(['"parallel"'] * len(shape))
    arguments = ", ".join(f"%arg{i}: {tensor}" for i in range(len(inputs)))
    operands = ", ".join(f"%arg{i}" for i in range(len(inputs)))
    types = ", ".join([tensor] * len(inputs))
    scalars = ", ".join(f"%x{i}: f32" for i in range(len(inputs)))
    values = {expr: f"%x{i}" for i, expr in enumerate(inputs)}
    body = []

    def emit(expr):
        if expr in values:
            return values[expr]
        args = [emit(arg) for arg in expr.args]
        name = f"%v{len(body)}"
        if expr.op == "const":
            operation = f"arith.constant {expr.value:.9e} : f32"
        else:
            operation = f"arith.{expr.op} {', '.join(args)} : f32"
        body.append(f"      {name} = {operation}")
        values[expr] = name
        return name

    value = emit(result)
    code = "\n".join(body)
    return f"""module {{
  func.func @kernel({arguments}, %out: {memref}) attributes {{llvm.emit_c_interface}} {{
    %dest = bufferization.to_tensor %out restrict writable : {memref} to {tensor}
    %result = linalg.generic {{
      indexing_maps = [{maps}],
      iterator_types = [{iters}]
    }} ins({operands} : {types}) outs(%dest : {tensor}) {{
    ^bb0({scalars}, %unused: f32):
{code}
      linalg.yield {value} : f32
    }} -> {tensor}
    bufferization.materialize_in_destination %result in writable %out : ({tensor}, {memref}) -> ()
    return
  }}
}}"""


source = """
def scale_bias(x: f32[8], bias: f32[8]) -> f32[8]:
    scaled = x * 2.0
    return scaled + bias
"""

if __name__ == "__main__":
    print(emit_mlir(*specialize(source)))
