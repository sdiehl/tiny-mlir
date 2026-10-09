"""Compile the same kernel body for a CPU or the NVIDIA device ABI."""

import ctypes as C
import hashlib
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from .kernels import Tensor, generate


def command(args):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed: {' '.join(map(str, args))}\n{result.stderr}")
    return result.stdout


class Compiler:
    def __init__(self, cache=".mlir-cache", chip="sm_75", libdevice=None):
        self.cache = Path(cache).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.chip = chip
        if not re.fullmatch(r"sm_[0-9]+a?", chip):
            raise ValueError("Invalid NVIDIA target")
        self.libdevice = Path(
            libdevice
            or os.environ.get("LIBDEVICE", "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc")
        )
        self.version = command(["mlir-opt", "--version"])
        self.version += command(["mlir-translate", "--version"])
        self.version += command(["clang", "--version"])
        self.version += command(["llc", "--version"])
        self.device_hash = None

    def build(self, source, target):
        device_hash = ""
        if target == "cuda":
            if not self.libdevice.is_file():
                raise FileNotFoundError(
                    "CUDA libdevice is required; set LIBDEVICE or pass --libdevice"
                )
            if self.device_hash is None:
                self.device_hash = hashlib.sha256(self.libdevice.read_bytes()).hexdigest()
            device_hash = self.device_hash
        # Include compiler source so a change to the lowering pipeline invalidates the cache.
        fingerprint = (
            source + target + self.chip + self.version + device_hash + Path(__file__).read_text()
        )
        key = hashlib.sha256(fingerprint.encode()).hexdigest()[:24]
        directory = self.cache / key
        directory.mkdir(exist_ok=True)
        result = directory / ("kernel.so" if target == "cpu" else "kernel.ptx")
        if result.exists():
            return result
        src = directory / "kernel.mlir"
        lowered = directory / "kernel-llvm.mlir"
        llvm_ir = directory / "kernel.ll"
        src.write_text(source)
        if target == "cpu":
            command(
                [
                    "mlir-opt",
                    src,
                    "--canonicalize",
                    "--convert-scf-to-cf",
                    "--convert-math-to-libm",
                    "--convert-math-to-llvm",
                    "--convert-arith-to-llvm",
                    "--convert-func-to-llvm",
                    "--convert-cf-to-llvm",
                    "--finalize-memref-to-llvm",
                    "--reconcile-unrealized-casts",
                    "-o",
                    lowered,
                ]
            )
            command(["mlir-translate", "--mlir-to-llvmir", lowered, "-o", llvm_ir])
            command(["clang", "-O3", "-shared", "-fPIC", llvm_ir, "-lm", "-o", result])
        else:
            command(
                [
                    "mlir-opt",
                    src,
                    "--pass-pipeline=builtin.module(gpu.module(canonicalize,convert-scf-to-cf,convert-gpu-to-nvvm{use-bare-ptr-memref-call-conv index-bitwidth=64},reconcile-unrealized-casts))",
                    "-o",
                    lowered,
                ]
            )
            # The generator emits exactly one gpu.module, without module attributes.
            # Unwrap that known container; this is not a general-purpose MLIR parser.
            lines = lowered.read_text().strip().splitlines()
            if (
                lines[0] != "module {"
                or lines[1].strip() != "gpu.module @device {"
                or [x.strip() for x in lines[-2:]] != ["}", "}"]
            ):
                raise RuntimeError("Unexpected GPU module structure")
            device = directory / "device.mlir"
            device.write_text("module {\n" + "\n".join(lines[2:-2]) + "\n}\n")
            command(["mlir-translate", "--mlir-to-llvmir", device, "-o", llvm_ir])
            linked, optimized = directory / "linked.bc", directory / "optimized.bc"
            command(["llvm-link", llvm_ir, self.libdevice, "-o", linked])
            command(
                [
                    "opt",
                    "-passes=internalize,globaldce,default<O3>",
                    "-internalize-public-api-list=kernel",
                    linked,
                    "-o",
                    optimized,
                ]
            )
            command(
                [
                    "llc",
                    "-march=nvptx64",
                    f"-mcpu={self.chip}",
                    "-mattr=+ptx75",
                    "-O3",
                    optimized,
                    "-o",
                    result,
                ]
            )
            if re.search(r"\.extern\s+\.func", result.read_text()):
                result.unlink()
                raise RuntimeError("Unresolved device function in PTX")
        return result


class MemRef(C.Structure):
    _fields_ = [
        ("allocated", C.c_void_p),
        ("aligned", C.c_void_p),
        ("offset", C.c_int64),
        ("sizes", C.c_int64 * 1),
        ("strides", C.c_int64 * 1),
    ]


def descriptor(array):
    return MemRef(
        array.ctypes.data,
        array.ctypes.data,
        0,
        (C.c_int64 * 1)(array.size),
        (C.c_int64 * 1)(1),
    )


def spec(array):
    dtype = {np.dtype("float32"): "f32", np.dtype("int32"): "i32"}.get(array.dtype)
    if dtype is None:
        raise TypeError(f"Unsupported dtype: {array.dtype}; use float32 or int32")
    return Tensor(tuple(array.shape), dtype)


class CPU:
    target = "cpu"

    def __init__(self, cache=".mlir-cache", **kwargs):
        self.compiler = Compiler(cache, **kwargs)
        self.functions = {}

    def array(self, value):
        array = np.asarray(value)
        spec(array)
        return np.ascontiguousarray(array)

    def numpy(self, value):
        return value

    def sync(self):
        pass

    def close(self):
        pass

    def __call__(self, op, *args, **options):
        args = tuple(self.array(a) for a in args)
        types = tuple(spec(a) for a in args)
        key = (op, types, tuple(sorted(options.items())))
        if key not in self.functions:
            module, output, _work = generate(op, types, self.target, **options)
            path = self.compiler.build(module, self.target)
            library = C.CDLL(str(path))
            fn = library._mlir_ciface_kernel
            fn.argtypes = [C.POINTER(MemRef)] * (len(args) + 1)
            fn.restype = None
            self.functions[key] = (library, fn, output)
        library, fn, output = self.functions[key]
        result = np.empty(output.shape, dtype=np.float32)
        descriptors = [descriptor(a) for a in (*args, result)]
        fn(*(C.byref(d) for d in descriptors))
        return result


class Numpy:
    target = "numpy"

    @staticmethod
    def array(value):
        return np.ascontiguousarray(value)

    @staticmethod
    def numpy(value):
        return value

    def sync(self):
        pass

    def close(self):
        pass

    def __call__(self, op, *a, **options):
        # Independent, vectorized reference implementations.
        x = a[0]
        if op == "gelu":
            return (
                np.float32(0.5)
                * x
                * (
                    np.float32(1)
                    + np.tanh(
                        np.float32(0.7978845608028654) * (x + np.float32(0.044715) * x * x * x)
                    )
                )
            )
        if op == "add":
            return x + a[1]
        if op == "softmax":
            e = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e / np.sum(e, axis=-1, keepdims=True)
        if op == "norm":
            return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(
                x.var(axis=-1, keepdims=True) + np.float32(1e-5)
            ) * a[1] + a[2]
        if op in ("linear", "linear_gelu"):
            y = x @ a[1] + a[2]
            return self("gelu", y) if op == "linear_gelu" else y
        if op == "matmul":
            return x @ (a[1].T if options.get("transpose") else a[1])
        if op == "logits":
            return x[-1:] @ a[1].T
        if op == "embedding":
            return a[1][x] + a[2][np.arange(len(x))]
        heads = options["heads"]
        tokens, packed = x.shape
        channels = packed // 3
        q, k, v = [
            part.reshape(tokens, heads, channels // heads).transpose(1, 0, 2)
            for part in np.split(x, 3, axis=-1)
        ]
        if op == "scores":
            scores = (q @ k.transpose(0, 2, 1)) * np.float32((channels // heads) ** -0.5)
            scores = np.where(np.tri(tokens, dtype=bool), scores, -np.inf)
            return scores.reshape(heads * tokens, tokens)
        if op == "context":
            p = a[1].reshape(heads, tokens, tokens)
            return (p @ v).transpose(1, 0, 2).reshape(tokens, channels)
        raise ValueError(op)
