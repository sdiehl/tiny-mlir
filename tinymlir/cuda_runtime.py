"""Minimal synchronous-use CUDA Driver API runtime. Requires Linux + NVIDIA.

Launches use the default stream. Buffers remain on the device between kernels;
only explicit numpy() calls copy data back. Device execution is a separate test
from cross-compiling PTX.
"""

import ctypes as C
import math
import sys

import numpy as np

from .compiler import Compiler, spec
from .kernels import generate


class DeviceArray:
    def __init__(self, owner, shape, dtype):
        self.owner, self.shape, self.dtype = owner, tuple(shape), np.dtype(dtype)
        self.nbytes = math.prod(self.shape) * self.dtype.itemsize
        self.ptr = owner.allocate(self.nbytes)

    def __del__(self):
        try:
            self.owner.release(self.ptr)
        except Exception:  # noqa: BLE001, S110
            # Explicit close() reports errors; destructors must not mask them.
            pass


class CUDA:
    target = "cuda"

    def __init__(self, cache=".mlir-cache", chip="sm_75", libdevice=None):
        if sys.platform != "linux":
            raise RuntimeError("CUDA execution requires Linux with an NVIDIA driver")
        self.compiler = Compiler(cache, chip, libdevice)
        self.driver = C.CDLL("libcuda.so.1")
        ptr, u64, uint, size = C.c_void_p, C.c_uint64, C.c_uint, C.c_size_t
        signatures = {
            "cuInit": [uint],
            "cuDeviceGet": [C.POINTER(C.c_int), C.c_int],
            "cuDeviceGetAttribute": [C.POINTER(C.c_int), C.c_int, C.c_int],
            "cuCtxCreate_v2": [C.POINTER(ptr), uint, C.c_int],
            "cuCtxDestroy_v2": [ptr],
            "cuCtxSynchronize": [],
            "cuMemAlloc_v2": [C.POINTER(u64), size],
            "cuMemFree_v2": [u64],
            "cuMemcpyHtoD_v2": [u64, ptr, size],
            "cuMemcpyDtoH_v2": [ptr, u64, size],
            "cuModuleLoadData": [C.POINTER(ptr), ptr],
            "cuModuleGetFunction": [C.POINTER(ptr), ptr, C.c_char_p],
            "cuModuleUnload": [ptr],
            "cuLaunchKernel": [
                ptr,
                uint,
                uint,
                uint,
                uint,
                uint,
                uint,
                uint,
                ptr,
                C.POINTER(ptr),
                C.POINTER(ptr),
            ],
        }
        for name, args in signatures.items():
            fn = getattr(self.driver, name)
            fn.argtypes, fn.restype = args, C.c_int
        self.functions, self.allocations, self.retired = {}, set(), set()
        self.context = ptr()
        self.call("cuInit", 0)
        device = C.c_int()
        self.call("cuDeviceGet", C.byref(device), 0)
        major, minor = C.c_int(), C.c_int()
        self.call("cuDeviceGetAttribute", C.byref(major), 75, device)
        self.call("cuDeviceGetAttribute", C.byref(minor), 76, device)
        requested = chip.removeprefix("sm_")
        if not requested.isdigit() or int(requested) > major.value * 10 + minor.value:
            raise ValueError(f"Target {chip} is not supported by this GPU")
        self.call("cuCtxCreate_v2", C.byref(self.context), 0, device)

    def call(self, name, *args):
        error = getattr(self.driver, name)(*args)
        if error:
            raise RuntimeError(f"{name} failed with CUDA error {error}")

    def allocate(self, nbytes):
        if not self.context or nbytes <= 0:
            raise ValueError("Expected a live context and a nonempty allocation")
        pointer = C.c_uint64()
        self.call("cuMemAlloc_v2", C.byref(pointer), nbytes)
        self.allocations.add(pointer.value)
        return pointer.value

    def release(self, pointer):
        if pointer in self.allocations:
            self.allocations.remove(pointer)
            self.retired.add(pointer)

    def array(self, value):
        if isinstance(value, DeviceArray):
            if value.owner is not self or value.ptr not in self.allocations:
                raise ValueError("Array belongs to another or closed CUDA context")
            return value
        host = np.ascontiguousarray(value)
        spec(host)
        result = DeviceArray(self, host.shape, host.dtype)
        self.call("cuMemcpyHtoD_v2", result.ptr, host.ctypes.data, host.nbytes)
        return result

    def numpy(self, value):
        self.array(value)  # Validate ownership.
        self.sync()
        host = np.empty(value.shape, dtype=value.dtype)
        self.call("cuMemcpyDtoH_v2", host.ctypes.data, value.ptr, host.nbytes)
        return host

    def sync(self):
        self.call("cuCtxSynchronize")
        for pointer in list(self.retired):
            self.call("cuMemFree_v2", pointer)
            self.retired.remove(pointer)

    def __call__(self, op, *args, **options):
        args = tuple(self.array(a) for a in args)
        types = tuple(spec(a) for a in args)
        key = (op, types, tuple(sorted(options.items())))
        if key not in self.functions:
            source, output, work = generate(op, types, "cuda", **options)
            path = self.compiler.build(source, "cuda")
            module, fn = C.c_void_p(), C.c_void_p()
            image = C.create_string_buffer(path.read_bytes())
            self.call("cuModuleLoadData", C.byref(module), C.cast(image, C.c_void_p))
            try:
                self.call("cuModuleGetFunction", C.byref(fn), module, b"kernel")
            except Exception:  # noqa: BLE001, S110
                self.call("cuModuleUnload", module)
                raise
            self.functions[key] = (module, fn, output, work)
        module, fn, output, work = self.functions[key]
        result = DeviceArray(self, output.shape, np.float32)
        # kernelParams points to host storage for each device pointer argument.
        values = [C.c_uint64(a.ptr) for a in (*args, result)]
        params = (C.c_void_p * len(values))(*(C.addressof(v) for v in values))
        block = 128
        self.call(
            "cuLaunchKernel",
            fn,
            (work + block - 1) // block,
            1,
            1,
            block,
            1,
            1,
            0,
            None,
            params,
            None,
        )
        return result

    def close(self):
        if not self.context:
            return
        for pointer in list(self.allocations):
            self.release(pointer)
        self.sync()
        for module, *_ in self.functions.values():
            self.call("cuModuleUnload", module)
        self.functions.clear()
        self.call("cuCtxDestroy_v2", self.context)
        self.context = C.c_void_p()
