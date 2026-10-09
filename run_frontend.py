import ctypes as C
from pathlib import Path


class MemRef1D(C.Structure):
    _fields_ = [
        ("allocated", C.POINTER(C.c_float)),
        ("aligned", C.POINTER(C.c_float)),
        ("offset", C.c_int64),
        ("sizes", C.c_int64 * 1),
        ("strides", C.c_int64 * 1),
    ]


def descriptor(array):
    ptr = C.cast(array, C.POINTER(C.c_float))
    return MemRef1D(ptr, ptr, 0, (C.c_int64 * 1)(len(array)), (C.c_int64 * 1)(1))


lib = C.CDLL(str(Path("kernel.so").resolve()))
kernel = lib._mlir_ciface_kernel
kernel.argtypes = [C.POINTER(MemRef1D)] * 3
kernel.restype = None

x = (C.c_float * 8)(*range(8))
bias = (C.c_float * 8)(*[0.5] * 8)
out = (C.c_float * 8)()
x_desc, bias_desc, out_desc = map(descriptor, (x, bias, out))
kernel(C.byref(x_desc), C.byref(bias_desc), C.byref(out_desc))

expected = [2.0 * value + 0.5 for value in x]
assert list(out) == expected
print(list(out))
