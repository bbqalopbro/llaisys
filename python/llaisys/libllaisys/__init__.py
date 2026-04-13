"""
libllaisys/__init__.py — C 动态库加载入口
===========================================
职责:
  1. 根据平台 (Linux/Win/Mac) 定位 libllaisys.so
  2. 用 ctypes.CDLL 加载动态库
  3. 依次调用 load_runtime / load_tensor / load_ops 注册所有 C 函数签名
  4. 导出 LIB_LLAISYS 全局对象供上层调用

加载顺序:  load_shared_library() → load_runtime() → load_tensor() → load_ops()
这样上层 models/qwen2.py 直接 import LIB_LLAISYS 就能调用所有 C API
"""
import os
import sys
import ctypes
from pathlib import Path

from .runtime import load_runtime
from .runtime import LlaisysRuntimeAPI
from .llaisys_types import llaisysDeviceType_t, DeviceType
from .llaisys_types import llaisysDataType_t, DataType
from .llaisys_types import llaisysMemcpyKind_t, MemcpyKind
from .llaisys_types import llaisysStream_t
from .tensor import llaisysTensor_t
from .tensor import load_tensor
from .ops import load_ops


def load_shared_library():
    """定位并加载 C 动态库, 返回 ctypes.CDLL 对象"""
    lib_dir = Path(__file__).parent

    # 跨平台动态库文件名
    if sys.platform.startswith("linux"):
        libname = "libllaisys.so"
    elif sys.platform == "win32":
        libname = "llaisys.dll"
    elif sys.platform == "darwin":
        libname = "llaisys.dylib"
    else:
        raise RuntimeError("Unsupported platform")

    lib_path = os.path.join(lib_dir, libname)

    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"Shared library not found: {lib_path}")

    return ctypes.CDLL(str(lib_path))


# 模块加载时立即执行: 加载 .so + 注册所有 C 函数签名
LIB_LLAISYS = load_shared_library()
load_runtime(LIB_LLAISYS)   # 注册 runtime API (内存/流操作)
load_tensor(LIB_LLAISYS)    # 注册 tensor API (创建/切片/数据搬运)
load_ops(LIB_LLAISYS)       # 注册 ops API (算子)


__all__ = [
    "LIB_LLAISYS",
    "LlaisysRuntimeAPI",
    "llaisysStream_t",
    "llaisysTensor_t",
    "llaisysDataType_t",
    "DataType",
    "llaisysDeviceType_t",
    "DeviceType",
    "llaisysMemcpyKind_t",
    "MemcpyKind",
    "llaisysStream_t",
]
