"""
ctypes bindings for LLAISYS Distributed Communication API (distributed.h).

Provides:
  - DistBackend enum
  - LlaisysDistConfig struct
  - comm_create / comm_destroy / comm_backend / comm_world_size / comm_rank
  - comm_all_reduce_sum_f32 / comm_barrier
  - comm_get_impl_ptr
"""

import ctypes
from enum import IntEnum
from . import LIB_LLAISYS


# ── Enums ──

class DistBackend(IntEnum):
    MOCK = 0
    NCCL = 1
    MPI  = 2


# ── C 结构体 ──

class LlaisysDistConfig(ctypes.Structure):
    _fields_ = [
        ("backend",      ctypes.c_int),
        ("world_size",   ctypes.c_int),
        ("rank",         ctypes.c_int),
        ("local_device", ctypes.c_int),
        # Phase 4: 多节点 NCCL 支持
        ("master_addr",  ctypes.c_char_p),   # rank 0 IP (NULL = 单机文件模式)
        ("master_port",  ctypes.c_int),      # rank 0 TCP 端口 (0 = 默认 29400)
    ]


# ── 函数签名配置 ──

def _setup_functions():
    lib = LIB_LLAISYS

    # Create
    if hasattr(lib, 'llaisysDistCommCreate'):
        lib.llaisysDistCommCreate.argtypes = [LlaisysDistConfig]
        lib.llaisysDistCommCreate.restype = ctypes.c_void_p

    # Destroy
    if hasattr(lib, 'llaisysDistCommDestroy'):
        lib.llaisysDistCommDestroy.argtypes = [ctypes.c_void_p]
        lib.llaisysDistCommDestroy.restype = None

    # Backend
    if hasattr(lib, 'llaisysDistCommBackend'):
        lib.llaisysDistCommBackend.argtypes = [ctypes.c_void_p]
        lib.llaisysDistCommBackend.restype = ctypes.c_int

    # WorldSize
    if hasattr(lib, 'llaisysDistCommWorldSize'):
        lib.llaisysDistCommWorldSize.argtypes = [ctypes.c_void_p]
        lib.llaisysDistCommWorldSize.restype = ctypes.c_int

    # Rank
    if hasattr(lib, 'llaisysDistCommRank'):
        lib.llaisysDistCommRank.argtypes = [ctypes.c_void_p]
        lib.llaisysDistCommRank.restype = ctypes.c_int

    # BackendAvailable
    if hasattr(lib, 'llaisysDistBackendAvailable'):
        lib.llaisysDistBackendAvailable.argtypes = [ctypes.c_int]
        lib.llaisysDistBackendAvailable.restype = ctypes.c_int

    # AllReduceSumF32
    if hasattr(lib, 'llaisysDistAllReduceSumF32'):
        lib.llaisysDistAllReduceSumF32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.llaisysDistAllReduceSumF32.restype = None

    # Barrier
    if hasattr(lib, 'llaisysDistBarrier'):
        lib.llaisysDistBarrier.argtypes = [ctypes.c_void_p]
        lib.llaisysDistBarrier.restype = None

    # GetImplPtr
    if hasattr(lib, 'llaisysDistCommGetImplPtr'):
        lib.llaisysDistCommGetImplPtr.argtypes = [ctypes.c_void_p]
        lib.llaisysDistCommGetImplPtr.restype = ctypes.c_void_p


_setup_functions()

# ── 导出便捷引用 ──

comm_create       = LIB_LLAISYS.llaisysDistCommCreate
comm_destroy      = LIB_LLAISYS.llaisysDistCommDestroy
comm_backend      = LIB_LLAISYS.llaisysDistCommBackend
comm_world_size   = LIB_LLAISYS.llaisysDistCommWorldSize
comm_rank         = LIB_LLAISYS.llaisysDistCommRank
backend_available = LIB_LLAISYS.llaisysDistBackendAvailable
all_reduce_sum_f32 = LIB_LLAISYS.llaisysDistAllReduceSumF32
barrier           = LIB_LLAISYS.llaisysDistBarrier
comm_get_impl_ptr = LIB_LLAISYS.llaisysDistCommGetImplPtr
