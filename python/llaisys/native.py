"""Optional pybind11 runtime boundary.

The legacy ctypes model loader remains available during migration. Scheduler
and batch-runtime calls should prefer this module when ``available`` is true.
"""

try:
    from . import _C  # type: ignore[attr-defined]
except ImportError:
    _C = None

available = _C is not None

if available:
    SamplingParams = _C.SamplingParams
    PrefillItem = _C.PrefillItem
    DecodeItem = _C.DecodeItem
    SchedulePlan = _C.SchedulePlan
    Qwen2BatchRuntime = _C.Qwen2BatchRuntime
else:
    SamplingParams = None
    PrefillItem = None
    DecodeItem = None
    SchedulePlan = None
    Qwen2BatchRuntime = None
