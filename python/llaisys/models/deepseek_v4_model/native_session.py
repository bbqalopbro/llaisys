"""Validated, explicit entry into the C++ model Session (no reference goldens).

Loading a bundle/module executes native code. Caller-supplied local artifacts
must be trusted; hashes detect accidental changes, not a malicious publisher.
"""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def _digest(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def load_native_session(source_model, checkpoint, bundle_manifest, module_path, *, slots=1, experimental_chunking=False):
    import torch
    from .config import InferenceConfig
    if type(slots) is not int or not 0 < slots <= 1024 or type(experimental_chunking) is not bool:
        raise ValueError("invalid native session slots/experimental flag")
    source, manifest, library = Path(source_model).resolve(), Path(bundle_manifest).resolve(), Path(module_path).resolve()
    bundle = json.loads(manifest.read_text())
    if (bundle.get("format") != "llaisys-v4-native-bundle-v1" or Path(bundle["source_model"]).resolve() != source
            or bundle["tilelang"] != "0.1.8" or bundle["tvm_ffi"] != "0.1.8.post2"
            or bundle["torch"] != torch.__version__ or bundle["cuda"] != torch.version.cuda
            or bundle["cxx11_abi"] is not True or not torch._C._GLIBCXX_USE_CXX11_ABI
            or not torch.cuda.is_available() or bundle["compute_capability"] != list(torch.cuda.get_device_capability(0))):
        raise ValueError("native bundle source/ABI/GPU mismatch")
    capacity = bundle["configuration"]["max_seq_len"]
    config = InferenceConfig.from_directory(source, max_seq_len=capacity)
    if json.loads(json.dumps(asdict(config))) != bundle["configuration"]: raise ValueError("native bundle model config changed")
    for relative in ("config.json", "inference/config.json", "inference/kernel.py"):
        path = source / relative
        if bundle["source_sha256"].get(str(path)) != _digest(path): raise ValueError("native bundle source changed")
    entries = {}
    for key, value in bundle["kernels"].items():
        path = (manifest.parent / value["library"]).resolve()
        if path.parent != manifest.parent or path.suffix != ".so" or _digest(path) != value["sha256"]:
            raise ValueError("invalid/changed native kernel artifact")
        entries[key] = (value["operation"], str(path))
    if "_v4_native" in sys.modules:
        native = sys.modules["_v4_native"]
        if Path(native.__file__).resolve() != library: raise RuntimeError("another native V4 module is already loaded")
    else:
        spec = importlib.util.spec_from_file_location("_v4_native", library)
        native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native); sys.modules["_v4_native"] = native
    return native.Session(str(source), str(Path(checkpoint).resolve()), capacity, slots, entries,
                          bundle["tilelang"], bundle["hadamard_version"], experimental_chunking)
