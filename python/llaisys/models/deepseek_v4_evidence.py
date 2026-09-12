"""Bounded-memory checkpoint identity for reproducible inference comparisons."""

import hashlib
import os
from pathlib import Path


def _signature(stat):
    return {name: getattr(stat, name) for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


def checkpoint_identity(path, *, chunk_bytes=8 * 1024 * 1024):
    """Hash every byte, including the payload, without loading weights in RAM.

    This intentionally has no mtime-based digest cache. Stat checks detect
    ordinary concurrent replacement/writes; they are not protection against an
    adversary able to rewrite both the file and filesystem metadata.
    """
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("hash chunk size must be a positive integer")
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = _signature(os.fstat(handle.fileno()))
        while data := handle.read(chunk_bytes):
            digest.update(data)
        after = _signature(os.fstat(handle.fileno()))
    if before != after or after != _signature(path.stat()):
        raise ValueError("checkpoint changed while computing payload identity")
    return {"path": str(path), "size_bytes": before["st_size"], "sha256": digest.hexdigest(),
            "scope": "entire_file_including_header_and_tensor_payload", "signature": before}


def verify_checkpoint_unchanged(path, identity):
    if _signature(Path(path).stat()) != identity["signature"]:
        raise ValueError("checkpoint changed after payload identity was computed")


def require_same_checkpoint(actual, expected):
    for identity in (actual, expected):
        digest = identity.get("sha256")
        if (type(identity.get("size_bytes")) is not int or identity["size_bytes"] < 0
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("invalid checkpoint payload identity")
    if (actual.get("scope") != "entire_file_including_header_and_tensor_payload"
            or expected.get("scope") != actual["scope"]
            or actual.get("size_bytes") != expected.get("size_bytes")
            or actual.get("sha256") != expected.get("sha256")):
        raise ValueError("baseline checkpoint payload identity differs from this run")
