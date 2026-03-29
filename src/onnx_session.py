"""ONNX Runtime session creation and noise management.

ONNX Runtime's C++ backend prints noisy messages to stderr during session
creation on ARM (Raspberry Pi). This module provides a file-descriptor-level
redirect to suppress them. Used by detection, embedding, and quantization.

Origin: Valenia runtime_utils.py — suppress_stderr_fd.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator


@contextlib.contextmanager
def suppress_stderr_fd(enabled: bool = True) -> Iterator[None]:
    """Suppress C/C++ stderr output by redirecting file descriptor 2 to /dev/null."""
    if not enabled:
        yield
        return

    saved_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
