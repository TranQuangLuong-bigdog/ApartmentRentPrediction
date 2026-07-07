"""File lock helpers for protecting `registry.json`.

Since the project requirements don't include external lock dependencies,
this module implements a simple lock using a lock file.

Lock strategy:
- Acquire: create a lock file next to registry.json using O_EXCL.
- Release: delete lock file.
- Wait until timeout.

This is sufficient for typical single-machine usage.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def registry_file_lock(registry_path: Path, *, timeout_seconds: float = 30.0, poll_interval: float = 0.1) -> Iterator[None]:
    """Acquire a lock for the given registry file.

    Args:
        registry_path: Path to registry.json.
        timeout_seconds: Max wait time.
        poll_interval: Sleep interval between attempts.

    Yields:
        None while lock is held.

    Raises:
        TimeoutError: if cannot acquire lock within timeout.
    """

    lock_path = registry_path.with_suffix(".lock")
    start = time.time()

    while True:
        try:
            # O_EXCL ensures the create fails if file exists.
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start > timeout_seconds:
                raise TimeoutError(f"Cannot acquire registry lock: {lock_path}")
            time.sleep(poll_interval)

    try:
        yield
    finally:
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            # Best-effort release.
            pass

