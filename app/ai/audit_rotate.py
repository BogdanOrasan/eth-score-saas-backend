import os
from pathlib import Path

MAX_BYTES = int(os.getenv("AI_AUDIT_MAX_BYTES", str(10 * 1024 * 1024)))  # 10MB


def rotate_if_needed(path: Path) -> None:
    try:
        if not path.exists():
            return
        if path.stat().st_size <= MAX_BYTES:
            return
        rotated = path.with_suffix(path.suffix + ".1")
        if rotated.exists():
            rotated.unlink()
        path.rename(rotated)
    except Exception:
        return
