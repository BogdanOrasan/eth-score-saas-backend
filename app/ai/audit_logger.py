import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.ai.audit_rotate import rotate_if_needed


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _last_jsonl_record(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        lines = path.read_text(encoding="utf-8").splitlines()
        for ln in reversed(lines):
            ln = ln.strip()
            if ln:
                return json.loads(ln)
        return {}
    except Exception:
        return {}


def log_ai_overlay(event: Dict[str, Any]) -> None:
    """
    Appends one JSON object per line to logs/ai_overlay.jsonl
    - rotates when file exceeds AI_AUDIT_MAX_BYTES (default 10MB)
    - dedup: skips if last record has same (route + engine.ts)
    Safe: never throws; on any error it silently skips logging.
    """
    try:
        log_dir = Path(os.getenv("AI_AUDIT_DIR", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "ai_overlay.jsonl"

        rotate_if_needed(path)

        # dedup key
        route = event.get("route")
        engine_ts = (event.get("engine") or {}).get("ts")
        key = f"{route}|{engine_ts}"

        last = _last_jsonl_record(path)
        last_route = last.get("route")
        last_engine_ts = (last.get("engine") or {}).get("ts")
        last_key = f"{last_route}|{last_engine_ts}"

        if key == last_key:
            return  # skip duplicate

        record = {"ts": _ts(), **event}

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        return
