import json
from pathlib import Path
from datetime import datetime, timezone

AUDIT = Path("logs/ai_overlay.jsonl")
OUTCOMES = Path("logs/outcomes.jsonl")
SUGGEST = Path("logs/suggestions.json")
OUT = Path("logs/recommendation.json")

HORIZONS = ["1h", "24h", "72h"]

def load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out

def last_jsonl(path: Path):
    xs = load_jsonl(path)
    return xs[-1] if xs else None

def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def main():
    last = last_jsonl(AUDIT)
    if not last:
        print("No audit events yet.")
        return

    eng = last.get("engine", {}) or {}
    ov = last.get("overlay", {}) or {}
    route = last.get("route", None)
    engine_ts = eng.get("ts")

    outcomes = load_jsonl(OUTCOMES)
    out_by_h = {}
    for o in outcomes:
        if o.get("engine_ts") == engine_ts and o.get("horizon") in HORIZONS:
            out_by_h[o.get("horizon")] = o

    suggestion = load_json(SUGGEST)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_route": route,
        "engine": eng,
        "overlay": ov,
        "outcomes": {h: out_by_h.get(h) for h in HORIZONS},
        "suggestion": suggestion,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
