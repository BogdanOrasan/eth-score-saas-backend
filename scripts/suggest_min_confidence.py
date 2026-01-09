import json
from pathlib import Path

AUDIT_LOG = Path("logs/ai_overlay.jsonl")
OUTCOMES = Path("logs/outcomes.jsonl")

# choose which horizon to use for suggestion
TARGET_HORIZON = "24h"  # change to "72h" when you have data

def load_jsonl(path: Path):
    items = []
    if not path.exists():
        return items
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            pass
    return items

def main():
    events = load_jsonl(AUDIT_LOG)
    if not events:
        print("No audit events in logs/ai_overlay.jsonl")
        return

    outcomes = [o for o in load_jsonl(OUTCOMES) if o.get("horizon") == TARGET_HORIZON]
    if not outcomes:
        print(f"No outcomes yet for horizon={TARGET_HORIZON}.")
        print("Run: python3 scripts/generate_outcomes_24h.py (or wait until enough time passes)")
        return

    out_by_ts = {o.get("engine_ts"): o for o in outcomes if o.get("engine_ts")}
    rows = []
    for e in events:
        eng = (e.get("engine") or {})
        ts = eng.get("ts")
        conf = eng.get("confidence")
        if ts is None or conf is None:
            continue
        o = out_by_ts.get(ts)
        if not o:
            continue
        ret = o.get("return_pct")
        if isinstance(ret, (int, float)) and isinstance(conf, (int, float)):
            rows.append((float(conf), float(ret)))

    if len(rows) < 10:
        print(f"Not enough matched samples yet: {len(rows)} (need >=10 for a stable suggestion).")
        return

    # Evaluate thresholds; objective: maximize high-bucket avg return with min size constraint
    thresholds = sorted(set(int(c) for c, _ in rows))
    best = None

    for T in thresholds:
        hi = [r for c, r in rows if c >= T]
        lo = [r for c, r in rows if c < T]
        if len(hi) < 10:
            continue
        hi_avg = sum(hi) / len(hi)
        hi_win = sum(1 for r in hi if r > 0) / len(hi)
        lo_avg = sum(lo) / len(lo) if lo else None

        score = hi_avg
        cand = {
            "suggested_min_confidence_ok": T,
            "horizon": TARGET_HORIZON,
            "matched_samples": len(rows),
            "high_bucket": {"n": len(hi), "avg_return_pct": round(hi_avg, 4), "win_rate": round(hi_win, 4)},
            "low_bucket_avg_return_pct": (round(lo_avg, 4) if lo_avg is not None else None),
        }
        if best is None or score > best["high_bucket"]["avg_return_pct"]:
            best = cand

    if not best:
        print("No threshold met minimum sample size (need >=10 in high bucket).")
        return

    Path("logs").mkdir(exist_ok=True)
    out = Path("logs/suggestions.json")
    out.write_text(json.dumps(best, indent=2), encoding="utf-8")

    print("Wrote suggestion to logs/suggestions.json")
    print(json.dumps(best, indent=2))

if __name__ == "__main__":
    main()
