import json
from pathlib import Path

AUDIT_LOG = Path("logs/ai_overlay.jsonl")
OUTCOMES = Path("logs/outcomes.jsonl")
# outcomes.jsonl format (one JSON per line):
# {"engine_ts":"2026-01-08T17:35:03.399437+00:00","horizon":"72h","return_pct":1.25}
# return_pct = (price_after - price_at_signal)/price_at_signal * 100

def load_jsonl(path: Path):
    items = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items

def main():
    events = load_jsonl(AUDIT_LOG)
    outcomes = load_jsonl(OUTCOMES)

    if not events:
        print("No audit events found in logs/ai_overlay.jsonl")
        return

    if not OUTCOMES.exists():
        print("Missing logs/outcomes.jsonl (needed for calibration).")
        print("Expected JSONL rows like:")
        print('  {"engine_ts":"<ts>","horizon":"72h","return_pct":1.25}')
        print("Tip: engine_ts must match the 'engine.ts' field from ai_overlay.jsonl")
        return

    if not outcomes:
        print("logs/outcomes.jsonl exists but is empty (no outcomes recorded yet).")
        print("This is normal until your signals are older than the chosen horizon (e.g. 24h/72h).")
        return

    # index outcomes by engine_ts
    out_by_ts = {}
    for o in outcomes:
        ts = o.get("engine_ts")
        if ts:
            out_by_ts[ts] = o

    # build dataset rows (confidence, return_pct)
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
        if not isinstance(ret, (int, float)):
            continue
        rows.append((float(conf), float(ret)))

    if not rows:
        print("No matched rows between ai_overlay.jsonl and outcomes.jsonl yet.")
        print("Add outcomes with engine_ts matching engine.ts.")
        return

    # evaluate thresholds
    # Idea: if confidence >= T => "high confidence bucket"
    # show average return, win rate, and sample size for each bucket
    thresholds = sorted(set(int(c) for c, _ in rows))
    thresholds = [t for t in thresholds if 0 <= t <= 100]

    def stats(sub):
        n = len(sub)
        if n == 0:
            return None
        rets = [r for _, r in sub]
        avg = sum(rets) / n
        win = sum(1 for r in rets if r > 0) / n
        return {"n": n, "avg_return_pct": round(avg, 4), "win_rate": round(win, 4)}

    print("=== Confidence Calibration (based on outcomes) ===")
    print(f"matched_samples: {len(rows)}")

    best = None
    for T in thresholds:
        hi = [(c, r) for c, r in rows if c >= T]
        lo = [(c, r) for c, r in rows if c < T]
        hi_s = stats(hi)
        lo_s = stats(lo)
        if not hi_s or hi_s["n"] < 5:
            continue

        # pick a simple objective: maximize avg_return_pct in high-confidence bucket
        score = hi_s["avg_return_pct"]
        if best is None or score > best["score"]:
            best = {"T": T, "score": score, "hi": hi_s, "lo": lo_s}

        print(f"T={T:>3}  HIGH {hi_s}   LOW {lo_s}")

    if best:
        print("\n--- Suggested min_confidence_ok ---")
        print(f"T* = {best['T']}  (best HIGH avg_return_pct={best['score']})")
        print(f"HIGH bucket: {best['hi']}")
        print(f"LOW bucket : {best['lo']}")
    else:
        print("\nNot enough data yet. Need at least 5 matched outcomes in a bucket.")

if __name__ == "__main__":
    main()
