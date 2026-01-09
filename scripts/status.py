import json
from pathlib import Path
from datetime import datetime, timezone

AUDIT = Path("logs/ai_overlay.jsonl")
OUTCOMES = Path("logs/outcomes.jsonl")
SUGGEST = Path("logs/suggestions.json")

HORIZONS = ["1h", "24h", "72h"]

def tail_jsonl(path: Path, n: int = 1):
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = []
    for ln in lines[-n:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out

def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

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

def main():
    last_events = tail_jsonl(AUDIT, 5)
    if not last_events:
        print("No audit events yet. Call /portfolio/plan_with_ai or /ai/overlay first.")
        return

    last = last_events[-1]
    eng = last.get("engine", {})
    ov = last.get("overlay", {})
    route = last.get("route", "?")
    symbol = eng.get("symbol")
    engine_ts = eng.get("ts")

    print("=== ETH Score SaaS — Status ===")
    print(f"time_utc: {datetime.now(timezone.utc).isoformat()}")
    print(f"last_route: {route}")
    print(f"symbol: {symbol}")
    print(f"engine_ts: {engine_ts}")
    print(f"decision: {eng.get('decision')}   confidence: {eng.get('confidence')}   weighted_score: {eng.get('weighted_score')}")
    print(f"scores: {eng.get('scores')}")
    print(f"overlay_verdict: {ov.get('verdict')}   risk_flags: {ov.get('risk_flags')}")
    print(f"what_to_watch: {ov.get('what_to_watch')}")

    # outcomes per horizon for THIS engine_ts (if available)
    outcomes = load_jsonl(OUTCOMES)
    out_by_h = {}
    for o in outcomes:
        if o.get("engine_ts") == engine_ts and o.get("horizon") in HORIZONS:
            out_by_h[o.get("horizon")] = o

    print("\n--- outcomes for last engine_ts ---")
    for h in HORIZONS:
        o = out_by_h.get(h)
        if not o:
            print(f"{h}: (not ready)")
        else:
            print(f"{h}: return_pct={o.get('return_pct'):.6f}  price_t0={o.get('price_t0')}  price_t1={o.get('price_t1')}")

    # last outcomes overall (last 5 lines)
    outs_tail = tail_jsonl(OUTCOMES, 5)
    print("\n--- last outcomes (any ts) ---")
    if outs_tail:
        for o in outs_tail:
            print(f"{o.get('engine_ts')}  horizon={o.get('horizon')}  return_pct={o.get('return_pct')}")
    else:
        print("(none yet)")

    # suggestion (if any)
    sug = load_json(SUGGEST)
    print("\n--- suggestion ---")
    if sug:
        print(f"min_confidence_ok -> {sug.get('suggested_min_confidence_ok')}  horizon={sug.get('horizon')}  matched={sug.get('matched_samples')}")
        print(f"high_bucket: {sug.get('high_bucket')}")
        print(f"low_bucket_avg_return_pct: {sug.get('low_bucket_avg_return_pct')}")
    else:
        print("(none yet) run: python3 scripts/suggest_min_confidence.py")

    # quick recent history
    print("\n--- recent events (last 5) ---")
    for e in last_events:
        eng2 = e.get("engine", {})
        ov2 = e.get("overlay", {})
        print(f"{eng2.get('ts')}  decision={eng2.get('decision')}  conf={eng2.get('confidence')}  verdict={ov2.get('verdict')}  route={e.get('route')}")

if __name__ == "__main__":
    main()
