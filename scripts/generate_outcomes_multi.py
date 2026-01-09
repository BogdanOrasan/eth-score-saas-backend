import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt

AUDIT_LOG = Path("logs/ai_overlay.jsonl")
OUTCOMES = Path("logs/outcomes.jsonl")

SYMBOL = "ETH/USDT"
TIMEFRAME = "1h"
HORIZONS = [1, 24, 72]  # hours

def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)

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

def append_jsonl(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    events = load_jsonl(AUDIT_LOG)
    if not events:
        print("No audit events found.")
        return

    existing = load_jsonl(OUTCOMES)
    existing_keys = set((o.get("engine_ts"), o.get("horizon")) for o in existing)

    ex = ccxt.kraken()  # public data only
    ex.load_markets()

    created = 0
    skipped_existing = 0
    skipped_not_ready = 0

    for e in events:
        eng = (e.get("engine") or {})
        engine_ts = eng.get("ts")
        if not engine_ts:
            continue

        t0 = parse_iso(engine_ts).astimezone(timezone.utc)

        # one OHLCV fetch per event, big enough for 72h window
        since_ms = int((t0 - timedelta(hours=6)).timestamp() * 1000)
        ohlcv = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since_ms, limit=2500)

        # precompute price at/after t0
        p0 = None
        idx0 = None
        for i, (ts_ms, o, h, l, c, v) in enumerate(ohlcv):
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            if ts >= t0:
                p0 = c
                idx0 = i
                break
        if p0 is None:
            continue

        for h in HORIZONS:
            horizon = f"{h}h"
            key = (engine_ts, horizon)
            if key in existing_keys:
                skipped_existing += 1
                continue

            t1 = t0 + timedelta(hours=h)

            p1 = None
            for ts_ms, o, hh, ll, c, v in ohlcv[idx0:]:
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                if ts >= t1:
                    p1 = c
                    break

            if p1 is None:
                skipped_not_ready += 1
                continue

            ret_pct = (p1 - p0) / p0 * 100.0
            append_jsonl(
                OUTCOMES,
                {
                    "engine_ts": engine_ts,
                    "horizon": horizon,
                    "symbol": SYMBOL,
                    "timeframe": TIMEFRAME,
                    "price_t0": p0,
                    "price_t1": p1,
                    "return_pct": ret_pct,
                },
            )
            created += 1

    print(f"created: {created}")
    print(f"skipped_existing: {skipped_existing}")
    print(f"skipped_not_ready: {skipped_not_ready}")
    print(f"outcomes_file: {OUTCOMES}")

if __name__ == "__main__":
    main()
