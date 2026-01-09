import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt

AUDIT_LOG = Path("logs/ai_overlay.jsonl")
OUTCOMES = Path("logs/outcomes.jsonl")

SYMBOL = "ETH/USDT"
TIMEFRAME = "1h"
HORIZON_HOURS = 72

def parse_iso(ts: str) -> datetime:
    # expects ISO with timezone (your engine uses +00:00)
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

    # Fetch recent OHLCV window wide enough: last N days
    # We'll fetch progressively per event to keep it simple (still fast on small logs).
    created = 0
    skipped = 0

    for e in events:
        eng = (e.get("engine") or {})
        engine_ts = eng.get("ts")
        if not engine_ts:
            continue

        key = (engine_ts, f"{HORIZON_HOURS}h")
        if key in existing_keys:
            skipped += 1
            continue

        t0 = parse_iso(engine_ts).astimezone(timezone.utc)
        t1 = t0 + timedelta(hours=HORIZON_HOURS)

        # Kraken OHLCV requires since in ms
        since_ms = int((t0 - timedelta(hours=5)).timestamp() * 1000)  # small buffer
        ohlcv = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since_ms, limit=2000)

        # Find closest candle at/after t0 and at/after t1
        p0 = None
        p1 = None
        for ts_ms, o, h, l, c, v in ohlcv:
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            if p0 is None and ts >= t0:
                p0 = c
            if ts >= t1:
                p1 = c
                break

        if p0 is None or p1 is None:
            # not enough history yet (e.g. t1 in future) — skip silently
            continue

        ret_pct = (p1 - p0) / p0 * 100.0

        append_jsonl(
            OUTCOMES,
            {
                "engine_ts": engine_ts,
                "horizon": f"{HORIZON_HOURS}h",
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME,
                "price_t0": p0,
                "price_t1": p1,
                "return_pct": ret_pct,
            },
        )
        created += 1

    print(f"created: {created}")
    print(f"skipped_existing: {skipped}")
    print(f"outcomes_file: {OUTCOMES}")

if __name__ == "__main__":
    main()
