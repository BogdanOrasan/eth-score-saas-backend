import json
import os
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

AUDIT_LOG = Path("logs/ai_overlay.jsonl")
OUTCOMES = Path("logs/outcomes.jsonl")

# horizon: 1h / 24h / 72h
HORIZON = os.getenv("H", "1h")

# bin size for confidence
BIN_SIZE = int(os.getenv("BIN", "5"))  # e.g. 5 => 0–5, 5–10, ...

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
    events = load_jsonl(AUDIT_LOG)
    outcomes = [o for o in load_jsonl(OUTCOMES) if o.get("horizon") == HORIZON]

    if not events or not outcomes:
        print(f"Missing data for horizon={HORIZON}")
        return

    out_by_ts = {o.get("engine_ts"): o for o in outcomes if o.get("engine_ts")}

    # collect (confidence, return)
    rows = []
    for e in events:
        eng = e.get("engine", {}) or {}
        ts = eng.get("ts")
        conf = eng.get("confidence")
        if ts is None or conf is None:
            continue
        o = out_by_ts.get(ts)
        if not o:
            continue
        ret = o.get("return_pct")
        if isinstance(conf, (int, float)) and isinstance(ret, (int, float)):
            rows.append((float(conf), float(ret)))

    if not rows:
        print("No matched samples.")
        return

    # binning
    bins = {}
    for conf, ret in rows:
        b = int(conf // BIN_SIZE) * BIN_SIZE
        bins.setdefault(b, []).append(ret)

    # compute means
    xs = sorted(bins.keys())
    ys = [mean(bins[b]) for b in xs]
    ns = [len(bins[b]) for b in xs]

    print("bins:", {b: len(bins[b]) for b in xs})

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(f"Confidence (bin size={BIN_SIZE})")
    plt.ylabel("Average return %")
    plt.title(f"Avg Return vs Confidence (binned, {HORIZON})")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
