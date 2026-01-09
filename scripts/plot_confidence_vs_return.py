import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

AUDIT_LOG = Path("logs/ai_overlay.jsonl")
OUTCOMES = Path("logs/outcomes.jsonl")

# alege horizon prin env: H=1h / 24h / 72h
HORIZON = os.getenv("H", "1h")

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

    if not events:
        print("No audit events found.")
        return
    if not outcomes:
        print(f"No outcomes found for horizon={HORIZON}.")
        print("Tip: generate outcomes first (e.g. python3 scripts/generate_outcomes_multi.py)")
        return

    out_by_ts = {o.get("engine_ts"): o for o in outcomes if o.get("engine_ts")}

    xs = []
    ys = []
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
        if isinstance(conf, (int, float)) and isinstance(ret, (int, float)):
            xs.append(float(conf))
            ys.append(float(ret))

    if not xs:
        print(f"No matched samples between audit and outcomes for horizon={HORIZON}.")
        return

    print(f"matched_samples: {len(xs)}  horizon={HORIZON}")

    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("Engine confidence")
    plt.ylabel("Return %")
    plt.title(f"Confidence vs Return ({HORIZON})")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
