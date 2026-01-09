import json
from pathlib import Path
from collections import Counter

AUDIT_LOG = Path("logs/ai_overlay.jsonl")
OUTCOMES = Path("logs/outcomes.jsonl")

TARGET_HORIZON = "24h"  # change to "72h" later
POS_THRESHOLD = 0.0     # return_pct > 0 => "win"

def load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

def main():
    events = load_jsonl(AUDIT_LOG)
    if not events:
        print("No audit events.")
        return

    outcomes = [o for o in load_jsonl(OUTCOMES) if o.get("horizon") == TARGET_HORIZON]
    if not outcomes:
        print(f"No outcomes yet for horizon={TARGET_HORIZON}.")
        return

    out_by_ts = {o.get("engine_ts"): o for o in outcomes if o.get("engine_ts")}
    rows = []
    for e in events:
        eng = (e.get("engine") or {})
        ov = (e.get("overlay") or {})
        ts = eng.get("ts")
        if not ts or ts not in out_by_ts:
            continue

        ret = out_by_ts[ts].get("return_pct")
        if not isinstance(ret, (int, float)):
            continue

        verdict = ov.get("verdict", "unknown")
        warned = (verdict == "REVIEW") or bool(ov.get("risk_flags"))

        outcome = "win" if ret > POS_THRESHOLD else "loss"
        rows.append((ts, warned, verdict, ret, outcome, e.get("route", "")))

    if not rows:
        print("No matched rows between audit and outcomes yet.")
        return

    # Attribution labels:
    # - warned & loss  => correct_warning
    # - warned & win   => false_alarm
    # - not_warned & loss => missed_risk
    # - not_warned & win  => ok
    counts = Counter()
    for _, warned, _, _, outcome, _ in rows:
        if warned and outcome == "loss":
            counts["correct_warning"] += 1
        elif warned and outcome == "win":
            counts["false_alarm"] += 1
        elif (not warned) and outcome == "loss":
            counts["missed_risk"] += 1
        else:
            counts["ok"] += 1

    print("=== Overlay Attribution ===")
    print(f"horizon: {TARGET_HORIZON}  samples: {len(rows)}  win_threshold: return_pct>{POS_THRESHOLD}")
    for k in ["correct_warning", "false_alarm", "missed_risk", "ok"]:
        print(f"{k}: {counts.get(k,0)}")

    # show last 10 details
    print("\n-- last 10 matched --")
    for ts, warned, verdict, ret, outcome, route in rows[-10:]:
        print(f"{ts}  route={route}  verdict={verdict}  warned={warned}  return_pct={ret:.3f}  outcome={outcome}")

if __name__ == "__main__":
    main()
