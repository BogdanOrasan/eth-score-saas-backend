import json
from collections import Counter
from pathlib import Path

LOG_PATH = Path("logs/ai_overlay.jsonl")

def main():
    if not LOG_PATH.exists():
        print(f"Missing log file: {LOG_PATH}")
        return

    routes = Counter()
    verdicts = Counter()
    risk_flags = Counter()
    confidences = []
    weighted_scores = []
    n = 0

    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            n += 1
            routes[obj.get("route", "unknown")] += 1

            overlay = obj.get("overlay", {}) or {}
            verdicts[overlay.get("verdict", "unknown")] += 1
            for rf in (overlay.get("risk_flags", []) or []):
                risk_flags[rf] += 1

            engine = obj.get("engine", {}) or {}
            c = engine.get("confidence")
            if isinstance(c, (int, float)):
                confidences.append(float(c))

            ws = engine.get("weighted_score")
            if isinstance(ws, (int, float)):
                weighted_scores.append(float(ws))

    def stats(xs):
        if not xs:
            return None
        xs = sorted(xs)
        return {
            "min": xs[0],
            "p25": xs[int(0.25 * (len(xs)-1))],
            "median": xs[int(0.50 * (len(xs)-1))],
            "p75": xs[int(0.75 * (len(xs)-1))],
            "max": xs[-1],
            "count": len(xs),
        }

    print("=== AI Overlay Audit Summary ===")
    print(f"events: {n}")
    print("\n-- routes --")
    for k, v in routes.most_common():
        print(f"{k}: {v}")

    print("\n-- verdicts --")
    for k, v in verdicts.most_common():
        print(f"{k}: {v}")

    print("\n-- top risk_flags --")
    for k, v in risk_flags.most_common(10):
        print(f"{k}: {v}")

    print("\n-- confidence stats --")
    cs = stats(confidences)
    print(cs if cs else "no confidence data")

    print("\n-- weighted_score stats --")
    ws = stats(weighted_scores)
    print(ws if ws else "no weighted_score data")

if __name__ == "__main__":
    main()
