import json
import sys
from pathlib import Path

APPLIED = Path("logs/applied_suggestions.jsonl")
SCHEMA = Path("app/ai/schemas.py")

def last_applied_value():
    if not APPLIED.exists():
        return None
    last = None
    for line in APPLIED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last = json.loads(line)
        except Exception:
            pass
    if not last:
        return None
    v = last.get("applied_min_confidence_ok")
    return v if isinstance(v, int) else None

def patch_schema(new_min: int):
    if not SCHEMA.exists():
        print("Missing app/ai/schemas.py")
        sys.exit(1)

    s = SCHEMA.read_text(encoding="utf-8")

    import re
    m = re.search(r"min_confidence_ok:\s*int\s*=\s*Field\(default=(\d+),\s*ge=0,\s*le=100\)", s)
    if not m:
        print("Could not find min_confidence_ok Field(...) line to patch.")
        sys.exit(1)

    current = int(m.group(1))
    s2 = re.sub(
        r"min_confidence_ok:\s*int\s*=\s*Field\(default=\d+,\s*ge=0,\s*le=100\)",
        f"min_confidence_ok: int = Field(default={new_min}, ge=0, le=100)",
        s,
    )
    SCHEMA.write_text(s2, encoding="utf-8")
    return current

def main():
    # Usage:
    #  python3 scripts/rollback_suggestion.py APPLY 45
    #  python3 scripts/rollback_suggestion.py APPLY --last
    print("=== Rollback min_confidence_ok ===")
    print("Dry-run by default. To apply, use APPLY.\n")

    if len(sys.argv) < 2 or sys.argv[1] != "APPLY":
        print("Examples:")
        print("  python3 scripts/rollback_suggestion.py APPLY 45")
        print("  python3 scripts/rollback_suggestion.py APPLY --last")
        return

    if len(sys.argv) < 3:
        print("Missing rollback target. Use a number or --last")
        sys.exit(1)

    target = sys.argv[2]
    if target == "--last":
        v = last_applied_value()
        if v is None:
            print("No last applied value found in logs/applied_suggestions.jsonl")
            sys.exit(1)
        new_min = v
    else:
        try:
            new_min = int(target)
        except Exception:
            print("Invalid target. Provide integer or --last")
            sys.exit(1)

    prev = patch_schema(new_min)

    Path("logs").mkdir(exist_ok=True)
    Path("logs/applied_suggestions.jsonl").open("a", encoding="utf-8").write(
        json.dumps({"rollback_min_confidence_ok": new_min, "previous": prev, "source": "rollback"}) + "\n"
    )

    print(f"✅ Rolled back min_confidence_ok: {prev} -> {new_min}")
    print("Restart/reload server to take effect.")

if __name__ == "__main__":
    main()
