import json
import sys
from pathlib import Path

SUGGEST = Path("logs/suggestions.json")
MAIN = Path("main.py")

def main():
    if not SUGGEST.exists():
        print("No logs/suggestions.json found. Run: python3 scripts/suggest_min_confidence.py")
        sys.exit(1)

    sug = json.loads(SUGGEST.read_text(encoding="utf-8"))
    new_min = sug.get("suggested_min_confidence_ok")
    if not isinstance(new_min, int):
        print("suggestions.json missing suggested_min_confidence_ok (int).")
        sys.exit(1)

    # show plan (no changes yet)
    print("=== Human-in-the-loop apply ===")
    print(f"Proposed change: OVERLAY min_confidence_ok -> {new_min}")
    print("This will NOT modify engine thresholds. Only overlay defaults.")
    print("\nTo apply, run:")
    print(f"  python3 scripts/apply_suggestion.py APPLY\n")

    if len(sys.argv) < 2 or sys.argv[1] != "APPLY":
        return

    # Patch app/ai/schemas.py default OverlayConstraints.min_confidence_ok
    schema = Path("app/ai/schemas.py")
    if not schema.exists():
        print("Missing app/ai/schemas.py")
        sys.exit(1)

    s = schema.read_text(encoding="utf-8")
    old = "min_confidence_ok: int = Field(default=45, ge=0, le=100)"
    if old not in s:
        # allow patching if current default differs
        import re
        m = re.search(r"min_confidence_ok:\s*int\s*=\s*Field\(default=(\d+),\s*ge=0,\s*le=100\)", s)
        if not m:
            print("Could not find min_confidence_ok Field(...) line to patch.")
            sys.exit(1)
        current = m.group(1)
        s = re.sub(
            r"min_confidence_ok:\s*int\s*=\s*Field\(default=\d+,\s*ge=0,\s*le=100\)",
            f"min_confidence_ok: int = Field(default={new_min}, ge=0, le=100)",
            s,
        )
        print(f"Patched schemas.py default {current} -> {new_min}")
    else:
        s = s.replace(old, f"min_confidence_ok: int = Field(default={new_min}, ge=0, le=100)")
        print(f"Patched schemas.py default 45 -> {new_min}")

    schema.write_text(s, encoding="utf-8")

    # Write applied marker (audit)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/applied_suggestions.jsonl").open("a", encoding="utf-8").write(
        json.dumps({"applied_min_confidence_ok": new_min, "source": "suggestions.json"}) + "\n"
    )

    print("✅ Applied. Restart/reload server to take effect.")

if __name__ == "__main__":
    main()
