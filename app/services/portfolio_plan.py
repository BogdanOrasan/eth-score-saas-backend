from typing import Optional, Any, Dict
from fastapi import HTTPException

def compute_portfolio_plan(
    *,
    engine,
    text,
    symbol: str,
    min_exposure: float,
    current_exposure: Optional[float],
    clamp,
) -> Dict[str, Any]:
    """
    Shared implementation for /portfolio/plan.
    Avoids importing from main/router modules (prevents circular imports).
    """
    d = None
    with engine.begin() as conn:
        row = conn.execute(
            text("""
            SELECT ts, decision, confidence, details
            FROM decisions
            WHERE symbol=:symbol
            ORDER BY ts DESC
            LIMIT 1
            """),
            dict(symbol=symbol)
        ).fetchone()
        if row:
            d = {"ts": row[0], "decision": row[1], "confidence": int(row[2]), "details": row[3]}

    if not d:
        raise HTTPException(status_code=404, detail="No decision yet. Try POST /admin/run_all")

    plan = (d["details"] or {}).get("plan", None)
    if not plan:
        raise HTTPException(status_code=500, detail="No plan in details (unexpected). Run POST /admin/run_all again.")

    weighted_score = (d["details"] or {}).get("weighted_score")
    scores = (d["details"] or {}).get("scores")

    step_percent = int(plan.get("step_percent"))
    hint = plan.get("hint")

    target_exposure = None
    adjusted_step_percent = None

    if current_exposure is not None:
        raw_target = clamp(float(current_exposure) + (step_percent / 100.0), 0.0, 1.0)
        if d["decision"] != "EXIT":
            raw_target = max(raw_target, float(min_exposure))
        target_exposure = float(raw_target)
        adjusted_step_percent = int(round((target_exposure - float(current_exposure)) * 100.0))

    return {
        "symbol": symbol,
        "ts": d["ts"],
        "decision": d["decision"],
        "confidence": d["confidence"],
        "weighted_score": weighted_score,
        "scores": scores,
        "step_percent": step_percent,
        "hint": hint,
        "current_exposure": current_exposure,
        "target_exposure": target_exposure,
        "adjusted_step_percent": adjusted_step_percent,
        "min_exposure": float(min_exposure),
    }
