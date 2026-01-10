from __future__ import annotations

from typing import Any, Dict


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(round(float(x)))
    except Exception:
        return default


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def normalize_plan_with_ai(base: Dict[str, Any], ai_overlay: Dict[str, Any], cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Returneaza o schema stabila pentru frontend, indiferent de cum arata base/overlay intern.
    """
    symbol = base.get("symbol") or "ETHUSDT"
    ts = base.get("ts")

    decision = (base.get("decision") or "HOLD").upper()
    confidence = _clamp(_to_float(base.get("confidence"), 0.0), 0.0, 100.0)
    weighted_score = _to_float(base.get("weighted_score"), 0.0)

    scores = base.get("scores") or {}
    scores_out = {
        "4h": _to_int(scores.get("4h"), 0),
        "1d": _to_int(scores.get("1d"), 0),
        "1w": _to_int(scores.get("1w"), 0),
    }

    current_exposure = _to_float(base.get("current_exposure"), 0.0)
    target_exposure = _to_float(base.get("target_exposure", current_exposure), current_exposure)

    step_percent = _to_float(base.get("step_percent"), 0.0)
    adjusted_step_percent = _to_float(base.get("adjusted_step_percent", step_percent), step_percent)

    # UI summary: scurt + actionabil (nu dump de text)
    if adjusted_step_percent > 0:
        ui_summary = f"BUY: increase exposure by {abs(adjusted_step_percent):.2f}% (to {target_exposure:.2f})."
    elif adjusted_step_percent < 0:
        ui_summary = f"SELL: reduce exposure by {abs(adjusted_step_percent):.2f}% (to {target_exposure:.2f})."
    else:
        ui_summary = f"HOLD: keep exposure at {current_exposure:.2f}."

    action = "NO_TRADE"
    if adjusted_step_percent > 0:
        action = "INCREASE"
    elif adjusted_step_percent < 0:
        action = "DECREASE"

    return {
        "ok": True,
        "overlay": {
            "symbol": symbol,
            "ts": ts,
            "decision": decision,
            "confidence": round(confidence, 1),
            "weighted_score": round(weighted_score, 2),
            "scores": scores_out,
            "current_exposure": round(current_exposure, 4),
            "target_exposure": round(target_exposure, 4),
            "step_percent": round(step_percent, 4),
            "adjusted_step_percent": round(adjusted_step_percent, 4),
            "action": action,
            "ui_summary": ui_summary,
            "thresholds": {
                "ACCUMULATE_THRESHOLD": (cfg or {}).get("ACCUMULATE_THRESHOLD"),
                "REDUCE_THRESHOLD": (cfg or {}).get("REDUCE_THRESHOLD"),
                "EXIT_THRESHOLD": (cfg or {}).get("EXIT_THRESHOLD"),
            },
            "ai_overlay": ai_overlay,  # păstrăm detaliile existente
        },
    }