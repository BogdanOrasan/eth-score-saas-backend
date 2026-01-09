from __future__ import annotations
from typing import List
from .schemas import OverlayRequest, OverlayResponse


def _divergence(scores: dict) -> float:
    vals = list(scores.values())
    if not vals:
        return 0.0
    return max(vals) - min(vals)


def build_overlay(req: OverlayRequest) -> OverlayResponse:
    engine = req.engine
    constraints = req.constraints

    risk_flags: List[str] = []
    key_drivers: List[str] = []
    what_to_watch: List[str] = []

    # 1) Confidence gate
    if engine.confidence < constraints.min_confidence_ok:
        risk_flags.append(f"low_confidence(<{constraints.min_confidence_ok})")

    # 2) Divergence gate between timeframes
    div = _divergence(engine.scores)
    if div >= constraints.divergence_review_threshold:
        risk_flags.append(f"timeframe_divergence(>={constraints.divergence_review_threshold})")

    # 3) Drivers summary (deterministic)
    for tf, sc in sorted(engine.scores.items()):
        if sc >= 20:
            key_drivers.append(f"{tf}: strong positive momentum ({sc})")
        elif sc >= 5:
            key_drivers.append(f"{tf}: mild positive bias ({sc})")
        elif sc <= -20:
            key_drivers.append(f"{tf}: strong negative momentum ({sc})")
        elif sc <= -5:
            key_drivers.append(f"{tf}: mild negative bias ({sc})")
        else:
            key_drivers.append(f"{tf}: neutral/mixed ({sc})")

    # 4) What to watch (use engine thresholds if provided)
    exit_thr = req.config.get("EXIT_THRESHOLD")
    reduce_thr = req.config.get("REDUCE_THRESHOLD")

    if exit_thr is not None:
        what_to_watch.append(f"if weighted_score <= {exit_thr}: engine likely EXIT")
    else:
        what_to_watch.append("watch weighted_score dropping further (exit risk)")

    if reduce_thr is not None:
        what_to_watch.append(f"if weighted_score <= {reduce_thr}: engine likely REDUCE")
    else:
        what_to_watch.append("watch persistent negative 1d/4h for reduce risk")

    # 5) Verdict
    verdict = "OK" if len(risk_flags) == 0 else "REVIEW"

    # 6) Explanation
    explanation_parts = [
        f"Engine decision: {engine.decision} (confidence {engine.confidence}, weighted_score {engine.weighted_score:.2f}).",
        f"Exposure now: {req.current_exposure:.2f}.",
    ]
    if risk_flags:
        explanation_parts.append("Flags: " + ", ".join(risk_flags) + ".")
    else:
        explanation_parts.append("No risk flags detected by overlay rules.")

    explanation = " ".join(explanation_parts)

    return OverlayResponse(
        verdict=verdict,
        explanation=explanation,
        key_drivers=key_drivers[:6],
        what_to_watch=what_to_watch[:5],
        risk_flags=risk_flags,
        suggested_action="keep_engine_decision",
    )
