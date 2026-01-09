from fastapi import APIRouter, Query
import requests
import os

from app.ai.schemas import OverlayRequest
from app.ai.overlay_heuristic import build_overlay as build_overlay_heuristic
from app.ai.audit_logger import log_ai_overlay

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

# Base URL for self-calls (Render listens on $PORT, not hardcoded 8000)
SELF_BASE_URL = os.getenv("SELF_BASE_URL") or f"http://127.0.0.1:{os.getenv('PORT', '8000')}"


@router.get("/plan_with_ai")
def plan_with_ai(current_exposure: float = Query(..., ge=0.0, le=1.0)):
    # 1) call existing deterministic plan endpoint
    r = requests.get(
        f"{SELF_BASE_URL}/portfolio/plan",
        params={"current_exposure": current_exposure},
        timeout=10
    )
    r.raise_for_status()
    base = r.json()

    # 2) fetch config snapshot from /health (for thresholds / weights)
    cfg = {}
    try:
        h = requests.get(f"{SELF_BASE_URL}/health", timeout=5)
        if h.ok:
            cfg = (h.json() or {}).get("config", {}) or {}
    except Exception:
        cfg = {}

    # 3) build overlay (heuristic; LLM can be added later too)
    overlay_req = OverlayRequest(
        engine=base,
        current_exposure=current_exposure,
        config=cfg,
        features={}
    )
    overlay = build_overlay_heuristic(overlay_req)

    # 4) audit log
    log_ai_overlay(
        {
            "route": "/portfolio/plan_with_ai",
            "use_llm": False,
            "symbol": overlay_req.engine.symbol,
            "engine": overlay_req.engine.dict(),
            "current_exposure": current_exposure,
            "config": cfg,
            "constraints": overlay_req.constraints.dict(),
            "overlay": overlay.dict(),
        }
    )

    # 5) return combined
    return {
        **base,
        "ai_overlay": overlay.dict()
    }
