from fastapi import APIRouter, Query
from app.services.portfolio_plan import compute_portfolio_plan
import os

from app.ai.schemas import OverlayRequest
from app.ai.overlay_heuristic import build_overlay as build_overlay_heuristic
from app.ai.audit_logger import log_ai_overlay

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

# Base URL for self-calls (Render listens on $PORT, not hardcoded 8000)
SELF_BASE_URL = os.getenv("SELF_BASE_URL") or f"http://127.0.0.1:{os.getenv('PORT', '8000')}"


@router.get("/plan_with_ai")
def plan_with_ai(current_exposure: float = Query(..., ge=0.0, le=1.0)):
    # 1) call deterministic plan directly (no HTTP self-call; Render-safe)
    from main import engine, text, SYMBOL_ETHUSDT, MIN_EXPOSURE, clamp
    base = compute_portfolio_plan(
        engine=engine,
        text=text,
        symbol=SYMBOL_ETHUSDT,
        min_exposure=MIN_EXPOSURE,
        current_exposure=current_exposure,
        clamp=clamp,
    )

    # Pydantic schema expects ts as str; DB may return datetime
    try:
        if isinstance(base.get('ts'), object) and hasattr(base.get('ts'), 'isoformat'):
            base['ts'] = base['ts'].isoformat()
    except Exception:
        pass

    # Safety: OverlayRequest expects non-negative step_percent
    try:
        if isinstance(base, dict) and 'step_percent' in base:
            sp = float(base.get('step_percent') or 0.0)
            if sp < 0.0:
                base['step_percent'] = 0.0
    except Exception:
        pass

    # 2) config snapshot (avoid HTTP self-call)
    cfg = {}
    try:
        from main import ACCUMULATE_THRESHOLD, REDUCE_THRESHOLD, EXIT_THRESHOLD, W_4H, W_1D, W_1W, MIN_EXPOSURE
        cfg = {
            "ACCUMULATE_THRESHOLD": int(ACCUMULATE_THRESHOLD),
            "REDUCE_THRESHOLD": int(REDUCE_THRESHOLD),
            "EXIT_THRESHOLD": int(EXIT_THRESHOLD),
            "weights": {"4h": float(W_4H), "1d": float(W_1D), "1w": float(W_1W)},
            "MIN_EXPOSURE": float(MIN_EXPOSURE),
        }
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

    # 5) return normalized overlay (stable schema for frontend)
    from app.ai_overlay_normalize import normalize_plan_with_ai
    return normalize_plan_with_ai(base=base, ai_overlay=overlay.dict(), cfg=cfg)
