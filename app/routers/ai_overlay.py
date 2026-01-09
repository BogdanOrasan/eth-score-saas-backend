import os
from fastapi import APIRouter

from app.ai.schemas import OverlayRequest, OverlayResponse
from app.ai.overlay_heuristic import build_overlay as build_overlay_heuristic
from app.ai.overlay_llm import build_overlay_llm
from app.ai.audit_logger import log_ai_overlay

router = APIRouter(prefix="/ai", tags=["ai"])


@router.post("/overlay", response_model=OverlayResponse)
def overlay(req: OverlayRequest) -> OverlayResponse:
    use_llm = os.getenv("USE_LLM_OVERLAY", "0") == "1"

    if use_llm:
        res = build_overlay_llm(req)
    else:
        res = build_overlay_heuristic(req)

    # audit log (never breaks request)
    log_ai_overlay(
        {
            "route": "/ai/overlay",
            "use_llm": use_llm,
            "symbol": req.engine.symbol,
            "engine": req.engine.dict(),
            "current_exposure": req.current_exposure,
            "config": req.config,
            "constraints": req.constraints.dict(),
            "overlay": res.dict(),
        }
    )

    return res
