from typing import Dict, List, Literal, Any
from pydantic import BaseModel, Field

Verdict = Literal["OK", "REVIEW"]


class EngineOutput(BaseModel):
    symbol: str
    ts: str
    decision: str
    confidence: int = Field(ge=0, le=100)
    weighted_score: float
    scores: Dict[str, float]  # ex: {"4h": -10, "1d": -19, "1w": -7}
    step_percent: float = Field(ge=0.0)


class OverlayConstraints(BaseModel):
    min_confidence_ok: int = Field(default=45, ge=0, le=100)
    divergence_review_threshold: float = Field(default=25.0, ge=0.0)
    allow_override_decision: bool = False  # safe default


class OverlayRequest(BaseModel):
    engine: EngineOutput
    current_exposure: float = Field(ge=0.0, le=1.0)
    config: Dict[str, Any] = Field(default_factory=dict)     # snapshot config (thresholds, weights)
    features: Dict[str, Any] = Field(default_factory=dict)   # optional feature summaries
    constraints: OverlayConstraints = Field(default_factory=OverlayConstraints)


class OverlayResponse(BaseModel):
    verdict: Verdict
    explanation: str
    key_drivers: List[str] = Field(default_factory=list)
    what_to_watch: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    suggested_action: str = "keep_engine_decision"
