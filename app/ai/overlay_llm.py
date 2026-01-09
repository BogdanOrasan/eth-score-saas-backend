from typing import Any, Dict
import json
import os

from openai import OpenAI

from app.ai.schemas import OverlayRequest, OverlayResponse
from app.ai.overlay_heuristic import build_overlay as heuristic_overlay


def build_overlay_llm(
    req: OverlayRequest,
    model: str = "gpt-4o-mini",
    timeout_s: int = 15,
) -> OverlayResponse:
    """
    LLM overlay with SAFE fallback.
    - Never overrides engine decision
    - If LLM fails or output invalid -> heuristic overlay
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[LLM] missing OPENAI_API_KEY -> fallback")
        return heuristic_overlay(req)

    try:
        client = OpenAI(api_key=api_key)

        payload: Dict[str, Any] = {
            "engine": req.engine.dict(),
            "current_exposure": req.current_exposure,
            "config": req.config,
            "constraints": req.constraints.dict(),
            "rules": {
                "must_not_override_decision": True,
                "output_must_match_schema": True,
                "if_uncertain_verdict": "REVIEW",
            },
            "schema": {
                "verdict": "OK|REVIEW",
                "explanation": "string",
                "key_drivers": "string[]",
                "what_to_watch": "string[]",
                "risk_flags": "string[]",
                "suggested_action": "keep_engine_decision"
            }
        }

        # Responses API (OpenAI Python SDK v2)
        r = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a risk/explainability layer for a crypto trading engine.\n"
                        "You MUST NOT change the engine decision.\n"
                        "Return STRICT JSON (no markdown, no extra text) matching the schema.\n"
                        "If uncertain, set verdict to REVIEW.\n"
                        "suggested_action must always be 'keep_engine_decision'."
                    ),
                },
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.2,
            max_output_tokens=400,
            timeout=timeout_s,
        )

        # Extract text output
        text = ""
        for item in (r.output or []):
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in ("output_text", "text"):
                        text += getattr(c, "text", "") or ""

        text = text.strip()
        if not text:
            print("[LLM] empty response -> fallback")
            return heuristic_overlay(req)

        data = json.loads(text)

        # hard safety: never allow override
        data["suggested_action"] = "keep_engine_decision"
        print("[LLM] response accepted")
        return OverlayResponse(**data)

    except Exception as e:
        print(f"[LLM] error -> fallback: {type(e).__name__}: {e}")
        return heuristic_overlay(req)
