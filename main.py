# ---------------- Runtime state (DB optional) ----------------
# NOTE: In local/dev we may run without Postgres. Keep the app functional.

DB_AVAILABLE = False
LAST_SCORES = {}   # timeframe -> score dict
LAST_DECISION = None  # decision dict

def set_db_available(value: bool) -> None:
    global DB_AVAILABLE
    DB_AVAILABLE = bool(value)

def set_last_score(timeframe: str, score: dict) -> None:
    global LAST_SCORES
    LAST_SCORES[timeframe] = score

def get_last_score(timeframe: str) -> dict:
    return LAST_SCORES.get(timeframe)

def set_last_decision(decision: dict) -> None:
    global LAST_DECISION
    LAST_DECISION = decision

def get_last_decision() -> dict:
    return LAST_DECISION

def write_recommendation_json(payload: dict, path: str = "logs/recommendation.json") -> None:
    import json
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str))
import os
import json

# Build marker (changes when file is reloaded)
BUILD_ID = __import__('datetime').datetime.utcnow().isoformat() + 'Z'

from datetime import datetime, timezone
from typing import Literal, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from app.routers.portfolio_ai import router as portfolio_ai_router
from app.routers.ai_overlay import router as ai_router
from sqlalchemy import create_engine, text
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()

# ---------------- Config helpers ----------------


# ---------------- In-memory fallback (when DB is down) ----------------
# Keys: (symbol, timeframe) -> last score dict
SCORE_CACHE = {}
# Last decision dict
DECISION_CACHE = None

def cache_score(symbol: str, timeframe: str, ts, total: int, pillars: dict, contrib: dict, action: str) -> None:
    SCORE_CACHE[(symbol, timeframe)] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "ts": ts,
        "score_total": int(total),
        "pillars": pillars,
        "contributors": contrib,
        "action": action,
    }

def cache_decision(symbol: str, ts, decision: str, confidence: int, details: dict) -> None:
    global DECISION_CACHE
    DECISION_CACHE = {
        "symbol": symbol,
        "ts": ts,
        "decision": decision,
        "confidence": int(confidence),
        "details": details,
    }
def _get_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return float(default)
    try:
        return float(v)
    except ValueError:
        return float(default)

def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return int(default)
    try:
        return int(float(v))
    except ValueError:
        return int(default)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://app:app@localhost:5432/ethscore")
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={
        "prepare_threshold": 0,
        "statement_cache_size": 0,
    },
)

# Decision thresholds (Option 1: EXIT only by weighted score)
ACCUMULATE_THRESHOLD = _get_int("ACCUMULATE_THRESHOLD", 20)
REDUCE_THRESHOLD = _get_int("REDUCE_THRESHOLD", -20)
EXIT_THRESHOLD = _get_int("EXIT_THRESHOLD", -45)

# TF weights
W_4H = _get_float("W_4H", 0.25)
W_1D = _get_float("W_1D", 0.35)
W_1W = _get_float("W_1W", 0.40)

# Exposure mgmt
MIN_EXPOSURE = _get_float("MIN_EXPOSURE", 0.30)

# Steps
ACC_STEP_SMALL = _get_int("ACCUMULATE_STEP_SMALL", 10)
ACC_STEP_BIG = _get_int("ACCUMULATE_STEP_BIG", 25)
ACC_BIG_CONF = _get_int("ACCUMULATE_BIG_CONF", 70)

RED_STEP_SMALL = _get_int("REDUCE_STEP_SMALL", 10)
RED_STEP_MED = _get_int("REDUCE_STEP_MED", 25)
RED_STEP_BIG = _get_int("REDUCE_STEP_BIG", 50)
REDUCE_MED_AT = _get_int("REDUCE_MED_AT", 25)  # |W| >= 25 -> medium
REDUCE_BIG_AT = _get_int("REDUCE_BIG_AT", 45)  # |W| >= 45 -> big

BINANCE_SPOT = "https://api.binance.com"
BINANCE_FAPI = "https://fapi.binance.com"

SYMBOL_ETHUSDT = "ETHUSDT"
SYMBOL_ETHBTC = "ETHBTC"
SYMBOL_BTCUSDT = "BTCUSDT"
SYMBOL_FUT = "ETHUSDT"

Timeframe = Literal["4h", "1d", "1w"]
Decision = Literal["ACCUMULATE", "HOLD", "REDUCE", "EXIT"]

app = FastAPI(title="ETH Score SaaS – Step 13 (Configurable thresholds & steps)")

DDL = """
CREATE TABLE IF NOT EXISTS candles (
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  open DOUBLE PRECISION NOT NULL,
  high DOUBLE PRECISION NOT NULL,
  low DOUBLE PRECISION NOT NULL,
  close DOUBLE PRECISION NOT NULL,
  volume DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (symbol, timeframe, ts)
);

CREATE TABLE IF NOT EXISTS derivatives_metrics (
  symbol TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  open_interest DOUBLE PRECISION NULL,
  funding_rate DOUBLE PRECISION NULL,
  PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS scores (
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  score_total INTEGER NOT NULL,
  pillars JSONB NOT NULL,
  contributors JSONB NOT NULL,
  action TEXT NOT NULL,
  PRIMARY KEY (symbol, timeframe, ts)
);

CREATE TABLE IF NOT EXISTS decisions (
  symbol TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  decision TEXT NOT NULL,
  confidence INTEGER NOT NULL,
  details JSONB NOT NULL,
  PRIMARY KEY (symbol, ts)
);
"""

def db_init():
    if not DB_AVAILABLE:
        return
    with engine.begin() as conn:
        for stmt in DDL.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))

# ---------------- Data fetch ----------------
def fetch_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    url = f"{BINANCE_SPOT}/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=20)
    r.raise_for_status()
    data = r.json()

    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_asset_volume","num_trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)

    # store OPEN time for consistency
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["ts","open","high","low","close","volume"]]

def fetch_funding_rate(symbol: str, limit: int = 100) -> pd.DataFrame:
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    return df.sort_values("fundingTime")

def fetch_open_interest(symbol: str) -> float:
    url = f"{BINANCE_FAPI}/fapi/v1/openInterest"
    r = requests.get(url, params={"symbol": symbol}, timeout=20)
    r.raise_for_status()
    return float(r.json()["openInterest"])

# ---------------- DB upserts ----------------
def upsert_candles(symbol: str, timeframe: str, df: pd.DataFrame):
    # DB may be down in local/dev; never crash the pipeline.
    try:
        with engine.begin() as conn:
            for _, r in df.iterrows():
                conn.execute(
                    text("""
                    INSERT INTO candles(symbol,timeframe,ts,open,high,low,close,volume)
                    VALUES(:symbol,:timeframe,:ts,:open,:high,:low,:close,:volume)
                    ON CONFLICT (symbol,timeframe,ts) DO UPDATE
                    SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                        close=EXCLUDED.close, volume=EXCLUDED.volume
                    """),
                    dict(
                        symbol=symbol,
                        timeframe=timeframe,
                        ts=r["ts"].to_pydatetime(),
                        open=float(r["open"]),
                        high=float(r["high"]),
                        low=float(r["low"]),
                        close=float(r["close"]),
                        volume=float(r["volume"]),
                    )
                )
    except Exception as e:
        try:
            set_db_available(False)
        except Exception:
            pass
        print("WARN: upsert_candles skipped (DB unavailable):", repr(e))
        return

def upsert_derivatives(symbol: str, ts: datetime, oi: Optional[float], funding: Optional[float]):
    # DB may be down in local/dev; never crash the pipeline.
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                INSERT INTO derivatives_metrics(symbol,ts,open_interest,funding_rate)
                VALUES(:symbol,:ts,:oi,:funding)
                ON CONFLICT (symbol,ts) DO UPDATE
                SET open_interest=EXCLUDED.open_interest,
                    funding_rate=EXCLUDED.funding_rate
                """),
                dict(symbol=symbol, ts=ts, oi=oi, funding=funding)
            )
    except Exception as e:
        try:
            set_db_available(False)
        except Exception:
            pass
        print("WARN: upsert_derivatives skipped (DB unavailable):", repr(e))
        return

def insert_score(symbol: str, timeframe: str, ts: datetime, total: int, pillars: dict, contrib: dict, action: str):
    # Always cache in memory (works without DB)
    try:
        cache_score(symbol, timeframe, ts, total, pillars, contrib, action)
    except Exception as e:
        print("WARN: cache_score failed:", repr(e))

    # Best-effort DB write
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                INSERT INTO scores(symbol,timeframe,ts,score_total,pillars,contributors,action)
                VALUES(:symbol,:timeframe,:ts,:score_total,CAST(:pillars AS jsonb),CAST(:contributors AS jsonb),:action)
                ON CONFLICT (symbol,timeframe,ts) DO UPDATE
                SET score_total=EXCLUDED.score_total,
                    pillars=EXCLUDED.pillars,
                    contributors=EXCLUDED.contributors,
                    action=EXCLUDED.action
                """),
                dict(
                    symbol=symbol,
                    timeframe=timeframe,
                    ts=ts,
                    score_total=int(total),
                    pillars=json.dumps(pillars),
                    contributors=json.dumps(contrib),
                    action=action,
                )
            )
    except Exception as e:
        print("WARN: insert_score skipped (DB unavailable):", repr(e))
        return

def insert_decision(symbol: str, ts: datetime, decision: str, confidence: int, details: dict):
    # Always cache in memory (works without DB)
    try:
        cache_decision(symbol, ts, decision, confidence, details)
    except Exception as e:
        print("WARN: cache_decision failed:", repr(e))

    # Best-effort DB write
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                INSERT INTO decisions(symbol,ts,decision,confidence,details)
                VALUES(:symbol,:ts,:decision,:confidence,CAST(:details AS jsonb))
                ON CONFLICT (symbol,ts) DO UPDATE
                SET decision=EXCLUDED.decision,
                    confidence=EXCLUDED.confidence,
                    details=EXCLUDED.details
                """),
                dict(
                    symbol=symbol,
                    ts=ts,
                    decision=decision,
                    confidence=int(confidence),
                    details=json.dumps(details),
                )
            )
    except Exception as e:
        print("WARN: insert_decision skipped (DB unavailable):", repr(e))
        return


def load_candles(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    if not DB_AVAILABLE:
        return
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
            SELECT ts, open, high, low, close, volume
            FROM candles
            WHERE symbol=:symbol AND timeframe=:timeframe
            ORDER BY ts ASC
            LIMIT :limit
            """),
            dict(symbol=symbol, timeframe=timeframe, limit=limit)
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"])

    return pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])

def load_recent_oi(symbol: str, limit: int = 200) -> pd.Series:
    if not DB_AVAILABLE:
        return
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
            SELECT ts, open_interest
            FROM derivatives_metrics
            WHERE symbol=:symbol AND open_interest IS NOT NULL
            ORDER BY ts ASC
            LIMIT :limit
            """),
            dict(symbol=symbol, limit=limit)
        ).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series([float(r[1]) for r in rows])


def load_latest_score(symbol: str, timeframe: str) -> Optional[dict]:
    # When DB is down, use in-memory cache (written by insert_score)
    if not DB_AVAILABLE:
        try:
            # support both key styles (tuple and "SYMBOL:TF")
            v = SCORE_CACHE.get((symbol, timeframe))
            if v is None:
                v = SCORE_CACHE.get(f"{symbol}:{timeframe}")
            return v
        except Exception:
            return None

    with engine.begin() as conn:
        row = conn.execute(
            text("""
            SELECT ts, score_total, action, pillars, contributors
            FROM scores
            WHERE symbol=:symbol AND timeframe=:timeframe
            ORDER BY ts DESC
            LIMIT 1
            """),
            dict(symbol=symbol, timeframe=timeframe)
        ).fetchone()

    if not row:
        return None

    return {
        "ts": row[0],
        "score_total": int(row[1]),
        "action": row[2],
        "pillars": row[3],
        "contributors": row[4],
    }

def load_last_decision(symbol: str) -> Optional[dict]:
    # When DB is down, use in-memory cache
    if not DB_AVAILABLE:
        return DECISION_CACHE
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
    if not row:
        return None
    return {"ts": row[0], "decision": row[1], "confidence": int(row[2]), "details": row[3]}

# ---------------- Indicators ----------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def zscore_series(x: pd.Series, window: int = 60) -> float:
    s = x.tail(window)
    if len(s) < max(10, window // 3):
        return 0.0
    mu = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd == 0:
        return 0.0
    return float((s.iloc[-1] - mu) / sd)

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# ---------------- Macro Regime (P5) ----------------
def macro_regime_score() -> Tuple[float, Dict[str, Any]]:
    contrib: Dict[str, Any] = {}
    ethbtc = load_candles(SYMBOL_ETHBTC, "1d", limit=400)
    btcw = load_candles(SYMBOL_BTCUSDT, "1w", limit=400)
    if ethbtc is None:
        ethbtc = pd.DataFrame()
    if btcw is None:
        btcw = pd.DataFrame()

    if ethbtc is None:
        ethbtc = pd.DataFrame()
    if btcw is None:
        btcw = pd.DataFrame()

    if ethbtc is None:
        ethbtc = pd.DataFrame()
    if btcw is None:
        btcw = pd.DataFrame()


    score = 0.0

    # ETHBTC factor (max +/-10)
    if len(ethbtc) >= 60:
        c = ethbtc["close"]
        e50 = ema(c, 50)
        slope = float(e50.iloc[-1] - e50.iloc[-6]) if len(e50) >= 6 else 0.0
        above = float(c.iloc[-1] - e50.iloc[-1])
        ethbtc_factor = (1.0 if above > 0 else -1.0) + (1.0 if slope > 0 else -1.0)
        ethbtc_score = (ethbtc_factor / 2.0) * 10.0
        score += ethbtc_score
        contrib["ethbtc_close"] = float(c.iloc[-1])
        contrib["ethbtc_ema50"] = float(e50.iloc[-1])
        contrib["ethbtc_ema50_slope"] = float(slope)
        contrib["ethbtc_score"] = float(ethbtc_score)
    else:
        contrib["ethbtc_score"] = 0.0

    # BTC weekly regime (max +/-10, fallback +/-6)
    if len(btcw) >= 220:
        c = btcw["close"]
        e200 = ema(c, 200)
        above = float(c.iloc[-1] - e200.iloc[-1])
        btc_score = 10.0 if above > 0 else -10.0
        score += btc_score
        contrib["btcw_close"] = float(c.iloc[-1])
        contrib["btcw_ema200"] = float(e200.iloc[-1])
        contrib["btcw_score"] = float(btc_score)
    elif len(btcw) >= 60:
        c = btcw["close"]
        e50 = ema(c, 50)
        above = float(c.iloc[-1] - e50.iloc[-1])
        btc_score = 6.0 if above > 0 else -6.0
        score += btc_score
        contrib["btcw_close"] = float(c.iloc[-1])
        contrib["btcw_ema50"] = float(e50.iloc[-1])
        contrib["btcw_score"] = float(btc_score)
    else:
        contrib["btcw_score"] = 0.0

    score = clamp(score, -20.0, 20.0)
    contrib["macro_regime_total"] = float(score)
    return score, contrib

# ---------------- Scoring ----------------
def score_generic(df: pd.DataFrame, funding_z: float, oi_change_z: float, p5: float) -> Tuple[int, Dict[str, float], Dict[str, Any], str]:
    close = df["close"]
    ret = close.pct_change()

    ema20, ema50 = ema(close, 20), ema(close, 50)
    ema200 = ema(close, 200) if len(close) >= 200 else ema(close, max(50, len(close)//2))
    a = atr(df, 14)
    atr_pct = (a / close).replace([np.inf, -np.inf], np.nan).fillna(0)

    trend_raw = 0.0
    trend_raw += 1.0 if close.iloc[-1] > ema50.iloc[-1] else -1.0
    trend_raw += 1.0 if ema20.iloc[-1] > ema50.iloc[-1] else -1.0
    trend_raw += 1.0 if close.iloc[-1] > ema200.iloc[-1] else -1.0
    mom = float(ret.tail(10).mean()) if len(ret) >= 10 else float(ret.mean() or 0.0)
    trend_raw += clamp(mom * 200.0, -2.0, 2.0)
    P1 = clamp(trend_raw * 5.0, -20.0, 20.0)

    lookback = min(60, len(df))
    recent_low = float(df["low"].tail(lookback).min())
    recent_high = float(df["high"].tail(lookback).max())
    price = float(close.iloc[-1])
    atr_now = float(a.iloc[-1] or 0.0)
    if atr_now <= 0:
        atr_now = max(1e-9, price * 0.01)

    room_down = (price - recent_low) / atr_now
    room_up = (recent_high - price) / atr_now
    sr_raw = clamp((room_up - room_down) / 6.0, -1.0, 1.0)
    P2 = sr_raw * 20.0

    deriv_raw = 0.0
    deriv_raw += clamp(-funding_z / 2.5, -1.0, 1.0)
    deriv_raw += clamp(oi_change_z / 3.0, -1.0, 1.0)
    P3 = clamp(deriv_raw * 20.0, -20.0, 20.0)

    vol_regime = float(atr_pct.tail(60).mean()) if len(atr_pct) >= 20 else float(atr_pct.mean() or 0.0)
    vol_score = clamp((0.015 - vol_regime) / 0.01, -1.0, 1.0)
    P4 = vol_score * 20.0

    P5 = float(p5)

    total = int(round(P1 + P2 + P3 + P4 + P5))
    total = int(clamp(total, -100, 100))

    if total >= 30:
        action = "BUY"
    elif total <= -30:
        action = "SELL"
    else:
        action = "HOLD"

    pillars = {
        "trend_momentum": float(P1),
        "support_resistance": float(P2),
        "derivatives": float(P3),
        "vol_risk": float(P4),
        "macro_regime": float(P5),
    }

    contributors = {
        "close": float(price),
        "funding_z": float(funding_z),
        "oi_change_z": float(oi_change_z),
        "room_up_atr": float(room_up),
        "room_down_atr": float(room_down),
        "atr_pct_mean": float(vol_regime),
        "price_vs_ema50": 1 if price > float(ema50.iloc[-1]) else -1,
        "ema20_vs_ema50": 1 if float(ema20.iloc[-1]) > float(ema50.iloc[-1]) else -1,
    }

    return total, pillars, contributors, action

# ---------------- Decision + Plan (configurable) ----------------
def decision_from_scores(s4h: int, s1d: int, s1w: int) -> Tuple[str, int, dict]:
    W = W_4H * s4h + W_1D * s1d + W_1W * s1w

    align = 1.0 if (np.sign(s4h) == np.sign(s1d) == np.sign(s1w)) else 0.0
    spread = float(max(s4h, s1d, s1w) - min(s4h, s1d, s1w))
    conf = int(clamp((abs(W) / 60.0) * 70.0 + align * 20.0 + (1.0 - min(spread/120.0, 1.0)) * 10.0, 0.0, 100.0))

    if W >= float(ACCUMULATE_THRESHOLD):
        dec = "ACCUMULATE"
    elif W <= float(EXIT_THRESHOLD):
        dec = "EXIT"
    elif W <= float(REDUCE_THRESHOLD):
        dec = "REDUCE"
    else:
        dec = "HOLD"

    return dec, conf, {"scores": {"4h": s4h, "1d": s1d, "1w": s1w}, "weighted_score": float(W), "raw_decision": dec, "confidence": conf}

def apply_hysteresis(raw_decision: str, last_decision: Optional[str], weighted_score: float) -> str:
    if not last_decision:
        return raw_decision

    ladder = ["EXIT", "REDUCE", "HOLD", "ACCUMULATE"]
    if raw_decision not in ladder or last_decision not in ladder:
        return raw_decision

    # Don't stay stuck in EXIT unless still below EXIT threshold
    if last_decision == "EXIT" and weighted_score > float(EXIT_THRESHOLD):
        last_decision = "REDUCE"

    raw_i = ladder.index(raw_decision)
    last_i = ladder.index(last_decision)

    # One-step movement per run
    if raw_i > last_i + 1:
        return ladder[last_i + 1]
    if raw_i < last_i - 1:
        return ladder[last_i - 1]
    return raw_decision

def portfolio_plan_steps(final_decision: str, confidence: int, weighted_score: float) -> dict:
    """
    Returns a step in percent and a hint.
    Step sizing is configurable via .env.
    """
    step = 0
    hint = "hold"

    if final_decision == "ACCUMULATE":
        step = ACC_STEP_BIG if confidence >= ACC_BIG_CONF else ACC_STEP_SMALL
        hint = f"buy/add ~{step}% (scale-in)"
    elif final_decision == "HOLD":
        step = 0
        hint = "do nothing"
    elif final_decision == "REDUCE":
        aw = abs(weighted_score)
        if aw >= float(REDUCE_BIG_AT):
            step = -abs(RED_STEP_BIG)
        elif aw >= float(REDUCE_MED_AT):
            step = -abs(RED_STEP_MED)
        else:
            step = -abs(RED_STEP_SMALL)
        hint = f"sell/reduce ~{abs(step)}% (de-risk)"
    elif final_decision == "EXIT":
        step = -100
        hint = "exit full position"

    return {"step_percent": int(step), "hint": hint}

def compute_and_store_decision() -> dict:
    s4 = load_latest_score(SYMBOL_ETHUSDT, "4h")
    s1d = load_latest_score(SYMBOL_ETHUSDT, "1d")
    s1w = load_latest_score(SYMBOL_ETHUSDT, "1w")
    if not s4 or not s1d or not s1w:
        raise RuntimeError("Missing scores. Run /admin/run_all first.")

    raw_dec, conf, details = decision_from_scores(s4["score_total"], s1d["score_total"], s1w["score_total"])
    last = load_last_decision(SYMBOL_ETHUSDT)
    last_dec = last["decision"] if last else None
    final_dec = apply_hysteresis(raw_dec, last_dec, details["weighted_score"])

    plan = portfolio_plan_steps(final_dec, conf, details["weighted_score"])

    ts = datetime.now(timezone.utc)
    details["last_decision"] = last_dec
    details["final_decision"] = final_dec
    details["plan"] = plan
    details["config"] = {
        "ACCUMULATE_THRESHOLD": ACCUMULATE_THRESHOLD,
        "REDUCE_THRESHOLD": REDUCE_THRESHOLD,
        "EXIT_THRESHOLD": EXIT_THRESHOLD,
        "weights": {"4h": W_4H, "1d": W_1D, "1w": W_1W},
        "MIN_EXPOSURE": MIN_EXPOSURE,
        "steps": {
            "ACC_SMALL": ACC_STEP_SMALL, "ACC_BIG": ACC_STEP_BIG, "ACC_BIG_CONF": ACC_BIG_CONF,
            "RED_SMALL": RED_STEP_SMALL, "RED_MED": RED_STEP_MED, "RED_BIG": RED_STEP_BIG,
            "REDUCE_MED_AT": REDUCE_MED_AT, "REDUCE_BIG_AT": REDUCE_BIG_AT,
        }
    }

    insert_decision(SYMBOL_ETHUSDT, ts, final_dec, conf, details)

    return {"symbol": SYMBOL_ETHUSDT, "ts": ts.isoformat(), "decision": final_dec, "confidence": conf, "details": details}

# ---------------- Orchestration ----------------
def ingest_all_candles():
    for tf in ["4h", "1d", "1w"]:
        df = fetch_klines(SYMBOL_ETHUSDT, tf, limit=500)
        upsert_candles(SYMBOL_ETHUSDT, tf, df)

    df = fetch_klines(SYMBOL_ETHBTC, "1d", limit=500)
    upsert_candles(SYMBOL_ETHBTC, "1d", df)

    df = fetch_klines(SYMBOL_BTCUSDT, "1w", limit=500)
    upsert_candles(SYMBOL_BTCUSDT, "1w", df)

def ingest_derivatives_snapshot():
    funding_df = fetch_funding_rate(SYMBOL_FUT, limit=100)
    last_funding = None
    if not funding_df.empty:
        last_funding = float(funding_df["fundingRate"].iloc[-1])
    oi = fetch_open_interest(SYMBOL_FUT)
    now = datetime.now(timezone.utc)
    upsert_derivatives(SYMBOL_FUT, now, oi, last_funding)

def compute_score_for(tf: Timeframe, p5: float, macro_contrib: dict):
    # Prefer DB cache, but if DB is down / empty, fall back to live Binance klines.
    df = load_candles(SYMBOL_ETHUSDT, tf, limit=500)

    def _ensure_df(x):
        return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

    df = _ensure_df(df)

    if len(df) < 60:
        df = fetch_klines(SYMBOL_ETHUSDT, tf, limit=500)
        df = _ensure_df(df)

    # Extra fallback: build 4h candles from 1h if needed
    if tf == "4h" and len(df) < 60:
        df1h = fetch_klines(SYMBOL_ETHUSDT, "1h", limit=1000)
        df1h = _ensure_df(df1h)
        if not df1h.empty and "ts" in df1h.columns:
            df1h = df1h.sort_values("ts").set_index("ts")
            agg = df1h.resample("4H").agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            ).dropna()
            df = agg.reset_index()

    if len(df) < 60:
        raise RuntimeError("Not enough ETHUSDT candles for timeframe: %s" % tf)

    funding_df = fetch_funding_rate(SYMBOL_FUT, limit=100)
    funding_z = 0.0
    if not funding_df.empty:
        funding_z = zscore_series(funding_df["fundingRate"], window=min(60, len(funding_df)))

    oi_series = load_recent_oi(SYMBOL_FUT, limit=200)
    if oi_series is None:
        oi_series = pd.Series(dtype=float)

    oi_change_z = 0.0
    if len(oi_series) >= 20:
        oi_change = oi_series.pct_change().fillna(0.0)
        oi_change_z = zscore_series(oi_change, window=min(120, len(oi_change)))

    total, pillars, contrib, action = score_generic(df, funding_z=funding_z, oi_change_z=oi_change_z, p5=p5)
    contrib.update(macro_contrib)

    ts = df["ts"].iloc[-1].to_pydatetime()
    insert_score(SYMBOL_ETHUSDT, tf, ts, total, pillars, contrib, action)

def run_all():
    ingest_all_candles()
    ingest_derivatives_snapshot()

    p5, macro_contrib = macro_regime_score()
    for tf in ["4h", "1d", "1w"]:
        compute_score_for(tf, p5=p5, macro_contrib=macro_contrib)

    payload = None
    try:
        payload = compute_and_store_decision()
    except Exception as e:
        print("decision compute failed:", repr(e))
        payload = {"error": "decision_compute_failed", "exception": repr(e)}

    # Always update logs/recommendation.json so frontend has fresh data even if DB is down
    try:
        write_recommendation_json(payload)
    except Exception as e:
        print("WARN: write_recommendation_json failed:", repr(e))

@app.on_event("startup")
def startup():
    # DB is optional in local/dev; do not crash if Postgres is down
    try:
        set_db_available(True)
        db_init()
    except Exception as e:
        set_db_available(False)
        print("WARN: DB init failed, continuing without DB:", repr(e))


    try:
        run_all()
    except Exception as e:
        print("startup run_all failed:", repr(e))

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_all,
        "interval",
        minutes=15,
        id="run_all",
        replace_existing=True
    )
    scheduler.start()
    app.state.scheduler = scheduler


@app.get("/health")
def health():
    return {
        "status": "ok", "build_id": BUILD_ID,
        "config": {
            "ACCUMULATE_THRESHOLD": ACCUMULATE_THRESHOLD,
            "REDUCE_THRESHOLD": REDUCE_THRESHOLD,
            "EXIT_THRESHOLD": EXIT_THRESHOLD,
            "weights": {"4h": W_4H, "1d": W_1D, "1w": W_1W},
            "MIN_EXPOSURE": MIN_EXPOSURE,
        }
    }


@app.get("/admin/debug_cache")
def admin_debug_cache():
    try:
        keys = sorted([f"{k[0]}:{k[1]}" for k in SCORE_CACHE.keys()])
    except Exception:
        keys = []
    return {
        "build_id": BUILD_ID if "BUILD_ID" in globals() else None,
        "score_cache_size": len(SCORE_CACHE) if "SCORE_CACHE" in globals() else None,
        "score_cache_keys": keys[:20],
        "decision_cache": DECISION_CACHE if "DECISION_CACHE" in globals() else None,
    }

@app.post("/admin/run_all")
def admin_run_all():
    try:
        run_all()
        return {"ok": True}
    except Exception:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=tb)

@app.get("/decision/latest")
def decision_latest():
    with engine.begin() as conn:
        row = conn.execute(
            text("""
            SELECT ts, decision, confidence, details
            FROM decisions
            WHERE symbol=:symbol
            ORDER BY ts DESC
            LIMIT 1
            """),
            dict(symbol=SYMBOL_ETHUSDT)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="No decision yet. Try POST /admin/run_all")

    return {"symbol": SYMBOL_ETHUSDT, "ts": row[0], "decision": row[1], "confidence": row[2], "details": row[3]}

@app.get("/portfolio/plan")
def portfolio_plan(
    current_exposure: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Your current ETH exposure (0..1). If provided, we compute target_exposure applying MIN_EXPOSURE unless EXIT.")
):
    """
    Returns the latest decision + step recommendation.
    If current_exposure is provided, returns target_exposure and an adjusted step
    that respects MIN_EXPOSURE (except EXIT).
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
            dict(symbol=SYMBOL_ETHUSDT)
        ).fetchone()
        if row:
            d = {"ts": row[0], "decision": row[1], "confidence": int(row[2]), "details": row[3]}

    if not d:
        raise HTTPException(status_code=404, detail="No decision yet. Try POST /admin/run_all")

    plan = d["details"].get("plan", None)
    if not plan:
        raise HTTPException(status_code=500, detail="No plan in details (unexpected). Run POST /admin/run_all again.")

    weighted_score = d["details"].get("weighted_score")
    scores = d["details"].get("scores")

    step_percent = int(plan.get("step_percent"))
    hint = plan.get("hint")

    target_exposure = None
    adjusted_step_percent = None

    if current_exposure is not None:
        # apply step to exposure
        raw_target = clamp(float(current_exposure) + (step_percent / 100.0), 0.0, 1.0)
        if d["decision"] != "EXIT":
            raw_target = max(raw_target, float(MIN_EXPOSURE))
        target_exposure = float(raw_target)
        adjusted_step_percent = int(round((target_exposure - float(current_exposure)) * 100.0))

    return {
        "symbol": SYMBOL_ETHUSDT,
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
        "min_exposure": MIN_EXPOSURE,
    }

app.include_router(ai_router)

app.include_router(portfolio_ai_router)

# ---------------- Public API: latest recommendation ----------------
@app.get("/recommendation/latest")
def recommendation_latest():
    """
    Return latest trading recommendation.
    Uses DB when available; otherwise uses in-memory cache.
    """
    last = load_last_decision(SYMBOL_ETHUSDT)
    if not last:
        raise HTTPException(status_code=404, detail="No decision yet. Run /admin/run_all first.")

    ts = last.get("ts")
    return {
        "symbol": SYMBOL_ETHUSDT,
        "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
        "decision": last.get("decision"),
        "confidence": int(last.get("confidence", 0)),
        "details": last.get("details") or {},
    }
