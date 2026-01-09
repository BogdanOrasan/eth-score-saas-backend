import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# -------------------- ENV CONFIG (shared with backend .env) --------------------
import os

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

ACCUMULATE_THRESHOLD = _get_int("ACCUMULATE_THRESHOLD", 20)
REDUCE_THRESHOLD     = _get_int("REDUCE_THRESHOLD", -20)
EXIT_THRESHOLD       = _get_int("EXIT_THRESHOLD", -45)

W_4H = _get_float("W_4H", 0.25)
W_1D = _get_float("W_1D", 0.35)
W_1W = _get_float("W_1W", 0.40)

MIN_EXPOSURE = _get_float("MIN_EXPOSURE", 0.30)

ACC_STEP_SMALL = _get_int("ACCUMULATE_STEP_SMALL", 10)
ACC_STEP_BIG   = _get_int("ACCUMULATE_STEP_BIG", 25)
ACC_BIG_CONF   = _get_int("ACCUMULATE_BIG_CONF", 70)

RED_STEP_SMALL = _get_int("REDUCE_STEP_SMALL", 10)
RED_STEP_MED   = _get_int("REDUCE_STEP_MED", 25)
RED_STEP_BIG   = _get_int("REDUCE_STEP_BIG", 50)

REDUCE_MED_AT  = _get_int("REDUCE_MED_AT", 25)
REDUCE_BIG_AT  = _get_int("REDUCE_BIG_AT", 45)
# ---------------------------------------------------------------------------

BINANCE_SPOT = "https://api.binance.com"
BINANCE_FAPI = "https://fapi.binance.com"

SYMBOL_ETHUSDT = "ETHUSDT"
SYMBOL_ETHBTC = "ETHBTC"
SYMBOL_BTCUSDT = "BTCUSDT"
SYMBOL_FUT = "ETHUSDT"

# -------------------- Utils --------------------
def utc_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def zscore_last(x: pd.Series, window: int = 60) -> float:
    s = x.tail(window)
    if len(s) < max(10, window // 3):
        return 0.0
    mu = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd == 0:
        return 0.0
    return float((s.iloc[-1] - mu) / sd)

# -------------------- Fetchers (free) --------------------
def fetch_klines_paginated(symbol: str, interval: str, start: datetime, end: datetime, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch Binance spot klines, paginated by startTime.
    Uses open_time as ts (UTC).
    """
    url = f"{BINANCE_SPOT}/api/v3/klines"
    out = []
    start_ms = utc_ms(start)
    end_ms = utc_ms(end)

    while True:
        params = {"symbol": symbol, "interval": interval, "limit": limit, "startTime": start_ms, "endTime": end_ms}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        out.extend(data)

        last_open = int(data[-1][0])
        # advance by 1ms to avoid repeating last candle
        start_ms = last_open + 1

        # throttle a little
        time.sleep(0.15)

        # stop if we got less than limit
        if len(data) < limit:
            break

    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_asset_volume","num_trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(out, columns=cols)
    if df.empty:
        return df

    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)

    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["ts","open","high","low","close","volume"]].drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df

def fetch_funding_rate_paginated(symbol: str, start: datetime, end: datetime, limit: int = 1000) -> pd.DataFrame:
    """
    Binance futures funding rate history. Free. Funding is usually every 8h.
    Endpoint: /fapi/v1/fundingRate supports startTime/endTime.
    """
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    out = []
    start_ms = utc_ms(start)
    end_ms = utc_ms(end)

    while True:
        params = {"symbol": symbol, "limit": limit, "startTime": start_ms, "endTime": end_ms}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        out.extend(data)

        last_t = int(data[-1]["fundingTime"])
        start_ms = last_t + 1

        time.sleep(0.15)

        if len(data) < limit:
            break

    if not out:
        return pd.DataFrame(columns=["fundingTime","fundingRate"])

    df = pd.DataFrame(out)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df = df.sort_values("fundingTime").reset_index(drop=True)
    return df[["fundingTime","fundingRate"]]

def try_fetch_open_interest_hist(symbol: str, period: str, start: datetime, end: datetime, limit: int = 500) -> Optional[pd.DataFrame]:
    """
    Tries Binance data endpoint for open interest history (may work without key, may not).
    If it fails, returns None.
    Endpoint: /futures/data/openInterestHist (UM futures)
    """
    url = f"{BINANCE_FAPI}/futures/data/openInterestHist"
    try:
        r = requests.get(url, params={
            "symbol": symbol,
            "period": period,
            "limit": limit,
            "startTime": utc_ms(start),
            "endTime": utc_ms(end),
        }, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        # fields: sumOpenInterest, sumOpenInterestValue, timestamp...
        if "timestamp" not in df.columns or "sumOpenInterest" not in df.columns:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df[["timestamp","sumOpenInterest"]]
    except Exception:
        return None

# -------------------- Scoring (same spirit as your API) --------------------
def macro_regime_score(ethbtc_1d: pd.DataFrame, btcusdt_1w: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    """
    P5 in [-20, +20]
    - ETHBTC 1D: close > EMA50 and EMA50 slope up => +10, else -10
    - BTCUSDT 1W: close > EMA200 => +10 else -10 (fallback EMA50 => +/-6)
    """
    contrib: Dict[str, Any] = {}
    score = 0.0

    # ETHBTC 1D
    if len(ethbtc_1d) >= 60:
        c = ethbtc_1d["close"]
        e50 = ema(c, 50)
        slope = float(e50.iloc[-1] - e50.iloc[-6]) if len(e50) >= 6 else 0.0
        above = float(c.iloc[-1] - e50.iloc[-1])
        ethbtc_factor = (1.0 if above > 0 else -1.0) + (1.0 if slope > 0 else -1.0)
        ethbtc_score = (ethbtc_factor / 2.0) * 10.0
        score += ethbtc_score
        contrib["ethbtc_score"] = float(ethbtc_score)
    else:
        contrib["ethbtc_score"] = 0.0

    # BTC weekly
    if len(btcusdt_1w) >= 220:
        c = btcusdt_1w["close"]
        e200 = ema(c, 200)
        above = float(c.iloc[-1] - e200.iloc[-1])
        btc_score = 10.0 if above > 0 else -10.0
        score += btc_score
        contrib["btcw_score"] = float(btc_score)
    elif len(btcusdt_1w) >= 60:
        c = btcusdt_1w["close"]
        e50 = ema(c, 50)
        above = float(c.iloc[-1] - e50.iloc[-1])
        btc_score = 6.0 if above > 0 else -6.0
        score += btc_score
        contrib["btcw_score"] = float(btc_score)
    else:
        contrib["btcw_score"] = 0.0

    score = clamp(score, -20.0, 20.0)
    contrib["macro_regime_total"] = float(score)
    return score, contrib

def score_generic(df: pd.DataFrame, funding_z: float, oi_change_z: float, p5: float) -> int:
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
    return total

# -------------------- Decision engine (Option 1) --------------------
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
    # same style as in your server: not sticky in EXIT unless truly bad
    if not last_decision:
        return raw_decision

    ladder = ["EXIT", "REDUCE", "HOLD", "ACCUMULATE"]
    if raw_decision not in ladder or last_decision not in ladder:
        return raw_decision

    if last_decision == "EXIT" and weighted_score > float(EXIT_THRESHOLD):
        last_decision = "REDUCE"

    raw_i = ladder.index(raw_decision)
    last_i = ladder.index(last_decision)

    if raw_i > last_i + 1:
        return ladder[last_i + 1]
    if raw_i < last_i - 1:
        return ladder[last_i - 1]
    return raw_decision

def portfolio_plan_steps(final_decision: str, confidence: int, weighted_score: float) -> Tuple[int, str]:
    # Uses .env step sizes + cutoffs
    if final_decision == "ACCUMULATE":
        step = ACC_STEP_BIG if confidence >= ACC_BIG_CONF else ACC_STEP_SMALL
        return step, f"buy/add ~{step}%"
    if final_decision == "HOLD":
        return 0, "do nothing"
    if final_decision == "REDUCE":
        aw = abs(weighted_score)
        if aw >= float(REDUCE_BIG_AT):
            step = -abs(RED_STEP_BIG)
        elif aw >= float(REDUCE_MED_AT):
            step = -abs(RED_STEP_MED)
        else:
            step = -abs(RED_STEP_SMALL)
        return step, f"sell/reduce ~{abs(step)}%"
    return -100, "exit full position"

# -------------------- Backtest --------------------
@dataclass
class BtConfig:
    days_4h: int
    years_1d: int
    years_1w: int
    fee_bps: float  # cost per exposure change (bps of notional), ex 5 bps = 0.05%
    start_exposure: float  # 0..1

def nearest_past_index(ts: pd.Timestamp, index: pd.DatetimeIndex) -> int:
    # returns position of last index <= ts, or -1
    pos = index.searchsorted(ts, side="right") - 1
    return int(pos)

def run_backtest(cfg: BtConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    now = datetime.now(timezone.utc)

    start_4h = now - timedelta(days=cfg.days_4h)
    start_1d = now - timedelta(days=cfg.years_1d * 365)
    start_1w = now - timedelta(days=cfg.years_1w * 365)

    print("Downloading candles (free Binance)…")
    eth_4h = fetch_klines_paginated(SYMBOL_ETHUSDT, "4h", start_4h, now)
    eth_1d = fetch_klines_paginated(SYMBOL_ETHUSDT, "1d", start_1d, now)
    eth_1w = fetch_klines_paginated(SYMBOL_ETHUSDT, "1w", start_1w, now)

    ethbtc_1d = fetch_klines_paginated(SYMBOL_ETHBTC, "1d", start_1d, now)
    btcusdt_1w = fetch_klines_paginated(SYMBOL_BTCUSDT, "1w", start_1w, now)

    if eth_4h.empty or eth_1d.empty or eth_1w.empty:
        raise RuntimeError("Missing candles. Binance returned empty data.")

    print(f"ETH 4h: {len(eth_4h)} rows | ETH 1d: {len(eth_1d)} | ETH 1w: {len(eth_1w)}")
    print("Downloading funding history…")
    funding = fetch_funding_rate_paginated(SYMBOL_FUT, start_4h, now)

    # Try OI history (optional)
    oi_hist = try_fetch_open_interest_hist(SYMBOL_FUT, "4h", start_4h, now)
    if oi_hist is None:
        print("Open interest history not available (OK). Using oi_change_z = 0.")
    else:
        print(f"Open interest hist: {len(oi_hist)} rows")

    # Build indices
    eth_4h = eth_4h.set_index("ts")
    eth_1d = eth_1d.set_index("ts")
    eth_1w = eth_1w.set_index("ts")
    ethbtc_1d = ethbtc_1d.set_index("ts")
    btcusdt_1w = btcusdt_1w.set_index("ts")
    funding = funding.set_index("fundingTime")

    # Precompute OI series (optional)
    oi_series = None
    if oi_hist is not None and not oi_hist.empty:
        oi_series = oi_hist.set_index("timestamp")["sumOpenInterest"].sort_index()

    # Backtest loop over 4h bars
    rows: List[Dict[str, Any]] = []

    equity = 1.0
    exposure = cfg.start_exposure  # 0..1
    last_decision: Optional[str] = None

    fee_rate = cfg.fee_bps / 10000.0

    # For returns
    eth_4h_close = eth_4h["close"]
    rets_4h = eth_4h_close.pct_change().fillna(0.0)

    idx_4h = eth_4h.index
    idx_1d = eth_1d.index
    idx_1w = eth_1w.index
    idx_ethbtc = ethbtc_1d.index
    idx_btcw = btcusdt_1w.index
    idx_fund = funding.index

    print("Running simulation…")

    for i, ts in enumerate(idx_4h):
        if i == 0:
            rows.append({
                "ts": ts, "equity": equity, "exposure": exposure,
                "decision": None, "step_percent": 0,
                "s4h": None, "s1d": None, "s1w": None,
                "weighted_score": None, "confidence": None,
                "funding_z": None, "oi_change_z": None, "macro_p5": None,
            })
            continue

        # Compute funding_z as zscore of last 60 funding points up to ts
        fi = nearest_past_index(ts, idx_fund)
        if fi >= 0:
            f_slice = funding.iloc[:fi+1]["fundingRate"]
            funding_z = zscore_last(f_slice, window=min(60, len(f_slice)))
        else:
            funding_z = 0.0

        # Compute oi_change_z (optional)
        oi_change_z = 0.0
        if oi_series is not None:
            # take up to ts
            oi_up = oi_series.loc[:ts]
            if len(oi_up) >= 20:
                oi_change = oi_up.pct_change().fillna(0.0)
                oi_change_z = zscore_last(oi_change, window=min(120, len(oi_change)))

        # Macro P5: compute using latest available ETHBTC 1D and BTCUSDT 1W up to ts
        ei = nearest_past_index(ts, idx_ethbtc)
        bi = nearest_past_index(ts, idx_btcw)
        p5 = 0.0
        if ei >= 0 and bi >= 0:
            p5, _ = macro_regime_score(ethbtc_1d.iloc[:ei+1].reset_index(), btcusdt_1w.iloc[:bi+1].reset_index())

        # Scores per TF computed at their latest candle <= ts
        i4 = nearest_past_index(ts, idx_4h)
        i1d = nearest_past_index(ts, idx_1d)
        i1w = nearest_past_index(ts, idx_1w)

        # Need enough history
        def safe_score(df: pd.DataFrame, upto_i: int) -> Optional[int]:
            if upto_i < 70:  # needs some history for EMA/ATR
                return None
            window_df = df.iloc[:upto_i+1].reset_index()
            return score_generic(window_df, funding_z=funding_z, oi_change_z=oi_change_z, p5=p5)

        s4h = safe_score(eth_4h, i4) if i4 >= 0 else None
        s1d = safe_score(eth_1d, i1d) if i1d >= 0 else None
        s1w = safe_score(eth_1w, i1w) if i1w >= 0 else None

        decision = None
        step_percent = 0
        W = None
        conf = None

        if s4h is not None and s1d is not None and s1w is not None:
            raw, conf, det = decision_from_scores(s4h, s1d, s1w)
            W = float(det["weighted_score"])
            final = apply_hysteresis(raw, last_decision, W)
            step_percent, _hint = portfolio_plan_steps(final, conf, W)
            decision = final

            # apply exposure step
            prev_exposure = exposure
            exposure = clamp(exposure + (step_percent / 100.0), 0.0, 1.0)

            # MIN_EXPOSURE floor (only if not EXIT)
            # MIN_EXPOSURE is loaded from .env (global MIN_EXPOSURE)
            if decision != "EXIT":
                exposure = max(exposure, float(MIN_EXPOSURE))

            # transaction cost on notional changed
            delta = abs(exposure - prev_exposure)
            if delta > 0:
                equity *= (1.0 - fee_rate * delta)

            last_decision = decision

        # Apply PnL for this bar with current exposure
        r = float(rets_4h.loc[ts])
        equity *= (1.0 + exposure * r)

        rows.append({
            "ts": ts, "equity": equity, "exposure": exposure,
            "decision": decision, "step_percent": step_percent,
            "s4h": s4h, "s1d": s1d, "s1w": s1w,
            "weighted_score": W, "confidence": conf,
            "funding_z": funding_z, "oi_change_z": oi_change_z, "macro_p5": p5,
        })

    bt = pd.DataFrame(rows).set_index("ts")

    # Metrics
    bt["peak"] = bt["equity"].cummax()
    bt["dd"] = bt["equity"] / bt["peak"] - 1.0
    max_dd = float(bt["dd"].min())
    total_ret = float(bt["equity"].iloc[-1] - 1.0)
    trades = int((bt["step_percent"].fillna(0) != 0).sum())
    exits = int((bt["decision"] == "EXIT").sum())
    time_in_mkt = float((bt["exposure"] > 0).mean())

    metrics = {
        "total_return_pct": total_ret * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "num_rebalance_events": trades,
        "num_exit_bars": exits,
        "time_in_market_pct": time_in_mkt * 100.0,
        "final_equity": float(bt["equity"].iloc[-1]),
        "rows": int(len(bt)),
    }
    return bt, metrics

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days4h", type=int, default=365, help="How many days of 4h data to backtest (default 365)")
    ap.add_argument("--years1d", type=int, default=3, help="How many years of 1d history to fetch (default 3)")
    ap.add_argument("--years1w", type=int, default=8, help="How many years of 1w history to fetch (default 8)")
    ap.add_argument("--fee_bps", type=float, default=5.0, help="Fee in bps per 100% notional change (default 5 = 0.05%)")
    ap.add_argument("--start_exposure", type=float, default=1.0, help="Start exposure 0..1 (default 1.0)")
    ap.add_argument("--out", type=str, default="backtest_results.csv", help="CSV output file")
    args = ap.parse_args()

    cfg = BtConfig(
        days_4h=args.days4h,
        years_1d=args.years1d,
        years_1w=args.years1w,
        fee_bps=args.fee_bps,
        start_exposure=float(args.start_exposure),
    )

    bt, metrics = run_backtest(cfg)

    bt.to_csv(args.out)
    print("\n=== METRICS ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()
