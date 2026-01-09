import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ccxt

# =========================
# Config (keep engine params fixed)
# =========================
SYMBOL = "ETH/USDT"
BASE_TF = "1h"
LOOKBACK_LIMIT = 2000

WEIGHTS = {"4h": 0.25, "1d": 0.35, "1w": 0.40}
ACCUMULATE_THRESHOLD = 20
REDUCE_THRESHOLD = -20
EXIT_THRESHOLD = -45
MIN_EXPOSURE = 0.30

STEP = 0.05
MAX_EXPOSURE = 1.00
GRID_SPACING_PCT = 0.75
GRID_LEVELS = 6
START_EXPOSURE = 0.40

EMA_PERIODS = [10, 20, 34, 50, 89, 100, 144, 200]

SCORE_CLAMP_PCT = 5.0

OUT_CSV = "logs/backtest_ema_grid_sweep.csv"


@dataclass
class Metrics:
    total_return_pct: float
    max_drawdown_pct: float
    adjustments: int


def fetch_ohlcv(symbol=SYMBOL, timeframe=BASE_TF, limit=LOOKBACK_LIMIT) -> pd.DataFrame:
    ex = ccxt.kraken()
    ex.load_markets()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts").sort_index()


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = {"4h": "4H", "1d": "1D", "1w": "1W"}[tf]
    return pd.DataFrame({
        "open": df["open"].resample(rule).first(),
        "high": df["high"].resample(rule).max(),
        "low": df["low"].resample(rule).min(),
        "close": df["close"].resample(rule).last(),
        "volume": df["volume"].resample(rule).sum(),
    }).dropna()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def ema_score(close: pd.Series, ema_line: pd.Series) -> pd.Series:
    dist_pct = (close - ema_line) / ema_line * 100.0
    score = (dist_pct / SCORE_CLAMP_PCT) * 100.0
    return score.clip(-100, 100)


def decision_from_weighted_score(ws: float) -> str:
    if ws <= EXIT_THRESHOLD:
        return "EXIT"
    if ws <= REDUCE_THRESHOLD:
        return "REDUCE"
    if ws >= ACCUMULATE_THRESHOLD:
        return "ACCUMULATE"
    return "HOLD"


def grid_adjust_exposure(exposure: float, decision: str, dist_pct: float):
    if decision == "EXIT":
        return max(MIN_EXPOSURE, 0.0), int(exposure != MIN_EXPOSURE)
    if decision == "HOLD":
        return exposure, 0

    levels = [(i + 1) * GRID_SPACING_PCT for i in range(GRID_LEVELS)]
    adj = 0

    if decision == "ACCUMULATE":
        d = -dist_pct
        if d <= 0:
            return exposure, 0
        k = sum(1 for lv in levels if d >= lv)
        new_exp = min(MAX_EXPOSURE, exposure + k * STEP)
        return new_exp, int(new_exp != exposure)

    if decision == "REDUCE":
        d = dist_pct
        if d <= 0:
            return exposure, 0
        k = sum(1 for lv in levels if d >= lv)
        new_exp = max(MIN_EXPOSURE, exposure - k * STEP)
        return new_exp, int(new_exp != exposure)

    return exposure, 0


def backtest(df_base: pd.DataFrame, ema_map: Dict[str, int]) -> Metrics:
    tfs = {tf: resample(df_base, tf) for tf in ["4h", "1d", "1w"]}
    scores, dists = {}, {}

    for tf, d in tfs.items():
        e = ema(d["close"], ema_map[tf])
        dists[tf] = ((d["close"] - e) / e * 100).reindex(df_base.index, method="ffill")
        scores[tf] = ema_score(d["close"], e).reindex(df_base.index, method="ffill")

    ws = (
        scores["4h"] * WEIGHTS["4h"]
        + scores["1d"] * WEIGHTS["1d"]
        + scores["1w"] * WEIGHTS["1w"]
    )

    decisions = ws.apply(decision_from_weighted_score)
    rets = df_base["close"].pct_change().fillna(0)

    exposure = START_EXPOSURE
    equity, peak, max_dd, adj = 1.0, 1.0, 0.0, 0

    ref_dist = dists["1d"].fillna(0)

    for i in range(1, len(df_base)):
        equity *= (1 + exposure * rets.iat[i])
        peak = max(peak, equity)
        max_dd = min(max_dd, (equity / peak - 1) * 100)

        exposure, a = grid_adjust_exposure(exposure, decisions.iat[i], ref_dist.iat[i])
        adj += a

    return Metrics((equity - 1) * 100, abs(max_dd), adj)


def sweep():
    df = fetch_ohlcv()
    rows = []

    for p in EMA_PERIODS:
        m = backtest(df, {"4h": p, "1d": p, "1w": p})
        rows.append(("ALL", p, p, p, m.total_return_pct, m.max_drawdown_pct, m.adjustments))

    base = 50
    for tf in ["4h", "1d", "1w"]:
        for p in EMA_PERIODS:
            mp = {"4h": base, "1d": base, "1w": base}
            mp[tf] = p
            m = backtest(df, mp)
            rows.append((tf, mp["4h"], mp["1d"], mp["1w"], m.total_return_pct, m.max_drawdown_pct, m.adjustments))

    df_out = pd.DataFrame(rows, columns=[
        "sweep_mode", "ema_4h", "ema_1d", "ema_1w",
        "total_return_pct", "max_drawdown_pct", "adjustments"
    ])

    df_out = df_out.sort_values(
        ["total_return_pct", "max_drawdown_pct"],
        ascending=[False, True]
    )

    df_out.to_csv(OUT_CSV, index=False)

    print(f"\nSaved CSV -> {OUT_CSV}\n")
    print(df_out.head(30).to_string(index=False))


if __name__ == "__main__":
    sweep()
