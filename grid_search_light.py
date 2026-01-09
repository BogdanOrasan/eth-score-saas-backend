import argparse
import itertools
import os
import pandas as pd

import backtest_full as bt  # uses .env defaults, but we'll override globals

def set_bt_params(acc_th, red_th, exit_th, min_exp, w4h, w1d, w1w):
    # thresholds
    bt.ACCUMULATE_THRESHOLD = int(acc_th)
    bt.REDUCE_THRESHOLD = int(red_th)
    bt.EXIT_THRESHOLD = int(exit_th)

    # exposure floor
    bt.MIN_EXPOSURE = float(min_exp)

    # weights
    bt.W_4H = float(w4h)
    bt.W_1D = float(w1d)
    bt.W_1W = float(w1w)

def run_one(cfg, acc_th, red_th, exit_th, min_exp, w4h, w1d, w1w):
    set_bt_params(acc_th, red_th, exit_th, min_exp, w4h, w1d, w1w)

    df, metrics = bt.run_backtest(cfg)

    out = {
        "ACCUMULATE_THRESHOLD": acc_th,
        "REDUCE_THRESHOLD": red_th,
        "EXIT_THRESHOLD": exit_th,
        "MIN_EXPOSURE": min_exp,
        "W_4H": w4h, "W_1D": w1d, "W_1W": w1w,
        "total_return_pct": metrics["total_return_pct"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "time_in_market_pct": metrics["time_in_market_pct"],
        "num_rebalance_events": metrics["num_rebalance_events"],
        "num_exit_bars": metrics["num_exit_bars"],
        "final_equity": metrics["final_equity"],
        "rows": metrics["rows"],
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days4h", type=int, default=365)
    ap.add_argument("--years1d", type=int, default=3)
    ap.add_argument("--years1w", type=int, default=8)
    ap.add_argument("--fee_bps", type=float, default=5.0)
    ap.add_argument("--start_exposure", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="grid_search_results.csv")
    args = ap.parse_args()

    cfg = bt.BtConfig(
        days_4h=args.days4h,
        years_1d=args.years1d,
        years_1w=args.years1w,
        fee_bps=args.fee_bps,
        start_exposure=float(args.start_exposure),
    )

    # --- GRID (light) ---
    # Around your current best:
    # ACC: 15 / 20 / 25
    # REDUCE: -15 / -20 / -25
    # EXIT fixed: -45
    # MIN_EXPOSURE: 0.20 / 0.30
    # weights fixed (you can add variations later)
    acc_list = [15, 20, 25]
    red_list = [-15, -20, -25]
    exit_list = [-45]
    minexp_list = [0.20, 0.30]
    weights_list = [(0.25, 0.35, 0.40)]

    combos = list(itertools.product(acc_list, red_list, exit_list, minexp_list, weights_list))
    print(f"Running {len(combos)} configs...")

    results = []
    for i, (acc, red, ex, minexp, (w4, w1d, w1w)) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] ACC={acc} RED={red} EXIT={ex} MIN_EXP={minexp} W=({w4},{w1d},{w1w})")
        try:
            row = run_one(cfg, acc, red, ex, minexp, w4, w1d, w1w)
            results.append(row)
        except Exception as e:
            results.append({
                "ACCUMULATE_THRESHOLD": acc,
                "REDUCE_THRESHOLD": red,
                "EXIT_THRESHOLD": ex,
                "MIN_EXPOSURE": minexp,
                "W_4H": w4, "W_1D": w1d, "W_1W": w1w,
                "error": repr(e),
            })

    df = pd.DataFrame(results)

    # Save full results
    df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")

    # Rank: prefer higher return, penalize huge DD and too-low time-in-market
    ok = df[df.get("error").isna()].copy()

    # Simple "score" for ranking (tweakable):
    # +Return
    # -0.5*|DD|
    # +0.05*TimeInMarket
    ok["rank_score"] = ok["total_return_pct"] - 0.5*abs(ok["max_drawdown_pct"]) + 0.05*ok["time_in_market_pct"]

    top = ok.sort_values("rank_score", ascending=False).head(10)

    print("\n=== TOP 10 (rank_score = return -0.5*|DD| + 0.05*time_in_mkt) ===")
    cols = [
        "ACCUMULATE_THRESHOLD","REDUCE_THRESHOLD","EXIT_THRESHOLD","MIN_EXPOSURE",
        "total_return_pct","max_drawdown_pct","time_in_market_pct",
        "num_rebalance_events","num_exit_bars","rank_score"
    ]
    print(top[cols].to_string(index=False))

if __name__ == "__main__":
    main()
