#!/usr/bin/env python3
# stress_adversarial.py
# Create perturbed copies of hybrid_out.csv and evaluate bear-capture & equity.
import argparse, numpy as np, pandas as pd, pathlib as p
rng = np.random.default_rng(42)

def load_hybrid(csv):
    df = pd.read_csv(csv)
    df["date"] = pd.to_datetime(df["date"])
    # normalize
    for c in ["wf_label","wf2_label","hybrid_label"]:
        if c in df: df[c] = df[c].astype(str).str.title()
    return df.sort_values("date").reset_index(drop=True)

def shift_labels(df, days:int):
    out = df.copy()
    out["hybrid_label_shift"] = out["hybrid_label"].shift(days).fillna(method="bfill")
    out["hybrid_label"] = out["hybrid_label_shift"]
    return out.drop(columns=["hybrid_label_shift"])

def flip_labels(df, flip_pct:float):
    out = df.copy()
    m = rng.random(len(out)) < flip_pct
    map_flip = {"Bull":"Bear","Bear":"Bull","Choppy":"Choppy"}
    out.loc[m, "hybrid_label"] = out.loc[m, "hybrid_label"].map(map_flip).fillna("Choppy")
    return out

def drift_thresholds(df, bull_bump=0.0, bear_bump=0.0):
    # cheap “proxy”: nudge direction when drift/rsi signals are near zero (mimics threshold drift)
    out = df.copy()
    # if you saved cumret10/rsi/dd20 (from your generator), use them; otherwise just bias some days.
    if {"cumret10","rsi","dd20"}.issubset(out.columns):
        near = (out["rsi"].between(48,52)) | (out["cumret10"].between(-0.01,0.01))
        bull_bias = near & (rng.random(len(out)) < abs(bull_bump))
        bear_bias = near & (~bull_bias) & (rng.random(len(out)) < abs(bear_bump))
        out.loc[bull_bias, "hybrid_label"] = "Bull"
        out.loc[bear_bias, "hybrid_label"] = "Bear"
    return out

def severity_weighted_capture(bear_win_csv):
    # expects your *_bear_windows_summary.csv schema
    w = pd.read_csv(bear_win_csv)
    if w.empty: return float("nan")
    neg = w["btc_ret"] < 0
    denom = (-w.loc[neg,"btc_ret"]).sum()
    num   = w.loc[neg,"strat_ret"].sum()
    return float(num/denom) if denom>0 else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid_csv", default="hybrid_out.csv")
    ap.add_argument("--out_dir", default="adv_out")
    ap.add_argument("--inverse_etf_ticker", default="BITI")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--flip_pct", type=float, default=0.10)      # 10% flips
    ap.add_argument("--lag_days", type=int, default=1)           # +1 day lag
    ap.add_argument("--bull_bump", type=float, default=0.05)     # 5% bias to bull near boundary
    ap.add_argument("--bear_bump", type=float, default=0.05)     # 5% bias to bear near boundary
    ap.add_argument("--run_mc", action="store_true")
    args = ap.parse_args()

    p.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    base = load_hybrid(args.hybrid_csv)

    variants = []
    # A) label lag
    v1 = shift_labels(base, args.lag_days); variants.append(("lag", v1))
    # B) random flips
    v2 = flip_labels(base, args.flip_pct); variants.append(("flip", v2))
    # C) threshold drift proxy
    v3 = drift_thresholds(base, args.bull_bump, args.bear_bump); variants.append(("drift", v3))
    # D) combo
    v4 = drift_thresholds(flip_labels(shift_labels(base, args.lag_days), args.flip_pct),
                          args.bull_bump, args.bear_bump); variants.append(("combo", v4))

    print("Variant, CAGR, Sharpe, MaxDD, SevCap, HitRate")
    for tag, df in variants:
        out_csv = f"{args.out_dir}/hybrid_out_{tag}.csv"
        df.to_csv(out_csv, index=False)

        # reuse your existing engine
        prefix = f"{args.out_dir}/res_{tag}"
        cmd = (
            f"python3 test_bear_modes_binance_nav_mc.py "
            f"--hybrid_csv {out_csv} --start {args.start} --end {args.end} "
            f"--inverse_etf_ticker {args.inverse_etf_ticker} "
            f"--out_prefix {prefix} "
        )
        if args.run_mc:
            # include your usual MC flags here as you like; defaults OK.
            pass
        import os
        os.system(cmd)

        # read summary + bear windows to extract quick stats
        summ = pd.read_csv(f"{prefix}_summary.csv")
        def get(metric, col="median"):
            return float(summ.loc[summ["metric"]==metric, col].values[0])
        sev = severity_weighted_capture(f"{prefix}_bear_windows_summary.csv")

        # hit-rate in bears: how many windows strat_ret>0
        w = pd.read_csv(f"{prefix}_bear_windows_summary.csv")
        hits = int((w["strat_ret"]>0).sum())
        total = int(len(w))
        hitrate = 100.0*hits/total if total else float("nan")

        print(f"{tag}, {get('CAGR'):.4f}, {get('Sharpe'):.2f}, {get('MaxDD'):.4f}, {sev:.3f}, {hitrate:.1f}%")

if __name__ == "__main__":
    main()