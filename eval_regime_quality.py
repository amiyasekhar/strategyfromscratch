#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, numpy as np, pandas as pd
from scipy import stats

# ---------------------------
# Helpers
# ---------------------------
REG_ORDER = ["Bear","Choppy","Bull"]
REG_SIGN  = {"Bear":-1, "Choppy":0, "Bull":+1}

def coerce_close_or_none(df: pd.DataFrame):
    """Return numeric close Series if available, else None."""
    c = None
    for cand in ["close","Close","CLOSE","adj_close","Adj Close","AdjClose"]:
        if cand in df.columns:
            c = pd.to_numeric(df[cand], errors="coerce")
            break
    if c is not None and np.isfinite(c).sum() >= max(10, int(0.5*len(df))):
        return c
    return None

def fwd_log_return_from_close(close: pd.Series, h: int) -> pd.Series:
    return np.log(close.shift(-h) / close)

def fwd_log_return_from_logrets(logret: pd.Series, h: int) -> pd.Series:
    # sum of next h daily log returns
    return logret.shift(-1).rolling(h, min_periods=h).sum()

def encode_regime(series: pd.Series) -> pd.Series:
    return series.map(REG_SIGN).astype(float)

def regime_flip_count(lbl: pd.Series) -> int:
    v = lbl.values
    return int(np.sum(v[1:] != v[:-1]))

def contiguous_durations(lbl: pd.Series):
    if lbl.empty: return []
    out=[]; v=lbl.values; idx=lbl.index
    start=idx[0]; cur=v[0]
    for i in range(1,len(lbl)):
        if v[i] != cur:
            out.append((cur, start, idx[i-1], (idx[i-1]-start).days+1))
            start=idx[i]; cur=v[i]
    out.append((cur, start, idx[-1], (idx[-1]-start).days+1))
    return out

def cohen_d(a, b):
    a=np.asarray(a); b=np.asarray(b)
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if len(a)<3 or len(b)<3: return np.nan
    m1=a.mean(); m2=b.mean()
    s1=a.std(ddof=1); s2=b.std(ddof=1)
    # pooled
    n1=len(a); n2=len(b)
    sp=np.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2)/max(1,(n1+n2-2)))
    if sp==0 or not np.isfinite(sp): return np.nan
    return (m1-m2)/sp

def diag_backtest(ts, logret, signal, cost_bps=10.0, lag=1):
    """
    Very simple diagnostic: position = sign(signal).shift(lag).
    Transaction cost = cost_bps per |Δposition|.
    Returns dict with CAGR, Sharpe, Ann.Vol, MaxDD, total_return.
    """
    pos = np.sign(signal).shift(lag).fillna(0.0)
    gross = pos * logret
    turns = pos.diff().abs().fillna(0.0)
    costs = (cost_bps/10000.0) * turns  # bps to fraction
    net = gross - costs
    cum = net.cumsum()
    total_ret = float(np.exp(cum.iloc[-1]) - 1.0)

    # annualization (assume 252 trading days)
    ann_mu = net.mean()*252.0
    ann_sd = net.std(ddof=0)*np.sqrt(252.0)
    sharpe = (ann_mu/ann_sd) if ann_sd>0 else np.nan
    cagr = (np.exp(net.sum())**(252.0/len(net))) - 1.0

    eq = cum.copy()
    peak = eq.cummax()
    dd = eq - peak
    maxdd = float(np.exp(dd.min()) - 1.0)

    return {
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "AnnVol": float(ann_sd),
        "MaxDD": maxdd,
        "TotalReturn": total_ret
    }

# ---------------------------
# Core metrics
# ---------------------------
def regime_metrics(df: pd.DataFrame, horizon: int):
    # Prepare forward log returns, robustly
    close = coerce_close_or_none(df)
    if close is not None:
        fwd = fwd_log_return_from_close(close, horizon)
    else:
        if "LogReturn" not in df.columns:
            raise RuntimeError("Neither numeric 'close' nor 'LogReturn' present in CSV.")
        fwd = fwd_log_return_from_logrets(pd.to_numeric(df["LogReturn"], errors="coerce"), horizon)

    # Encode label -> signal
    lab = df["Label_rt"].astype(str)
    sig = encode_regime(lab)

    # Hit-rate: sign match between signal and realized fwd return
    sign_match = np.sign(sig) == np.sign(fwd)
    hit_rate = float(sign_match.replace({False:0, True:1}).mean())

    # Information coefficients
    ic_p = float(pd.Series(sig).corr(fwd, method="pearson"))
    ic_s = float(pd.Series(sig).corr(fwd, method="spearman"))

    # Forward mean by label
    means_by_lbl = {}
    for k in REG_ORDER:
        means_by_lbl[k] = float(fwd[lab==k].mean())

    # Cohen's d between regime forward-return distributions
    d_bull_bear   = cohen_d(fwd[lab=="Bull"],   fwd[lab=="Bear"])
    d_bull_choppy = cohen_d(fwd[lab=="Bull"],   fwd[lab=="Choppy"])
    d_choppy_bear = cohen_d(fwd[lab=="Choppy"], fwd[lab=="Bear"])

    return {
        "hit_rate": hit_rate,
        "ic_pearson": ic_p,
        "ic_spearman": ic_s,
        "means_by_lbl": means_by_lbl,
        "cohen_d": {
            "Bull-Bear": float(d_bull_bear),
            "Bull-Choppy": float(d_bull_choppy),
            "Choppy-Bear": float(d_choppy_bear),
        }
    }

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate regime quality (robust to stringy closes).")
    ap.add_argument("--csv", required=True, help="Path to CSV produced by your regime script")
    ap.add_argument("--backtest", type=int, default=0, help="1 to run simple diagnostic backtest")
    ap.add_argument("--cost_bps", type=float, default=10.0, help="per |Δposition| cost in bps (diag backtest)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # parse timestamp
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    else:
        # try index
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.dropna().sort_index()

    # coerce needed columns
    if "Label_rt" not in df.columns:
        raise RuntimeError("CSV must contain 'Label_rt' column.")

    # Print header
    print(f"Loaded {len(df):,} rows from {args.csv}")
    print(f"Date span: {df.index.min()} → {df.index.max()}\n")

    # Label counts & flip stats
    counts = df["Label_rt"].value_counts().to_dict()
    flips = regime_flip_count(df["Label_rt"])
    durs = contiguous_durations(df["Label_rt"])
    dur_days = [d for (_, _, _, d) in durs]
    avg_d = float(np.mean(dur_days)) if dur_days else np.nan
    med_d = float(np.median(dur_days)) if dur_days else np.nan
    print(f"Label counts: {counts}")
    print(f"Flips: {flips} | Avg duration: {avg_d:.1f}d | Median duration: {med_d:.1f}d")

    # Horizons to check
    horizons = [1, 5, 20]
    for h in horizons:
        res = regime_metrics(df, h)
        m = res["means_by_lbl"]
        d = res["cohen_d"]
        print(f"\n[H={h}d] HitRate={res['hit_rate']:.3f} | IC(P)={res['ic_pearson']:.3f} | IC(S)={res['ic_spearman']:.3f} | "
              f"mean_fwd_logret: Bull={m.get('Bull',np.nan):+0.5f}, Bear={m.get('Bear',np.nan):+0.5f}, Choppy={m.get('Choppy',np.nan):+0.5f}")
        print(f"      Cohen's d: Bull−Bear={d['Bull-Bear']:+0.2f}, Bull−Choppy={d['Bull-Choppy']:+0.2f}, Choppy−Bear={d['Choppy-Bear']:+0.2f}")

    # Same-day purity
    same_day = {}
    if "LogReturn" in df.columns:
        lr = pd.to_numeric(df["LogReturn"], errors="coerce")
        print("\n=== Regime Purity (same-day returns) ===")
        for k in REG_ORDER:
            x = lr[df["Label_rt"]==k]
            mu = float(x.mean()); sd = float(x.std(ddof=1)); n = int(x.count())
            same_day[k] = (mu, sd, n)
            print(f"{k:<7} | mu={mu:+0.5f}  sigma={sd:0.5f}  n={n}")

    # Optional diagnostic backtest (NOT the goal)
    if args.backtest:
        # Need log returns; if absent, derive from close
        if "LogReturn" in df.columns:
            logret = pd.to_numeric(df["LogReturn"], errors="coerce")
        else:
            close = coerce_close_or_none(df)
            if close is None:
                raise RuntimeError("Backtest requires numeric 'close' or 'LogReturn'.")
            logret = np.log(close/close.shift(1))
        signal = encode_regime(df["Label_rt"])
        stats_bt = diag_backtest(df.index, logret, signal, cost_bps=args.cost_bps, lag=1)
        print("\n=== Optional lagged allocation backtest (diagnostic, not the goal) ===")
        print(f"Costs: {args.cost_bps:.1f} bps per |Δposition| (full flip costs {2*args.cost_bps:.1f} bps)")
        print(f"CAGR: {stats_bt['CAGR']*100:0.2f}% | Sharpe: {stats_bt['Sharpe'] if np.isfinite(stats_bt['Sharpe']) else float('nan'):.2f} "
              f"| Ann.Vol: {stats_bt['AnnVol']*100:0.2f}% | MaxDD: {stats_bt['MaxDD']*100:0.2f}%")
        print(f"Total return (strategy): {(stats_bt['TotalReturn']*100):+0.2f}%")

if __name__ == "__main__":
    main()