#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rotation strategy + Monte-Carlo on BTC/PAXG guided by your regime detector.

Rules:
- If regime == Bull: long BTC (AFTER DXY-DOWN confirm at transition into Bull)
- If regime == Bear: long PAXG (AFTER DXY-UP confirm at transition into Bear, and optional momentum gate)
- If regime == Choppy: flat (no position)
- Stay in the asset while regime persists (no repeated DXY gates).
- On any regime change, exit current asset immediately, then wait for the new gate.

Monte-Carlo knobs:
- Random entry delay (0..mc_max_delay_days)
- Random slippage on entry & exit (bps)
- Small i.i.d. noise added to daily log-returns

Outputs:
- <out_prefix>_summary.csv
- <out_prefix>_equity_paths.csv
- <out_prefix>_trade_log.csv
- <out_prefix>_equity_band.png
- <out_prefix>_cagr_hist.png
- <out_prefix>_positions.png
- <out_prefix>_btc_hodl_summary.csv
- <out_prefix>_bear_windows_summary.csv
"""

import argparse, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- utils ----------------
def _to_utc_index(idx):
    return pd.to_datetime(idx, utc=True)

def _ensure_series_1d(x, index=None):
    if isinstance(x, pd.DataFrame):
        x = x.squeeze()
    if isinstance(x, pd.Series):
        return pd.Series(np.asarray(x.values).reshape(-1), index=_to_utc_index(x.index))
    arr = np.asarray(x).reshape(-1)
    if index is None:
        raise ValueError("Index required when coercing raw array to Series.")
    return pd.Series(arr, index=_to_utc_index(index))

def drawdown(equity):
    peak = equity.cummax()
    return equity/peak - 1.0

def perf_stats_from_logrets(logrets):
    if len(logrets) == 0:
        return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan, max_dd=np.nan, tot_ret=np.nan)
    equity = np.exp(logrets.cumsum())
    yrs = max((equity.index[-1] - equity.index[0]).days/365.25, 1e-9)
    tot_ret = float(equity.iloc[-1]) - 1.0
    cagr = float(equity.iloc[-1])**(1/yrs) - 1.0
    vol = float(logrets.std()*np.sqrt(252))
    sharpe = float((logrets.mean()*252) / (vol + 1e-12))
    max_dd = float(drawdown(equity).min())
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, max_dd=max_dd, tot_ret=tot_ret)

# ---------------- data ----------------
def fetch_yf(ticker, start, end):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned empty for {ticker}")
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
    df.index = _to_utc_index(df.index)
    df = df[["open","high","low","close","volume"]].dropna()
    return df.sort_index()

def load_regimes_csv(path):
    df = pd.read_csv(path, parse_dates=["ts"])
    if "ts" not in df.columns:
        raise ValueError("regimes_csv must contain 'ts' column.")
    df["ts"] = _to_utc_index(df["ts"])
    df = df.set_index("ts").sort_index()
    for c in ["Label_rt","LogReturn"]:
        if c not in df.columns:
            raise ValueError(f"regimes_csv missing required column: {c}")
    return df

# -------------- DXY confirmation --------------
def dxy_confirm(dxy_close, ema_fast=10, ema_slow=30, hold_days=3):
    """
    Returns:
      up_flag   : True after 'hold_days' consecutive ef>es (DXY uptrend)
      down_flag : True after 'hold_days' consecutive ef<es (DXY downtrend)
    If hold_days <= 0, returns immediate raw ef>es and ef<es (no gating).
    """
    close = _ensure_series_1d(dxy_close)
    ef = close.ewm(span=ema_fast, adjust=False).mean()
    es = close.ewm(span=ema_slow, adjust=False).mean()
    up_raw   = (ef > es)
    down_raw = (ef < es)

    if hold_days <= 0:
        return up_raw.astype(bool), down_raw.astype(bool)

    def hold_flag(raw_bool):
        raw = np.asarray(raw_bool, dtype=bool).reshape(-1)
        out = np.zeros_like(raw, dtype=bool)
        run = 0
        for i, flag in enumerate(raw):
            run = run + 1 if flag else 0
            out[i] = (run >= hold_days)
        return pd.Series(out, index=close.index)

    return hold_flag(up_raw), hold_flag(down_raw)

# -------------- position builder --------------
def build_positions(dates, regimes, dxy_up, dxy_down, mom_series=None, mom_thresh=0.0):
    """
    mom_series: Series used as a Bear entry filter (absolute or relative momentum).
                Enter PAXG in Bear only when mom_series >= mom_thresh.
    """
    regimes = regimes.reindex(dates).ffill()
    dxy_up   = dxy_up.reindex(dates).fillna(False)
    dxy_down = dxy_down.reindex(dates).fillna(False)
    if mom_series is not None:
        mom_series = _ensure_series_1d(mom_series, index=dates).reindex(dates).fillna(0.0)

    pos = []
    last_pos = "CASH"
    last_regime = None
    waiting_gate = False
    wait_target = None  # "BTC" or "PAXG"

    for dt, reg in zip(dates, regimes.values):
        if reg not in ("Bull","Bear","Choppy"):
            reg = "Choppy"

        # regime change
        if (last_regime is None) or (reg != last_regime):
            last_pos = "CASH"
            waiting_gate = False
            wait_target = None

            if reg == "Bear":
                mom_ok = True
                if mom_series is not None:
                    mom_ok = bool(mom_series.loc[dt] >= mom_thresh)
                if mom_ok:
                    if dxy_up.loc[dt]:
                        last_pos = "PAXG"
                    else:
                        waiting_gate = True; wait_target = "PAXG"
                else:
                    last_pos = "CASH"

            elif reg == "Bull":
                if dxy_down.loc[dt]:
                    last_pos = "BTC"
                else:
                    waiting_gate = True; wait_target = "BTC"

            else:
                last_pos = "CASH"

            last_regime = reg
            pos.append(last_pos)
            continue

        # same regime
        if reg == "Choppy":
            last_pos = "CASH"; waiting_gate = False; wait_target = None

        elif reg == "Bear":
            mom_ok = True
            if mom_series is not None:
                mom_ok = bool(mom_series.loc[dt] >= mom_thresh)
            if not mom_ok:
                last_pos = "CASH"; waiting_gate = False; wait_target = None
            else:
                if last_pos != "PAXG":
                    if waiting_gate and wait_target == "PAXG" and dxy_up.loc[dt]:
                        last_pos = "PAXG"; waiting_gate=False; wait_target=None

        elif reg == "Bull":
            if last_pos != "BTC":
                if waiting_gate and wait_target == "BTC" and dxy_down.loc[dt]:
                    last_pos = "BTC"; waiting_gate=False; wait_target=None

        pos.append(last_pos)

    return pd.Series(pos, index=dates, name="target_pos")

# -------------- Monte-Carlo --------------
def simulate_mc(dates, pos_target, ret_btc, ret_paxg,
                sims=1000, max_delay_days=3,
                slip_in_bps_mean=5, slip_out_bps_mean=5,
                slip_bps_std_frac=0.5,
                ret_noise_sigma=0.001,
                seed=42):
    """
    Deterministic per-simulation RNG controls delays, slippage, and noise.
    Exit is applied before entry on switch bars.
    """
    dates = pd.DatetimeIndex(dates)
    N = len(dates)

    rb = np.asarray(_ensure_series_1d(ret_btc, index=dates).reindex(dates).fillna(0.0).values).reshape(-1)
    rp = np.asarray(_ensure_series_1d(ret_paxg, index=dates).reindex(dates).fillna(0.0).values).reshape(-1)

    tgt = pos_target.reindex(dates).fillna("CASH").astype(str).values

    switch_idx = []
    prev = tgt[0]
    for i in range(1, N):
        if tgt[i] != prev:
            switch_idx.append(i)
            prev = tgt[i]

    equity_paths = np.empty((sims, N), dtype=float)
    trade_rows = []

    for s in range(sims):
        rng = np.random.default_rng(seed + 1000 + s)

        Rb = rb + rng.normal(0.0, ret_noise_sigma, size=N)
        Rp = rp + rng.normal(0.0, ret_noise_sigma, size=N)

        eff = tgt.copy()
        if max_delay_days > 0 and len(switch_idx) > 0:
            delays = rng.integers(0, max_delay_days + 1, size=len(switch_idx))
            for j, i_sw in enumerate(switch_idx):
                d = int(delays[j])
                if d > 0:
                    old_state = eff[i_sw - 1]
                    i_end = min(N, i_sw + d)
                    eff[i_sw:i_end] = old_state  # extend old state during delay

        eq = 1.0
        path = np.zeros(N, dtype=float)
        cur_pos = "CASH"

        for t in range(N):
            pos = eff[t]
            entry = (cur_pos != pos) and (pos in ("BTC","PAXG"))
            exit_ = (cur_pos in ("BTC","PAXG")) and (pos != cur_pos)

            # EXIT first, then ENTRY
            if exit_:
                slip_out = max(0.0, rng.normal(slip_out_bps_mean, slip_out_bps_mean * slip_bps_std_frac)) / 10000.0
                eq *= (1.0 - slip_out)
                trade_rows.append({"sim": s, "date": dates[t], "action": f"EXIT_{cur_pos}", "equity": eq})

            if entry:
                slip_in = max(0.0, rng.normal(slip_in_bps_mean, slip_in_bps_mean * slip_bps_std_frac)) / 10000.0
                eq *= (1.0 - slip_in)
                trade_rows.append({"sim": s, "date": dates[t], "action": f"BUY_{pos}", "equity": eq})

            if pos == "BTC":
                eq *= math.exp(Rb[t])
            elif pos == "PAXG":
                eq *= math.exp(Rp[t])
            # CASH → no change

            path[t] = eq
            cur_pos = pos

        equity_paths[s, :] = path

    equity_wide = pd.DataFrame(equity_paths.T, index=dates, columns=[f"sim_{i}" for i in range(sims)])
    # daily log returns by path
    logrets = equity_wide.apply(lambda col: np.log(col/col.shift(1)).fillna(0.0))
    stats = []
    for i in range(sims):
        st = perf_stats_from_logrets(logrets.iloc[:, i])
        st["sim"] = i
        stats.append(st)
    summary_df = pd.DataFrame(stats).set_index("sim")

    trade_log = pd.DataFrame(trade_rows)
    if not trade_log.empty:
        trade_log["date"] = pd.to_datetime(trade_log["date"], utc=True)
        trade_log = trade_log.sort_values(["sim","date"]).reset_index(drop=True)

    return summary_df, equity_wide, trade_log

# ---------------- Bear-window evaluation ----------------
def find_bear_windows(regimes_series):
    reg = regimes_series.astype(str)
    idx = reg.index
    is_bear = (reg == "Bear").values
    windows = []
    i = 0
    while i < len(is_bear):
        if is_bear[i]:
            j = i
            while j + 1 < len(is_bear) and is_bear[j + 1]:
                j += 1
            windows.append((idx[i], idx[j]))
            i = j + 1
        else:
            i += 1
    return windows

def summarize_bears(pref, dates, regimes_series, equity_wide, r_btc, r_pax):
    reg = regimes_series.reindex(dates).ffill()
    windows = find_bear_windows(reg)

    logret_wide = equity_wide.apply(lambda col: np.log(col/col.shift(1))).fillna(0.0)

    rows = []
    all_mask = pd.Series(False, index=dates)
    for (a, b) in windows:
        mask = (dates >= a) & (dates <= b)
        all_mask |= mask
        n = int(mask.sum())

        btc_ret = float(np.exp(r_btc.reindex(dates)[mask].sum()) - 1.0)
        pax_ret = float(np.exp(r_pax.reindex(dates)[mask].sum()) - 1.0)

        sim_rets = np.exp(logret_wide[mask].sum(axis=0).values) - 1.0
        p10 = float(np.percentile(sim_rets, 10))
        p50 = float(np.percentile(sim_rets, 50))
        p90 = float(np.percentile(sim_rets, 90))

        rows.append(dict(window="BEAR",
                         start=a.strftime("%Y-%m-%d"),
                         end=b.strftime("%Y-%m-%d"),
                         days=n,
                         btc_ret=btc_ret, paxg_ret=pax_ret,
                         strat_p10=p10, strat_p50=p50, strat_p90=p90))

    if all_mask.any():
        btc_ret_all = float(np.exp(r_btc.reindex(dates)[all_mask].sum()) - 1.0)
        pax_ret_all = float(np.exp(r_pax.reindex(dates)[all_mask].sum()) - 1.0)
        sim_rets_all = np.exp(logret_wide[all_mask].sum(axis=0).values) - 1.0
        rows.append(dict(window="ALL_BEARS",
                         start=pd.to_datetime(dates[all_mask].min()).strftime("%Y-%m-%d"),
                         end=pd.to_datetime(dates[all_mask].max()).strftime("%Y-%m-%d"),
                         days=int(all_mask.sum()),
                         btc_ret=btc_ret_all, paxg_ret=pax_ret_all,
                         strat_p10=float(np.percentile(sim_rets_all, 10)),
                         strat_p50=float(np.percentile(sim_rets_all, 50)),
                         strat_p90=float(np.percentile(sim_rets_all, 90))))

    out = pd.DataFrame(rows, columns=["window","start","end","days","btc_ret","paxg_ret","strat_p10","strat_p50","strat_p90"])
    out.to_csv(f"{pref}_bear_windows_summary.csv", index=False)

    if not out.empty:
        print("\n=== Bear regime performance ===")
        last = out.iloc[-1] if out.iloc[-1]["window"] == "ALL_BEARS" else None
        for _, row in out.iterrows():
            tag = row["window"]
            print(f"{tag:>10}  {row['start']} → {row['end']}  ({int(row['days'])}d)  "
                  f"BTC {row['btc_ret']:+.1%} | PAXG {row['paxg_ret']:+.1%} | "
                  f"Strat p50 {row['strat_p50']:+.1%} [p10 {row['strat_p10']:+.1%}, p90 {row['strat_p90']:+.1%}]")
        if last is not None:
            print("→ Across ALL Bear days your strategy’s median return was "
                  f"{last['strat_p50']:+.1%} vs BTC {last['btc_ret']:+.1%} and PAXG {last['paxg_ret']:+.1%}")

# ---------------- plotting helpers ----------------
def plot_equity_band_vs_btc(pref, dates, equity_wide, r_btc):
    eq_med = equity_wide.median(axis=1)
    eq_p10 = equity_wide.quantile(0.10, axis=1)
    eq_p90 = equity_wide.quantile(0.90, axis=1)
    eq_btc = np.exp(_ensure_series_1d(r_btc, index=dates).reindex(dates).fillna(0.0).cumsum())

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(dates, eq_med, label="Strategy (median)", linewidth=2.0)
    ax.fill_between(dates, eq_p10, eq_p90, alpha=0.2, label="Strategy p10–p90")
    ax.plot(dates, eq_btc, label="BTC HODL", linewidth=1.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_title("Equity (log) — Strategy band vs BTC HODL")
    ax.set_xlabel("Date"); ax.set_ylabel("Equity (start=1.0)")
    ax.grid(alpha=0.25); ax.legend()
    plt.tight_layout(); plt.savefig(f"{pref}_equity_band.png", dpi=140); plt.close()

def plot_cagr_hist(pref, summary_df):
    fig, ax = plt.subplots(figsize=(10,5.5))
    ax.hist(summary_df["cagr"], bins=40)
    ax.set_title("Distribution of CAGR across simulations")
    ax.set_xlabel("CAGR"); ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(f"{pref}_cagr_hist.png", dpi=140); plt.close()

def plot_positions(pref, dates, pos_series):
    vals = pos_series.reindex(dates).fillna("CASH").astype(str)
    btc = (vals == "BTC").astype(int)
    pax = (vals == "PAXG").astype(int)
    cas = (vals == "CASH").astype(int)

    fig, ax = plt.subplots(figsize=(12,3.8))
    ax.stackplot(dates, cas, pax, btc, labels=["CASH","PAXG","BTC"])
    ax.set_title("Target position over time")
    ax.set_xlabel("Date"); ax.set_yticks([0,1]); ax.set_yticklabels([" "," "])
    ax.grid(alpha=0.2, axis="x"); ax.legend(loc="upper left", ncol=3)
    plt.tight_layout(); plt.savefig(f"{pref}_positions.png", dpi=140); plt.close()

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes_csv", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--btc", default="BTC-USD")
    ap.add_argument("--paxg", default="PAXG-USD")
    ap.add_argument("--dxy_ticker", default="DX=F")
    ap.add_argument("--dxy_ema_fast", type=int, default=10)
    ap.add_argument("--dxy_ema_slow", type=int, default=30)
    ap.add_argument("--dxy_hold_days", type=int, default=3, help="0 => disable DXY gates (immediate entries)")
    # --- Bear momentum gate (NEW) ---
    ap.add_argument("--bear_mom_len", type=int, default=0, help="0=off; else lookback L for momentum")
    ap.add_argument("--bear_mom_thresh", type=float, default=0.0, help="threshold in log terms")
    ap.add_argument("--bear_rel_mom", action="store_true",
                    help="Use RELATIVE momentum (PAXG vs BTC) in Bear instead of absolute PAXG momentum")
    # MC
    ap.add_argument("--mc_sims", type=int, default=500)
    ap.add_argument("--mc_max_delay_days", type=int, default=3)
    ap.add_argument("--mc_slip_in_bps_mean", type=float, default=5.0)
    ap.add_argument("--mc_slip_out_bps_mean", type=float, default=5.0)
    ap.add_argument("--mc_ret_noise_sigma", type=float, default=0.001)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end   = pd.Timestamp(args.end, tz="UTC")

    reg = load_regimes_csv(args.regimes_csv)
    reg = reg.loc[(reg.index >= start) & (reg.index <= end)].copy()
    if reg.empty:
        raise RuntimeError("No regime rows in requested window.")

    btc = fetch_yf(args.btc, args.start, args.end)
    pax = fetch_yf(args.paxg, args.start, args.end)
    try:
        dxy = fetch_yf(args.dxy_ticker, args.start, args.end)
    except Exception:
        dxy = fetch_yf("^DXY", args.start, args.end)

    dates = reg.index.union(btc.index).union(pax.index).union(dxy.index)
    dates = dates.sort_values()

    def logrets_from_close(df):
        cs = df["close"].reindex(dates).ffill()
        lr = np.log(cs/cs.shift(1)).fillna(0.0)
        return _ensure_series_1d(lr, index=dates)

    r_btc = logrets_from_close(btc)
    r_pax = logrets_from_close(pax)
    dxy_close = _ensure_series_1d(dxy["close"].reindex(dates).ffill(), index=dates)

    up_flag, down_flag = dxy_confirm(dxy_close,
                                     ema_fast=args.dxy_ema_fast,
                                     ema_slow=args.dxy_ema_slow,
                                     hold_days=args.dxy_hold_days)

    # --- momentum series (absolute or relative) for Bear gate ---
    mom_series = None
    if args.bear_mom_len and args.bear_mom_len > 0:
        L = int(args.bear_mom_len)
        pax_close = _ensure_series_1d(pax["close"], index=pax.index).reindex(dates).ffill()
        if args.bear_rel_mom:
            btc_close = _ensure_series_1d(btc["close"], index=btc.index).reindex(dates).ffill()
            mom_series = np.log(pax_close / btc_close) - np.log(pax_close.shift(L) / btc_close.shift(L))
        else:
            mom_series = np.log(pax_close / pax_close.shift(L))

    regimes_series = reg["Label_rt"].reindex(dates).ffill()
    pos_target = build_positions(dates, regimes_series, up_flag, down_flag,
                                 mom_series=mom_series, mom_thresh=float(args.bear_mom_thresh))

    summary_df, equity_wide, trade_log = simulate_mc(
        dates, pos_target, r_btc, r_pax,
        sims=args.mc_sims,
        max_delay_days=args.mc_max_delay_days,
        slip_in_bps_mean=args.mc_slip_in_bps_mean,
        slip_out_bps_mean=args.mc_slip_out_bps_mean,
        ret_noise_sigma=args.mc_ret_noise_sigma,
        seed=42
    )

    pref = args.out_prefix
    summary_df.to_csv(f"{pref}_summary.csv", index=True)
    equity_wide.to_csv(f"{pref}_equity_paths.csv", index=True)
    trade_log.to_csv(f"{pref}_trade_log.csv", index=False)

    hodl_stats = perf_stats_from_logrets(r_btc)
    pd.DataFrame([hodl_stats]).to_csv(f"{pref}_btc_hodl_summary.csv", index=False)

    med = summary_df.median(); p10 = summary_df.quantile(0.10); p90 = summary_df.quantile(0.90)
    print("\n=== Monte-Carlo summary (median [p10..p90]) ===")
    print(f"CAGR    : {med['cagr']:+.2%}  [{p10['cagr']:+.2%} .. {p90['cagr']:+.2%}]")
    print(f"Vol     : {med['vol']:.2%}   [{p10['vol']:.2%} .. {p90['vol']:.2%}]")
    print(f"Sharpe  : {med['sharpe']:.2f}   [{p10['sharpe']:.2f} .. {p90['sharpe']:.2f}]")
    print(f"MaxDD   : {med['max_dd']:.2%}   [{p10['max_dd']:.2%} .. {p90['max_dd']:.2%}]")
    print(f"TotRet  : {med['tot_ret']:+.2%}  [{p10['tot_ret']:+.2%} .. {p90['tot_ret']:+.2%}]")

    print("\n=== BTC HODL benchmark (same window) ===")
    print(f"CAGR    : {hodl_stats['cagr']:+.2%}")
    print(f"Vol     : {hodl_stats['vol']:.2%}")
    print(f"Sharpe  : {hodl_stats['sharpe']:.2f}")
    print(f"MaxDD   : {hodl_stats['max_dd']:.2%}")
    print(f"TotRet  : {hodl_stats['tot_ret']:+.2%}")

    plot_equity_band_vs_btc(pref, dates, equity_wide, r_btc)
    plot_cagr_hist(pref, summary_df)
    plot_positions(pref, dates, pos_target)

    summarize_bears(pref, dates, regimes_series, equity_wide, r_btc, r_pax)

    print(f"[OK] Wrote:")
    print(f"  {pref}_summary.csv")
    print(f"  {pref}_equity_paths.csv")
    print(f"  {pref}_trade_log.csv")
    print(f"  {pref}_btc_hodl_summary.csv")
    print(f"  {pref}_equity_band.png")
    print(f"  {pref}_cagr_hist.png")
    print(f"  {pref}_positions.png")
    print(f"  {pref}_bear_windows_summary.csv")

if __name__ == "__main__":
    main()