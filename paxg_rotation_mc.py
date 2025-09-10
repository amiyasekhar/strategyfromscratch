#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rotation strategy + Monte-Carlo on BTC/PAXG guided by your regime detector.

Bear market behavior:
- Enter PAXG only when BOTH BTC is weak (absolute or relative) AND PAXG is strong.
- Optional PAXG trailing stop with cool-down; re-enter only after cool-down and PAXG regains strength.
- Bull => long BTC (optional DXY-down gate)
- Choppy => flat
- On regime change, exit immediately; then wait for the new gate.

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

# -------------- indicators --------------
def true_range(h,l,c):
    pc = c.shift(1)
    a = (h - l)
    b = (h - pc).abs()
    d = (l - pc).abs()
    return pd.concat([a,b,d], axis=1).max(axis=1)

def atr_from_ohlc(df, n=14):
    h = df["high"]; l = df["low"]; c = df["close"]
    tr = true_range(h,l,c)
    return tr.rolling(n).mean()

# -------------- DXY confirmation --------------
def dxy_confirm(dxy_close, ema_fast=10, ema_slow=30, hold_days=3):
    close = _ensure_series_1d(dxy_close)
    ef = close.ewm(span=ema_fast, adjust=False).mean()
    es = close.ewm(span=ema_slow, adjust=False).mean()
    up_raw   = ef > es
    down_raw = ef < es

    def hold_flag(raw_bool):
        raw = np.asarray(raw_bool, dtype=bool).reshape(-1)
        out = np.zeros_like(raw, dtype=bool)
        run = 0
        for i, flag in enumerate(raw):
            run = run + 1 if flag else 0
            out[i] = (run >= hold_days)
        return pd.Series(out, index=close.index)

    if hold_days <= 0:
        return pd.Series(up_raw.values, index=close.index), pd.Series(down_raw.values, index=close.index)
    return hold_flag(up_raw), hold_flag(down_raw)

# -------------- position builder --------------
def build_positions(
    dates,
    regimes,
    dxy_up, dxy_down,
    use_dxy=True,
    # BTC weakness gate
    btc_mom=None, bear_mom_thresh=0.0,
    # PAXG strength gate
    paxg_mom=None, paxg_mom_thresh=0.0,
    # PAXG trailing stop bits
    pax_close=None, pax_atr=None,
    use_paxg_trail=False, trail_k=3.0, reenter_cooldown=5
):
    """
    Enter PAXG in Bear only if BTC is weak AND PAXG is strong.
    Optional PAXG trailing stop => exit to CASH; after cooldown, allow re-entry if gates satisfied again.
    """
    regimes = regimes.reindex(dates).ffill()
    dxy_up   = dxy_up.reindex(dates).fillna(False)
    dxy_down = dxy_down.reindex(dates).fillna(False)

    if btc_mom is not None:
        btc_mom = _ensure_series_1d(btc_mom, index=dates).reindex(dates).fillna(0.0)
    if paxg_mom is not None:
        paxg_mom = _ensure_series_1d(paxg_mom, index=dates).reindex(dates).fillna(0.0)

    if pax_close is not None:
        pax_close = _ensure_series_1d(pax_close, index=dates).reindex(dates).ffill()
    if pax_atr is not None:
        pax_atr = _ensure_series_1d(pax_atr, index=dates).reindex(dates).ffill()

    pos = []
    last_regime = None
    last_pos = "CASH"
    cooldown_left = 0
    high_close = None

    for dt, reg in zip(dates, regimes.values):
        if reg not in ("Bull","Bear","Choppy"):
            reg = "Choppy"

        # regime flip => exit, reset
        if (last_regime is None) or (reg != last_regime):
            last_pos = "CASH"
            high_close = None
            if reg == "Bear":
                if (btc_mom is None or btc_mom.loc[dt] <= bear_mom_thresh) and \
                   (paxg_mom is None or paxg_mom.loc[dt] >= paxg_mom_thresh) and \
                   (cooldown_left == 0):
                    last_pos = "PAXG"
                    high_close = float(pax_close.loc[dt]) if pax_close is not None else None
            elif reg == "Bull":
                if not use_dxy or dxy_down.loc[dt]:
                    last_pos = "BTC"
            else:  # Choppy
                last_pos = "CASH"

            last_regime = reg
            pos.append(last_pos)
            continue

        # same-regime day
        if reg == "Choppy":
            last_pos = "CASH"
            high_close = None

        elif reg == "Bear":
            # tick cooldown
            if cooldown_left > 0:
                cooldown_left -= 1

            # trailing stop (only when long PAXG)
            if use_paxg_trail and last_pos == "PAXG" and pax_close is not None and pax_atr is not None:
                px = float(pax_close.loc[dt]); atr = float(pax_atr.loc[dt]) if np.isfinite(pax_atr.loc[dt]) else 0.0
                if high_close is None:
                    high_close = px
                else:
                    high_close = max(high_close, px)
                stop = high_close - trail_k * atr if np.isfinite(atr) else -np.inf
                if px < stop:
                    last_pos = "CASH"
                    high_close = None
                    cooldown_left = reenter_cooldown

            # consider entries (or re-entries) only if not currently in PAXG
            if last_pos != "PAXG":
                enter_ready = (
                    (btc_mom is None or btc_mom.loc[dt] <= bear_mom_thresh) and
                    (paxg_mom is None or paxg_mom.loc[dt] >= paxg_mom_thresh) and
                    (cooldown_left == 0)
                )
                if enter_ready:
                    if not use_dxy or dxy_up.loc[dt]:
                        last_pos = "PAXG"
                        high_close = float(pax_close.loc[dt]) if pax_close is not None else None

        elif reg == "Bull":
            if last_pos != "BTC":
                if not use_dxy or dxy_down.loc[dt]:
                    last_pos = "BTC"

        pos.append(last_pos)

    return pd.Series(pos, index=dates, name="target_pos")

# -------------- Monte-Carlo --------------
def simulate_mc(dates, pos_target, ret_btc, ret_paxg,
                sims=1000, max_delay_days=3,
                slip_in_bps_mean=5, slip_out_bps_mean=5,
                slip_bps_std_frac=0.5,
                ret_noise_sigma=0.001,
                seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(dates)
    N = len(dates)

    rb = np.asarray(_ensure_series_1d(ret_btc, index=dates).reindex(dates).fillna(0.0).values).reshape(-1)
    rp = np.asarray(_ensure_series_1d(ret_paxg, index=dates).reindex(dates).fillna(0.0).values).reshape(-1)

    noise_btc  = rng.normal(0.0, ret_noise_sigma, size=(sims, N))
    noise_paxg = rng.normal(0.0, ret_noise_sigma, size=(sims, N))

    Rb = np.tile(rb[None, :], (sims, 1)) + noise_btc
    Rp = np.tile(rp[None, :], (sims, 1)) + noise_paxg

    tgt = pos_target.reindex(dates).fillna("CASH").astype(str).values

    # switch indices for delay simulation
    switch_idx = []
    prev = tgt[0]
    for i in range(1, N):
        if tgt[i] != prev:
            switch_idx.append(i)
            prev = tgt[i]

    equity_paths = np.empty((sims, N), dtype=float)
    trade_rows = []

    for s in range(sims):
        eff = tgt.copy()
        if max_delay_days > 0 and len(switch_idx) > 0:
            rng_delays = np.random.default_rng(seed + 13*s)
            delays = rng_delays.integers(0, max_delay_days+1, size=len(switch_idx))
            for j, i_sw in enumerate(switch_idx):
                d = int(delays[j])
                if d > 0:
                    new_state = eff[i_sw]
                    old_state = eff[i_sw-1]
                    i_end = min(N, i_sw + d)
                    eff[i_sw:i_end] = old_state

        eq = 1.0
        path = np.zeros(N, dtype=float)
        cur_pos = "CASH"
        for t in range(N):
            pos = eff[t]
            entry = (cur_pos != pos) and (pos in ("BTC","PAXG"))
            exit_ = (cur_pos in ("BTC","PAXG")) and (pos != cur_pos)

            if entry:
                slip_in = max(0.0, np.random.normal(slip_in_bps_mean, slip_in_bps_mean*slip_bps_std_frac))/10000.0
                eq *= (1.0 - slip_in)
                trade_rows.append({"sim": s, "date": dates[t], "action": f"BUY_{pos}", "equity": eq})
            if exit_:
                slip_out = max(0.0, np.random.normal(slip_out_bps_mean, slip_out_bps_mean*slip_bps_std_frac))/10000.0
                eq *= (1.0 - slip_out)
                trade_rows.append({"sim": s, "date": dates[t], "action": f"EXIT_{cur_pos}", "equity": eq})

            if pos == "BTC":
                eq *= math.exp(Rb[s, t])
            elif pos == "PAXG":
                eq *= math.exp(Rp[s, t])
            # CASH → no change

            path[t] = eq
            cur_pos = pos

        equity_paths[s, :] = path

    equity_wide = pd.DataFrame(equity_paths.T, index=dates, columns=[f"sim_{i}" for i in range(sims)])
    # daily log-returns per simulation
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

    return summary_df, equity_wide, logrets, trade_log

# ---------------- plotting ----------------
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
    plt.tight_layout(); fig.savefig(f"{pref}_equity_band.png", dpi=140); plt.close(fig)

def plot_cagr_hist(pref, summary_df):
    fig, ax = plt.subplots(figsize=(10,5.5))
    ax.hist(summary_df["cagr"], bins=40)
    ax.set_title("Distribution of CAGR across simulations")
    ax.set_xlabel("CAGR"); ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)
    plt.tight_layout(); fig.savefig(f"{pref}_cagr_hist.png", dpi=140); plt.close(fig)

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
    plt.tight_layout(); fig.savefig(f"{pref}_positions.png", dpi=140); plt.close(fig)

# ---------------- helpers: bear windows summary ----------------
def contiguous_windows(mask: pd.Series):
    idx = mask.index
    m = mask.values.astype(bool)
    out = []
    if len(m) == 0: return out
    i = 0
    while i < len(m):
        if not m[i]:
            i += 1; continue
        j = i
        while j+1 < len(m) and m[j+1]:
            j += 1
        out.append((idx[i], idx[j]))
        i = j + 1
    return out

def summarize_bears(pref, dates, regimes, r_btc, r_paxg, sim_logrets):
    bear_mask = regimes.reindex(dates).eq("Bear").fillna(False)
    wins = contiguous_windows(bear_mask)

    rows = []
    for (a,b) in wins:
        span = (dates >= a) & (dates <= b)
        n = int(span.sum())
        btc_ret = float(np.exp(r_btc.reindex(dates)[span].sum()) - 1.0)
        pax_ret = float(np.exp(r_paxg.reindex(dates)[span].sum()) - 1.0)
        sim_span_rets = np.exp(sim_logrets.loc[span, :].sum(axis=0)) - 1.0
        p50 = float(np.median(sim_span_rets))
        p10 = float(np.percentile(sim_span_rets, 10))
        p90 = float(np.percentile(sim_span_rets, 90))
        rows.append(["BEAR", a, b, n, btc_ret, pax_ret, p50, p10, p90])

    if bear_mask.any():
        span = bear_mask.values
        n = int(span.sum())
        btc_ret = float(np.exp(r_btc.reindex(dates)[span].sum()) - 1.0)
        pax_ret = float(np.exp(r_paxg.reindex(dates)[span].sum()) - 1.0)
        sim_span_rets = np.exp(sim_logrets.loc[span, :].sum(axis=0)) - 1.0
        p50 = float(np.median(sim_span_rets))
        p10 = float(np.percentile(sim_span_rets, 10))
        p90 = float(np.percentile(sim_span_rets, 90))
        rows.append(["ALL_BEARS", dates.min(), dates.max(), n, btc_ret, pax_ret, p50, p10, p90])

    out = pd.DataFrame(rows, columns=[
        "label","start","end","days","btc_ret","paxg_ret","strat_p50","strat_p10","strat_p90"
    ])
    out.to_csv(f"{pref}_bear_windows_summary.csv", index=False)

    print("\n=== Bear regime performance ===")
    for _,r in out.iterrows():
        lab = f"{r['label']:<10}"
        a = pd.Timestamp(r["start"]).date(); b = pd.Timestamp(r["end"]).date()
        print(f"{lab} {a} → {b}  ({int(r['days'])}d)  "
              f"BTC {r['btc_ret']:+.1%} | PAXG {r['paxg_ret']:+.1%} | "
              f"Strat p50 {r['strat_p50']:+.1%} [p10 {r['strat_p10']:+.1%}, p90 {r['strat_p90']:+.1%}]")
    if not out.empty and out.iloc[-1,0] == "ALL_BEARS":
        r = out.iloc[-1]
        print(f"→ Across ALL Bear days your strategy’s median return was "
              f"{r['strat_p50']:+.1%} vs BTC {r['btc_ret']:+.1%} and PAXG {r['paxg_ret']:+.1%}")

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
    ap.add_argument("--dxy_hold_days", type=int, default=3)  # 0 => ignore DXY

    # Bear gating: BTC weakness (absolute or relative if --bear_rel_mom)
    ap.add_argument("--bear_mom_len", type=int, default=0, help="0=off; lookback L for BTC momentum or (BTC-PAXG) if --bear_rel_mom")
    ap.add_argument("--bear_mom_thresh", type=float, default=0.0, help="BTC weakness threshold (<= means weak). For --bear_rel_mom this applies to BTC-PAXG.")
    ap.add_argument("--bear_rel_mom", action="store_true", help="Use relative momentum (BTC minus PAXG) for the BTC weakness check.")

    # PAXG strength gate
    ap.add_argument("--paxg_mom_len", type=int, default=0, help="0=off; lookback L for PAXG momentum (>= means strong).")
    ap.add_argument("--paxg_mom_thresh", type=float, default=0.0, help="PAXG strength threshold (>= means strong).")

    # PAXG trailing stop
    ap.add_argument("--paxg_trailing_stop", action="store_true", help="Enable PAXG trailing stop during Bear.")
    ap.add_argument("--paxg_trail_atr_len", type=int, default=14, help="ATR length for trailing stop.")
    ap.add_argument("--paxg_trail_k", type=float, default=2.5, help="Stop = highest close since entry minus k*ATR.")
    ap.add_argument("--paxg_reenter_cooldown", type=int, default=7, help="Days to wait after a stop-out before re-considering PAXG.")

    # Monte-Carlo
    ap.add_argument("--mc_sims", type=int, default=500)
    ap.add_argument("--mc_max_delay_days", type=int, default=3)
    ap.add_argument("--mc_slip_in_bps_mean", type=float, default=5.0)
    ap.add_argument("--mc_slip_out_bps_mean", type=float, default=5.0)
    ap.add_argument("--mc_ret_noise_sigma", type=float, default=0.001)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end   = pd.Timestamp(args.end, tz="UTC")

    # regimes
    reg = load_regimes_csv(args.regimes_csv)
    reg = reg.loc[(reg.index >= start) & (reg.index <= end)].copy()
    if reg.empty:
        raise RuntimeError("No regime rows in requested window.")

    # prices
    btc = fetch_yf(args.btc, args.start, args.end)
    pax = fetch_yf(args.paxg, args.start, args.end)
    try:
        dxy = fetch_yf(args.dxy_ticker, args.start, args.end)
    except Exception:
        dxy = fetch_yf("^DXY", args.start, args.end)

    # master dates
    dates = reg.index.union(btc.index).union(pax.index).union(dxy.index)
    dates = dates.sort_values()

    # returns
    def logrets_from_close(df):
        cs = df["close"].reindex(dates).ffill()
        lr = np.log(cs/cs.shift(1)).fillna(0.0)
        return _ensure_series_1d(lr, index=dates)

    r_btc = logrets_from_close(btc)
    r_pax = logrets_from_close(pax)

    # momentum series for gates
    btc_close = _ensure_series_1d(btc["close"], index=btc.index).reindex(dates).ffill()
    pax_close = _ensure_series_1d(pax["close"], index=pax.index).reindex(dates).ffill()

    btc_mom = None
    if args.bear_mom_len and args.bear_mom_len > 0:
        Lb = int(args.bear_mom_len)
        if args.bear_rel_mom:
            btc_mom = np.log(btc_close/btc_close.shift(Lb)) - np.log(pax_close/pax_close.shift(Lb))
        else:
            btc_mom = np.log(btc_close/btc_close.shift(Lb))

    paxg_mom = None
    if args.paxg_mom_len and args.paxg_mom_len > 0:
        Lp = int(args.paxg_mom_len)
        paxg_mom = np.log(pax_close/pax_close.shift(Lp))

    # PAXG ATR for trailing stop (if enabled)
    pax_atr = None
    if args.paxg_trailing_stop:
        pax_atr = atr_from_ohlc(pax, n=int(args.paxg_trail_atr_len)).reindex(dates).ffill()

    dxy_close = _ensure_series_1d(dxy["close"].reindex(dates).ffill(), index=dates)
    up_flag, down_flag = dxy_confirm(dxy_close,
                                     ema_fast=args.dxy_ema_fast,
                                     ema_slow=args.dxy_ema_slow,
                                     hold_days=args.dxy_hold_days)
    use_dxy = args.dxy_hold_days > 0

    # target positions
    regimes_series = reg["Label_rt"].reindex(dates).ffill()
    pos_target = build_positions(
        dates, regimes_series, up_flag, down_flag,
        use_dxy=use_dxy,
        btc_mom=btc_mom, bear_mom_thresh=float(args.bear_mom_thresh),
        paxg_mom=paxg_mom, paxg_mom_thresh=float(args.paxg_mom_thresh),
        pax_close=pax_close, pax_atr=pax_atr,
        use_paxg_trail=bool(args.paxg_trailing_stop),
        trail_k=float(args.paxg_trail_k),
        reenter_cooldown=int(args.paxg_reenter_cooldown)
    )

    # Monte-Carlo
    summary_df, equity_wide, sim_logrets, trade_log = simulate_mc(
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

    # BTC HODL benchmark
    hodl_stats = perf_stats_from_logrets(r_btc)
    pd.DataFrame([hodl_stats]).to_csv(f"{pref}_btc_hodl_summary.csv", index=False)

    # Text summary
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

    # Plots
    plot_equity_band_vs_btc(pref, dates, equity_wide, r_btc)
    plot_cagr_hist(pref, summary_df)
    plot_positions(pref, dates, pos_target)

    # Bear windows summary
    summarize_bears(pref, dates, regimes_series, r_btc, r_pax, sim_logrets)

    print("[OK] Wrote:")
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