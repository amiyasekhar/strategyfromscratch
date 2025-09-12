#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test.py — Rotation strategy + Robust Monte-Carlo on BTC/PAXG guided by your regime detector.

What’s inside (key features):
- Bear: long PAXG only when BTC is weak AND PAXG is strong (your gates).
- Bull: long BTC (optional DXY gating via --dxy_hold_days > 0).
- Choppy: flat.
- Trailing stop on PAXG (ATR*k) with N-day confirmation and fast re-entry on new-high.
- Robust Monte-Carlo with:
  * Entry/exit delays (geometric), partial fills, missed trades
  * Random “drought” (temporarily disable new entries)
  * Fees + spread + slippage with regime multipliers and ATR%-linked bump
  * Return noise, clustered “shock” days, optional weekend gaps
- Output: summary CSVs, equity paths, trade log, plots, and bear windows performance table.

Usage example (matches your last run shape):
python3 test.py \
  --regimes_csv btc_usd_regimes_2014train_2019plus_test.csv \
  --start 2019-01-01 --end 2025-12-31 \
  --dxy_hold_days 0 \
  --bear_rel_mom --bear_mom_len 40 --bear_mom_thresh 0.0 \
  --paxg_mom_len 40 --paxg_mom_thresh 0.01 \
  --paxg_trailing_stop --paxg_trail_atr_len 14 --paxg_trail_k 2.5 \
  --stop_confirm_days 2 --reenter_mode newhigh --reenter_eps 0.003 \
  --mc_sims 1000 \
  --mc_max_delay_days 3 --mc_delay_geom_p 0.5 \
  --mc_partial_fill_max_days 2 \
  --mc_miss_prob 0.01 \
  --mc_drought_prob 0.005 --mc_drought_mean_days 3 \
  --mc_fee_bps_in 4 --mc_fee_bps_out 4 --mc_spread_bps 2 \
  --mc_slip_in_bps_mean 5 --mc_slip_out_bps_mean 5 \
  --mc_slip_mult_bull 1.0 --mc_slip_mult_bear 1.3 --mc_slip_mult_choppy 1.15 \
  --mc_vol_slip_k 50 \
  --mc_ret_noise_sigma 0.001 \
  --mc_shock_prob 0.01 --mc_shock_sigma 0.02 --mc_shock_mean_days 1 \
  --mc_gap_weekend --mc_gap_sigma 0.01 \
  --out_prefix rot_dual_mom_realisticMC_2019_2025
"""

import argparse, math, warnings, sys
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
        s = pd.Series(np.asarray(x.values).reshape(-1), index=_to_utc_index(x.index))
        return s
    arr = np.asarray(x)
    if arr.ndim > 1:
        raise ValueError(f"_ensure_series_1d expected 1-D, got {arr.ndim}-D with shape {arr.shape}")
    if index is None:
        raise ValueError("Index required when coercing raw array to Series.")
    return pd.Series(arr.reshape(-1), index=_to_utc_index(index))

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
    use_paxg_trail=False, trail_k=3.0,
    # NEW: stop confirmation + re-entry control
    stop_confirm_days=1,                 # require N consecutive closes below trail to stop
    reenter_mode="cooldown",             # "cooldown" or "newhigh"
    reenter_cooldown=5,                  # used if reenter_mode="cooldown"
    reenter_eps=0.003                    # used if reenter_mode="newhigh"
):
    """
    Enter PAXG in Bear only if BTC is weak AND PAXG is strong.
    Trailing stop can be confirmed over multiple days.
    After stop-out, either wait a fixed cooldown or re-enter on new-high vs the stop-out anchor.
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

    def btc_is_weak(dt):
        return True if btc_mom is None else bool(btc_mom.loc[dt] <= bear_mom_thresh)

    def paxg_is_strong(dt):
        return True if paxg_mom is None else bool(paxg_mom.loc[dt] >= paxg_mom_thresh)

    pos = []
    last_regime = None
    last_pos = "CASH"

    # --- trailing stop state ---
    trail_high_close = None              # running high since entry
    trail_breach_run = 0                 # consecutive closes below trail
    # re-entry state
    cooldown_left = 0
    stopout_anchor_high = None           # last trail high at stop-out (for "newhigh" re-entry)

    for dt, reg in zip(dates, regimes.values):
        if reg not in ("Bull","Bear","Choppy"):
            reg = "Choppy"

        # regime flip => hard reset of position; keep cooldown ticking
        if (last_regime is None) or (reg != last_regime):
            last_pos = "CASH"
            trail_high_close = None
            trail_breach_run = 0
            stopout_anchor_high = None
            if cooldown_left > 0:
                cooldown_left -= 1

            if reg == "Bear":
                enter_ready = btc_is_weak(dt) and paxg_is_strong(dt)
                if enter_ready and ( (not use_dxy) or dxy_up.loc[dt] ):
                    if reenter_mode == "cooldown" and cooldown_left > 0:
                        pass
                    else:
                        last_pos = "PAXG"
                        if pax_close is not None:
                            trail_high_close = float(pax_close.loc[dt])
                        trail_breach_run = 0
                        stopout_anchor_high = None

            elif reg == "Bull":
                if (not use_dxy) or dxy_down.loc[dt]:
                    last_pos = "BTC"

            else:  # Choppy
                last_pos = "CASH"

            last_regime = reg
            pos.append(last_pos)
            continue

        # same-regime logic
        if reg == "Choppy":
            last_pos = "CASH"
            trail_high_close = None
            trail_breach_run = 0
            stopout_anchor_high = None

        elif reg == "Bear":
            # tick cooldown
            if cooldown_left > 0:
                cooldown_left -= 1

            # manage trailing stop if holding PAXG
            if use_paxg_trail and last_pos == "PAXG" and pax_close is not None and pax_atr is not None:
                px  = float(pax_close.loc[dt])
                atr = float(pax_atr.loc[dt]) if np.isfinite(pax_atr.loc[dt]) else 0.0
                trail_high_close = max(trail_high_close or px, px)
                stop_lvl = trail_high_close - trail_k * atr if np.isfinite(atr) else -np.inf

                if px < stop_lvl:
                    trail_breach_run += 1
                else:
                    trail_breach_run = 0

                if trail_breach_run >= max(1, int(stop_confirm_days)):
                    # confirmed stop-out
                    last_pos = "CASH"
                    cooldown_left = (reenter_cooldown if reenter_mode == "cooldown" else 0)
                    stopout_anchor_high = trail_high_close
                    trail_high_close = None
                    trail_breach_run = 0

            # entries / re-entries when NOT in PAXG
            if last_pos != "PAXG":
                need_btc_weak = btc_is_weak(dt)
                need_paxg_str = paxg_is_strong(dt)
                allow_entry = need_btc_weak and need_paxg_str

                if reenter_mode == "cooldown":
                    allow_entry = allow_entry and (cooldown_left == 0)
                elif reenter_mode == "newhigh":
                    if stopout_anchor_high is not None and pax_close is not None:
                        px_now = float(pax_close.loc[dt])
                        allow_entry = allow_entry and (px_now >= (1.0 + float(reenter_eps)) * float(stopout_anchor_high))

                if allow_entry and ( (not use_dxy) or dxy_up.loc[dt] ):
                    last_pos = "PAXG"
                    if pax_close is not None:
                        trail_high_close = float(pax_close.loc[dt])
                    trail_breach_run = 0
                    stopout_anchor_high = None

        else:  # Bull
            last_pos = "BTC"
            trail_high_close = None
            trail_breach_run = 0
            stopout_anchor_high = None

        pos.append(last_pos)

    return pd.Series(pos, index=dates, name="target_pos")

# ---------------- Robust Monte-Carlo ----------------
def _geom_delay(rng, p, max_days):
    """Geometric delay sampled as number of extra days (0..max_days)."""
    if p <= 0 or p > 1:
        return 0
    # Geometric on {1,2,...}; we want 0-based; cap at max_days
    k = rng.geometric(p) - 1
    return int(min(max_days, max(0, k)))

def simulate_mc_robust(
    dates,
    pos_target,
    ret_btc,
    ret_paxg,
    btc_atr_pct,
    pax_atr_pct,
    sims=1000,
    max_delay_days=3,
    delay_geom_p=0.5,
    partial_fill_max_days=2,
    miss_prob=0.0,
    drought_prob=0.0,
    drought_mean_days=3,
    fee_bps_in=0.0,
    fee_bps_out=0.0,
    spread_bps=0.0,
    slip_in_bps_mean=5.0,
    slip_out_bps_mean=5.0,
    slip_mult_bull=1.0,
    slip_mult_bear=1.2,
    slip_mult_choppy=1.1,
    vol_slip_k=50.0,           # add (vol_slip_k * ATR%) bps to slips
    ret_noise_sigma=0.0,
    shock_prob=0.0,
    shock_sigma=0.02,
    shock_mean_days=1,
    gap_weekend=False,
    gap_sigma=0.0,
    regimes_series=None,
    seed=42
):
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(dates)
    N = len(dates)

    # Core series (1-D aligned)
    rb = _ensure_series_1d(ret_btc, index=dates).reindex(dates).fillna(0.0).values.reshape(-1)
    rp = _ensure_series_1d(ret_paxg, index=dates).reindex(dates).fillna(0.0).values.reshape(-1)
    atr_b = _ensure_series_1d(btc_atr_pct, index=dates).reindex(dates).fillna(0.0).values.reshape(-1)
    atr_p = _ensure_series_1d(pax_atr_pct, index=dates).reindex(dates).fillna(0.0).values.reshape(-1)

    tgt = pos_target.reindex(dates).fillna("CASH").astype(str).values
    regimes = regimes_series.reindex(dates).fillna("Choppy").astype(str).values if regimes_series is not None else np.array(["Choppy"]*N)

    # Switch indices
    switch_idx = []
    prev = tgt[0]
    for i in range(1, N):
        if tgt[i] != prev:
            switch_idx.append(i)
            prev = tgt[i]

    # Helper: per-day slip multiplier by regime
    reg_mult = np.ones(N, dtype=float)
    for i in range(N):
        r = regimes[i]
        if r == "Bear":
            reg_mult[i] = float(slip_mult_bear)
        elif r == "Bull":
            reg_mult[i] = float(slip_mult_bull)
        else:
            reg_mult[i] = float(slip_mult_choppy)

    # Shock cluster flags (once for all sims; randomness per sim can be added by mixing seed if desired)
    shock_flags = np.zeros(N, dtype=bool)
    if shock_prob > 0 and shock_sigma > 0:
        i = 0
        while i < N:
            if rng.random() < shock_prob:
                dur = max(1, int(round(rng.poisson(lam=max(1e-6, shock_mean_days)))))
                shock_flags[i:i+dur] = True
                i += dur
            else:
                i += 1

    # Weekend gap mask (Mondays)
    is_monday = (pd.DatetimeIndex(dates).weekday == 0)

    equity_paths = np.empty((sims, N), dtype=float)
    trade_rows = []

    for s in range(sims):
        rs = np.random.default_rng(seed + 101*s)

        # Drought timer (days left where new entries are disallowed)
        drought_left = 0

        # Effective state array after delays/partial fills/missed trades
        eff = tgt.copy()
        # We’ll track “position scaling” 0..1 to model partial fills across days
        scale = np.ones(N, dtype=float)

        if len(switch_idx) > 0 and max_delay_days > 0:
            # For each switch, possibly miss or delay, and maybe partial-fill
            for j, i_sw in enumerate(switch_idx):
                # Miss trade?
                if rs.random() < miss_prob:
                    # Stay in previous state until the next switch (i.e., ignore this change)
                    old_state = eff[i_sw-1]
                    i_end = switch_idx[j+1] if (j+1 < len(switch_idx)) else N
                    eff[i_sw:i_end] = old_state
                    continue

                # Delay days (geometric)
                d = _geom_delay(rs, delay_geom_p, max_delay_days)
                if d > 0:
                    new_state = eff[i_sw]
                    old_state = eff[i_sw-1]
                    i_end = min(N, i_sw + d)
                    eff[i_sw:i_end] = old_state
                    # After delay, if we switch into BTC or PAXG, we might partially fill
                    if new_state in ("BTC","PAXG") and partial_fill_max_days > 0:
                        fill_days = min(partial_fill_max_days, N - i_end)
                        if fill_days > 0:
                            # Linear ramp  (1/fill_days, 2/fill_days, ..., 1.0)
                            for k in range(fill_days):
                                scale[i_end + k] = (k+1)/fill_days

                else:
                    # No delay; still can partial-fill if entering risk
                    new_state = eff[i_sw]
                    if new_state in ("BTC","PAXG") and partial_fill_max_days > 0:
                        fill_days = min(partial_fill_max_days, N - i_sw)
                        if fill_days > 0:
                            for k in range(fill_days):
                                scale[i_sw + k] = (k+1)/fill_days

        # Apply droughts (disable new entries while drought_left>0)
        # We implement it during the P&L loop by preventing transitions into BTC/PAXG.
        eq = 1.0
        path = np.zeros(N, dtype=float)
        cur_pos = "CASH"
        cur_scale = 0.0

        for t in range(N):
            # Start-of-day: maybe start a drought
            if drought_left <= 0 and drought_prob > 0 and rs.random() < drought_prob:
                drought_left = max(1, int(round(rs.poisson(lam=max(1e-6, drought_mean_days)))))
            # Enforce drought on state transitions into risk
            desired = eff[t]
            desired_scale = scale[t]

            if drought_left > 0:
                # If trying to enter BTC/PAXG from CASH, block and remain CASH
                if (cur_pos == "CASH") and (desired in ("BTC","PAXG")):
                    desired = "CASH"
                    desired_scale = 0.0
                drought_left -= 1

            # Apply transaction costs when state changes (entry/exit)
            entry = (cur_pos != desired) and (desired in ("BTC","PAXG"))
            exit_ = (cur_pos in ("BTC","PAXG")) and (desired != cur_pos)

            # Fee + spread + slippage (regime + ATR%-scaled)
            # Compute per-side slippage in decimal (not bps)
            def slip_bps_for(side, t):
                base = slip_in_bps_mean if side == "in" else slip_out_bps_mean
                # ATR%-based bump: choose underlying ATR% based on desired or current
                atr_pct = 0.0
                if (side == "in" and desired == "BTC") or (side == "out" and cur_pos == "BTC"):
                    atr_pct = atr_b[t]
                elif (side == "in" and desired == "PAXG") or (side == "out" and cur_pos == "PAXG"):
                    atr_pct = atr_p[t]
                # regime multiplier
                m = reg_mult[t]
                # randomization around base (normal, pos-clipped)
                rnd = max(0.0, rs.normal(base, base*0.5))
                # add ATR%-linked bump
                bump = vol_slip_k * max(0.0, float(atr_pct))  # in bps
                total_bps = (rnd + bump) * m
                # plus spread (half on entry, half on exit)
                spread_half = (spread_bps * 0.5)
                total_bps += spread_half
                return total_bps

            if entry:
                # fees + slip
                total_in_bps = fee_bps_in + slip_bps_for("in", t)
                eq *= (1.0 - total_in_bps/10000.0)
                trade_rows.append({"sim": s, "date": dates[t], "action": f"BUY_{desired}", "equity": eq})
            if exit_:
                total_out_bps = fee_bps_out + slip_bps_for("out", t)
                eq *= (1.0 - total_out_bps/10000.0)
                trade_rows.append({"sim": s, "date": dates[t], "action": f"EXIT_{cur_pos}", "equity": eq})

            # Returns (base + noise + shocks + optional weekend gaps)
            ret_bt = rb[t]
            ret_px = rp[t]

            # Base i.i.d. noise
            if ret_noise_sigma > 0:
                ret_bt += rs.normal(0.0, ret_noise_sigma)
                ret_px += rs.normal(0.0, ret_noise_sigma)

            # Shock cluster add-on
            if shock_flags[t] and shock_sigma > 0:
                ret_bt += rs.normal(0.0, shock_sigma)
                ret_px += rs.normal(0.0, shock_sigma)

            # Weekend gaps (apply only on Mondays to mimic Fri->Mon gap)
            if gap_weekend and gap_sigma > 0 and is_monday[t]:
                ret_bt += rs.normal(0.0, gap_sigma)
                ret_px += rs.normal(0.0, gap_sigma)

            # Apply position
            if desired == "BTC":
                eq *= math.exp(ret_bt * float(desired_scale))
            elif desired == "PAXG":
                eq *= math.exp(ret_px * float(desired_scale))
            # CASH => no change

            path[t] = eq
            cur_pos = desired
            cur_scale = desired_scale

        equity_paths[s, :] = path

    equity_wide = pd.DataFrame(equity_paths.T, index=dates, columns=[f"sim_{i}" for i in range(sims)])
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
    ap.add_argument("--dxy_hold_days", type=int, default=0)  # 0 => ignore DXY

    # Bear gating: BTC weakness (absolute or relative if --bear_rel_mom)
    ap.add_argument("--bear_mom_len", type=int, default=0)
    ap.add_argument("--bear_mom_thresh", type=float, default=0.0)
    ap.add_argument("--bear_rel_mom", action="store_true")

    # PAXG strength gate
    ap.add_argument("--paxg_mom_len", type=int, default=0)
    ap.add_argument("--paxg_mom_thresh", type=float, default=0.0)

    # Trailing stop controls
    ap.add_argument("--paxg_trailing_stop", action="store_true")
    ap.add_argument("--paxg_trail_atr_len", type=int, default=14)
    ap.add_argument("--paxg_trail_k", type=float, default=2.5)

    # Stop confirmation + re-entry policy
    ap.add_argument("--stop_confirm_days", type=int, default=1)
    ap.add_argument("--reenter_mode", choices=["cooldown","newhigh"], default="cooldown")
    ap.add_argument("--paxg_reenter_cooldown", type=int, default=7)
    ap.add_argument("--reenter_eps", type=float, default=0.003)

    # Robust Monte-Carlo flags
    ap.add_argument("--mc_sims", type=int, default=500)
    ap.add_argument("--mc_max_delay_days", type=int, default=3)
    ap.add_argument("--mc_delay_geom_p", type=float, default=0.5)
    ap.add_argument("--mc_partial_fill_max_days", type=int, default=2)
    ap.add_argument("--mc_miss_prob", type=float, default=0.0)
    ap.add_argument("--mc_drought_prob", type=float, default=0.0)
    ap.add_argument("--mc_drought_mean_days", type=float, default=3.0)
    ap.add_argument("--mc_fee_bps_in", type=float, default=0.0)
    ap.add_argument("--mc_fee_bps_out", type=float, default=0.0)
    ap.add_argument("--mc_spread_bps", type=float, default=0.0)
    ap.add_argument("--mc_slip_in_bps_mean", type=float, default=5.0)
    ap.add_argument("--mc_slip_out_bps_mean", type=float, default=5.0)
    ap.add_argument("--mc_slip_mult_bull", type=float, default=1.0)
    ap.add_argument("--mc_slip_mult_bear", type=float, default=1.3)
    ap.add_argument("--mc_slip_mult_choppy", type=float, default=1.15)
    ap.add_argument("--mc_vol_slip_k", type=float, default=50.0)
    ap.add_argument("--mc_ret_noise_sigma", type=float, default=0.0)
    ap.add_argument("--mc_shock_prob", type=float, default=0.0)
    ap.add_argument("--mc_shock_sigma", type=float, default=0.02)
    ap.add_argument("--mc_shock_mean_days", type=float, default=1.0)
    ap.add_argument("--mc_gap_weekend", action="store_true")
    ap.add_argument("--mc_gap_sigma", type=float, default=0.0)

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

    # DXY confirm
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
        stop_confirm_days=int(args.stop_confirm_days),
        reenter_mode=str(args.reenter_mode),
        reenter_cooldown=int(args.paxg_reenter_cooldown),
        reenter_eps=float(args.reenter_eps)
    )

    # Realized vol (ATR %) for slippage realism
    atr_len = 14
    btc_atr = atr_from_ohlc(btc, n=atr_len).reindex(dates).ffill()
    pax_atr = atr_from_ohlc(pax, n=atr_len).reindex(dates).ffill()
    btc_atr_pct = (btc_atr / btc_close).fillna(0.0)
    pax_atr_pct = (pax_atr / pax_close).fillna(0.0)

    # Robust Monte-Carlo
    summary_df, equity_wide, sim_logrets, trade_log = simulate_mc_robust(
        dates=dates,
        pos_target=pos_target,
        ret_btc=r_btc,
        ret_paxg=r_pax,
        btc_atr_pct=btc_atr_pct,
        pax_atr_pct=pax_atr_pct,
        sims=args.mc_sims,
        max_delay_days=args.mc_max_delay_days,
        delay_geom_p=args.mc_delay_geom_p,
        partial_fill_max_days=args.mc_partial_fill_max_days,
        miss_prob=args.mc_miss_prob,
        drought_prob=args.mc_drought_prob,
        drought_mean_days=args.mc_drought_mean_days,
        fee_bps_in=args.mc_fee_bps_in,
        fee_bps_out=args.mc_fee_bps_out,
        spread_bps=args.mc_spread_bps,
        slip_in_bps_mean=args.mc_slip_in_bps_mean,
        slip_out_bps_mean=args.mc_slip_out_bps_mean,
        slip_mult_bull=args.mc_slip_mult_bull,
        slip_mult_bear=args.mc_slip_mult_bear,
        slip_mult_choppy=args.mc_slip_mult_choppy,
        vol_slip_k=args.mc_vol_slip_k,
        ret_noise_sigma=args.mc_ret_noise_sigma,
        shock_prob=args.mc_shock_prob,
        shock_sigma=args.mc_shock_sigma,
        shock_mean_days=args.mc_shock_mean_days,
        gap_weekend=args.mc_gap_weekend,
        gap_sigma=args.mc_gap_sigma,
        regimes_series=regimes_series,
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
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        raise
