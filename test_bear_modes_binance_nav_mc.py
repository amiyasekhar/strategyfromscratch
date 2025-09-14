#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, io, math, json, time, datetime as dt
from dataclasses import dataclass
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

# Optional: inverse ETF via Yahoo
try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False

# ---------------- Utils

def _to_dt(x):
    if isinstance(x, pd.Timestamp): return x.normalize()
    return pd.to_datetime(x).normalize()

def _coerce_series_1d(arr, index=None, name=None):
    if isinstance(arr, pd.Series):
        s = arr.copy()
    elif isinstance(arr, pd.DataFrame):
        if arr.shape[1] == 1:
            s = arr.iloc[:, 0].copy()
        else:
            raise ValueError(f"_coerce_series_1d got DataFrame with shape {arr.shape}")
    else:
        s = pd.Series(np.asarray(arr).reshape(-1,), index=index, name=name)
    if index is not None:
        s = s.reindex(pd.Index(index))
    if name is not None:
        s.name = name
    return s.astype(float)

def _pct_to_logret(pct): return np.log1p(pct)

def _ret_to_equity(logrets, start_equity=1.0):
    lr = _coerce_series_1d(logrets)
    eq = start_equity * np.exp(lr.cumsum())
    eq.name = "equity"
    return eq

def _summary_from_equity(eq):
    daily_lr = np.log(eq).diff().fillna(0.0)
    totret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    nyrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float((eq.iloc[-1] / eq.iloc[0])**(1/nyrs) - 1.0) if nyrs > 0 else np.nan
    vol = float(daily_lr.std() * np.sqrt(365))
    sharpe = float((daily_lr.mean() / daily_lr.std()) * np.sqrt(365)) if daily_lr.std() > 0 else np.nan
    rollmax = eq.cummax()
    dd = eq / rollmax - 1.0
    maxdd = float(dd.min())
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, max_dd=maxdd, tot_ret=totret)

def _rolling_sma(s, n): return s.rolling(n, min_periods=1).mean()
def _momentum(s, n): return s / s.shift(n) - 1.0
def _is_weekend(ts): return ts.weekday() >= 5
def _mc_percentile(x, q): return float(np.percentile(np.asarray(x, dtype=float), q))

def _scalar_at(series: pd.Series, d, default=0.0) -> float:
    try:
        v = series.loc[d]
        if isinstance(v, pd.Series):
            v = v.iloc[-1]
        return float(v)
    except Exception:
        try:
            return float(series.reindex([d]).iloc[0])
        except Exception:
            return float(default)

# ---------------- Data loading

def load_regimes_csv(path):
    df = pd.read_csv(path)
    lower_cols = {c.lower(): c for c in df.columns}

    # date
    date_col = None
    for c in ["date","ts","timestamp"]:
        if c in df.columns: date_col = c; break
        if c in lower_cols: date_col = lower_cols[c]; break
    if date_col is None: raise ValueError(f"Could not find a date column in {list(df.columns)}")
    df["date"] = pd.to_datetime(df[date_col]).dt.normalize()

    # BTC close
    close_col = None
    for c in ["close","btc_close","px","price"]:
        if c in df.columns: close_col = c; break
        if c in lower_cols: close_col = lower_cols[c]; break
    if close_col is None:
        raise ValueError("Could not find BTC 'close' in CSV (looked for close/btc_close/px/price)")
    df["close"] = pd.to_numeric(df[close_col], errors="coerce")

    # regime label preference
    regime_col = None
    for c in ["hybrid_label","wf_label","wf2_label","regime","state","label"]:
        if c in df.columns: regime_col = c; break
        if c in lower_cols: regime_col = lower_cols[c]; break
    if regime_col is None:
        raise ValueError("No regime column found (hybrid_label/wf_label/wf2_label/regime/state/label)")

    reg = df[["date","close", regime_col]].rename(columns={regime_col: "regime"}).copy()
    reg["regime"] = reg["regime"].astype(str).str.strip().str.lower()
    return reg.sort_values("date").drop_duplicates("date").reset_index(drop=True)

# ---------------- Market data fetchers

BINANCE_API = "https://api.binance.com"

def fetch_binance_klines(symbol="BTCDOWNUSDT", interval="1d", start=None, end=None):
    params = {"symbol": symbol, "interval": interval, "limit": 1000}
    if start is not None:
        params["startTime"] = int(pd.Timestamp(start, tz='UTC').timestamp() * 1000)
    if end is not None:
        params["endTime"] = int(pd.Timestamp(end, tz='UTC').timestamp() * 1000)

    out = []
    while True:
        r = requests.get(BINANCE_API + "/api/v3/klines", params=params, timeout=20)
        r.raise_for_status()
        arr = r.json()
        if not arr: break
        out.extend(arr)
        if len(arr) < 1000: break
        params["startTime"] = arr[-1][0] + 1
        time.sleep(0.2)
    if not out:
        return pd.DataFrame(columns=["date","close"])

    rows = []
    for k in out:
        ts = pd.to_datetime(k[0], unit="ms", utc=True).tz_convert(None).normalize()
        rows.append((ts, float(k[4])))
    df = pd.DataFrame(rows, columns=["date","close"]).drop_duplicates("date").set_index("date").sort_index()
    return df

def fetch_inverse_etf_series(ticker: str, start, end):
    if not _HAS_YF:
        raise RuntimeError("yfinance not installed. `pip install yfinance`")
    df = yf.download(
        ticker,
        start=pd.to_datetime(start) - pd.Timedelta(days=5),
        end=pd.to_datetime(end) + pd.Timedelta(days=2),
        auto_adjust=True,
        progress=False,
        interval="1d",
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data from Yahoo for {ticker}")

    # Prefer Adj Close; fall back to Close; squeeze to 1-D in case Yahoo returns a 2-D frame.
    if isinstance(df, pd.DataFrame) and "Adj Close" in df.columns:
        px = df["Adj Close"]
    elif isinstance(df, pd.DataFrame) and "Close" in df.columns:
        px = df["Close"]
    else:
        px = df.squeeze()

    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0].squeeze()

    px = pd.Series(px, copy=True)  # ensure Series
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px.groupby(px.index).last().sort_index()
    px.name = "close"
    return pd.DataFrame({"close": px})

def fetch_btcdow_series(start, end):
    df = fetch_binance_klines("BTCDOWNUSDT", "1d", start=start, end=end)
    if df.empty:
        raise RuntimeError("No BTCDOWNUSDT data from Binance.")
    return df

# ---------------- Strategy pieces

@dataclass
class GuardParams:
    use_guard: bool = False
    sma_len: int = 200
    mom_len: int = 20
    mom_thresh: float = -0.05
    cooldown_days: int = 1
    stop_loss_cap: float = 0.30       # 30% cap for daily loss filter
    atr_len: int = 20                 # ATR proxy length on BTC
    atr_mult: float = 5.0             # 5x ATR20 cap
    stop_strikes: int = 2             # consecutive breach days to stop
    # Crash add-in (Option C)
    use_crash_addin: bool = True
    crash_lookback_days: int = 5
    crash_thresh: float = -0.10       # -10% 5d drawdown triggers entry

def _btc_atr_proxy(btc_close: pd.Series, atr_len: int) -> pd.Series:
    """
    ATR proxy using rolling mean of absolute daily pct change (we lack H/L).
    """
    daily_abs = btc_close.pct_change().abs()
    return daily_abs.rolling(atr_len, min_periods=1).mean().fillna(0.0)

def build_returns(reg_df: pd.DataFrame,
                  start, end,
                  guard: GuardParams,
                  inverse_etf_ticker: str | None):
    mask = (reg_df["date"] >= _to_dt(start)) & (reg_df["date"] <= _to_dt(end))
    reg = reg_df.loc[mask, ["date","close","regime"]].dropna(subset=["close"]).copy()
    reg["date"] = pd.to_datetime(reg["date"]).dt.normalize()
    reg = reg.drop_duplicates("date").set_index("date").sort_index()

    btc_close = reg["close"].astype(float)
    btc_logret = np.log(btc_close / btc_close.shift(1)).fillna(0.0)
    btc_logret.name = "btc_logret"

    # Bear asset series (ETF preferred if provided)
    if inverse_etf_ticker:
        etf_df = fetch_inverse_etf_series(inverse_etf_ticker, start, end)
        etf_df = etf_df[~etf_df.index.duplicated(keep="last")].sort_index()
        etf_aligned = etf_df["close"].reindex(reg.index)
        first_live = etf_df.index.min()
        etf_close = etf_aligned.ffill()
        etf_close[reg.index < first_live] = np.nan
        bear_close = etf_close
        bear_label = f"ETF({inverse_etf_ticker})"
    else:
        btcd_df = fetch_btcdow_series(start, end)
        btcd_df = btcd_df[~btcd_df.index.duplicated(keep="last")].sort_index()
        btcd_aligned = btcd_df["close"].reindex(reg.index)
        first_live = btcd_df.index.min()
        btcd_close = btcd_aligned.ffill()
        btcd_close[reg.index < first_live] = np.nan
        bear_close = btcd_close
        bear_label = "BTC3S(Binance)"

    bear_lr_series = np.log(bear_close / bear_close.shift(1))
    bear_lr_series = bear_lr_series.where(~bear_close.isna(), 0.0).fillna(0.0)
    bear_lr_series.name = "bear_lr"

    # ---- Guard + Crash add-in + Smarter stop (B + C)
    if guard.use_guard or guard.use_crash_addin:
        sma = _rolling_sma(btc_close, guard.sma_len)
        mom = _momentum(btc_close, guard.mom_len).fillna(0.0)
        gate_basic = (btc_close < sma) & (mom <= guard.mom_thresh)

        # Crash add-in (C): 5d drawdown <= crash_thresh
        if guard.use_crash_addin:
            roll_max = btc_close.rolling(guard.crash_lookback_days, min_periods=1).max()
            dd_5d = (btc_close / roll_max - 1.0).fillna(0.0)
            gate_crash = dd_5d <= guard.crash_thresh
            gate = gate_basic | gate_crash
        else:
            gate = gate_basic

        # ATR-proxy & smarter stop (B)
        atrp = _btc_atr_proxy(btc_close, guard.atr_len)  # ~daily pct
        dyn_cap = np.minimum(guard.stop_loss_cap, guard.atr_mult * atrp)  # daily pct cap
        dyn_cap_log = np.log1p(-dyn_cap.clip(0.0, 0.95))                 # negative number

        cool_left = 0
        strikes = 0
        use_mask = []
        for d in reg.index:
            if cool_left > 0:
                use_mask.append(False)
                cool_left -= 1
                strikes = 0
                continue

            if (reg.loc[d, "regime"] == "bear") and bool(gate.loc[d]):
                # consider in bear asset today
                use_today = True

                # smarter stop: if today's bear return < dyn_cap_log (i.e., worse loss) → add a strike
                ret_today = _scalar_at(bear_lr_series, d, default=0.0)
                cap_today = _scalar_at(dyn_cap_log, d, default=np.log1p(-guard.stop_loss_cap))
                if ret_today < cap_today:
                    strikes += 1
                else:
                    strikes = 0

                if strikes >= guard.stop_strikes:
                    # stop and cooldown
                    use_today = False
                    cool_left = max(1, guard.cooldown_days)
                    strikes = 0

                use_mask.append(use_today)
            else:
                use_mask.append(False)
                strikes = 0
        use_mask = pd.Series(use_mask, index=reg.index)

        bear_asset_logret = np.where(
            (reg["regime"] == "bear") & (use_mask),
            bear_lr_series, 0.0
        )
    else:
        # If no guard nor crash add-in, default to CASH in bear (0 return)
        bear_asset_logret = np.where(reg["regime"] == "bear", 0.0, 0.0)

    bear_asset_logret = _coerce_series_1d(bear_asset_logret, index=reg.index, name="bear_asset_logret")
    strat_logret = np.where(reg["regime"] == "bear", bear_asset_logret, btc_logret)
    strat_logret = _coerce_series_1d(strat_logret, index=reg.index, name="strat_logret")

    return reg.index, strat_logret, btc_logret, reg["regime"], btc_close, bear_asset_logret, bear_label

# ---------------- Monte Carlo

@dataclass
class MCParams:
    sims: int = 1000
    max_delay_days: int = 3
    delay_geom_p: float = 0.5
    partial_fill_max_days: int = 2
    miss_prob: float = 0.01
    drought_prob: float = 0.005
    drought_mean_days: int = 3
    fee_bps_in: float = 4.0
    fee_bps_out: float = 4.0
    spread_bps: float = 2.0
    slip_in_bps_mean: float = 5.0
    slip_out_bps_mean: float = 5.0
    slip_mult_bull: float = 1.0
    slip_mult_bear: float = 1.3
    slip_mult_choppy: float = 1.15
    vol_slip_k: float = 50.0
    ret_noise_sigma: float = 0.001
    shock_prob: float = 0.01
    shock_sigma: float = 0.02
    shock_mean_days: int = 1
    gap_weekend: bool = True
    gap_sigma: float = 0.01

def simulate_mc(idx, strat_logret, regimes, btc_logret, mc: MCParams):
    rng = np.random.default_rng(42)
    idx = pd.DatetimeIndex(idx)
    base = _coerce_series_1d(strat_logret, index=idx, name="base_logret")
    regimes = pd.Series(regimes, index=idx).astype(str).str.lower()

    vol_d = _coerce_series_1d(btc_logret, index=idx).rolling(20, min_periods=1).std().fillna(0.0)
    reg_mult = regimes.map({"bull": mc.slip_mult_bull, "bear": mc.slip_mult_bear, "choppy": mc.slip_mult_choppy}).fillna(1.0)

    paths, all_logrets = [], []
    reg_change = regimes.ne(regimes.shift(1)).fillna(False)

    def draw_delay():
        d = rng.geometric(mc.delay_geom_p) - 1
        return int(min(d, mc.max_delay_days))

    for _ in range(mc.sims):
        lr = base.copy()

        miss_mask = rng.random(len(idx)) < mc.miss_prob
        lr[miss_mask] = 0.0

        drought_mask = pd.Series(False, index=idx)
        i = 0
        while i < len(idx):
            if rng.random() < mc.drought_prob:
                dlen = max(1, int(rng.poisson(mc.drought_mean_days)))
                drought_mask.iloc[i:i+dlen] = True
                i += dlen
            else:
                i += 1
        lr[drought_mask] = 0.0

        for d in idx[reg_change]:
            delay = draw_delay()
            if delay > 0:
                lr.loc[d] = 0.0
            pf = rng.integers(0, mc.partial_fill_max_days + 1)
            if pf > 0:
                share = 1.0 / (pf + 1)
                for k in range(pf + 1):
                    t = d + pd.Timedelta(days=k)
                    if t in lr.index:
                        lr.loc[t] *= share

        turn_cost = (mc.fee_bps_in + mc.fee_bps_out + mc.spread_bps) / 1e4
        lr.loc[idx[reg_change]] += np.log1p(-turn_cost)

        slip_bps = mc.slip_in_bps_mean * reg_mult + mc.vol_slip_k * vol_d
        slip = (slip_bps / 1e4).clip(lower=0.0)
        lr += np.log1p(-slip)

        lr += rng.normal(0.0, mc.ret_noise_sigma, size=len(lr))

        shock_mask = rng.random(len(idx)) < mc.shock_prob
        shock = rng.normal(0.0, mc.shock_sigma, size=len(idx))
        lr[shock_mask] += shock[shock_mask]

        if mc.gap_weekend:
            wk_mask = pd.Series([_is_weekend(d) for d in idx], index=idx)
            gap = rng.normal(0.0, mc.gap_sigma, size=len(idx))
            lr[wk_mask] += gap[wk_mask]

        eq = _ret_to_equity(lr, 1.0)
        paths.append(eq)
        all_logrets.append(lr)

    equity_wide = pd.concat(paths, axis=1)
    equity_wide.columns = [f"sim_{i:04d}" for i in range(len(paths))]
    sim_logrets = pd.concat(all_logrets, axis=1)
    sim_logrets.columns = equity_wide.columns

    end_vals = equity_wide.iloc[-1, :].values
    start_vals = equity_wide.iloc[0, :].values
    ny = (equity_wide.index[-1] - equity_wide.index[0]).days / 365.25
    cagrs = (end_vals / start_vals)**(1/ny) - 1.0

    vols, sharpes, maxdds, totrets = [], [], [], []
    for c in equity_wide.columns:
        s = _summary_from_equity(equity_wide[c])
        vols.append(s["vol"]); sharpes.append(s["sharpe"]); maxdds.append(s["max_dd"]); totrets.append(s["tot_ret"])

    summary = {
        "CAGR_med": float(np.median(cagrs)),
        "CAGR_p10": _mc_percentile(cagrs, 10),
        "CAGR_p90": _mc_percentile(cagrs, 90),
        "Vol_med": float(np.median(vols)),
        "Vol_p10": _mc_percentile(vols, 10),
        "Vol_p90": _mc_percentile(vols, 90),
        "Sharpe_med": float(np.median(sharpes)),
        "Sharpe_p10": _mc_percentile(sharpes, 10),
        "Sharpe_p90": _mc_percentile(sharpes, 90),
        "MaxDD_med": float(np.median(maxdds)),
        "MaxDD_p10": _mc_percentile(maxdds, 10),
        "MaxDD_p90": _mc_percentile(maxdds, 90),
        "TotRet_med": float(np.median(totrets)),
        "TotRet_p10": _mc_percentile(totrets, 10),
        "TotRet_p90": _mc_percentile(totrets, 90),
        "sims": int(len(paths)),
    }
    return summary, equity_wide, sim_logrets

# ---------------- Bear windows

def summarize_bear_windows(idx, regimes, btc_logret, bear_asset_logret, strat_logret):
    idx = pd.DatetimeIndex(idx)
    reg = pd.Series(regimes, index=idx)
    r_btc  = _coerce_series_1d(btc_logret, index=idx)
    r_bear = _coerce_series_1d(bear_asset_logret, index=idx)
    r_str  = _coerce_series_1d(strat_logret, index=idx)

    rows, in_bear, start = [], False, None
    for d in idx:
        if (reg.loc[d] == "bear") and not in_bear:
            in_bear, start = True, d
        if in_bear and ((reg.loc[d] != "bear") or (d == idx[-1])):
            end = d if reg.loc[d] != "bear" else d
            span = pd.date_range(start, end, freq="D")
            btc_ret   = float(np.exp(r_btc.reindex(span).fillna(0).sum()) - 1.0)
            bear_ret  = float(np.exp(r_bear.reindex(span).fillna(0).sum()) - 1.0)
            strat_ret = float(np.exp(r_str.reindex(span).fillna(0).sum()) - 1.0)
            rows.append([start.date(), end.date(), len(span), btc_ret, bear_ret, strat_ret])
            in_bear, start = False, None
    return pd.DataFrame(rows, columns=["start","end","days","btc_ret","bear_asset_ret","strat_ret"])

# ---------------- Plots

def plot_equity_vs_btc(idx, strat_lr, btc_lr, out_png):
    eq_s = _ret_to_equity(strat_lr)
    eq_b = _ret_to_equity(btc_lr)
    plt.figure(figsize=(10,5))
    plt.plot(eq_s.index, eq_s.values, label="Strategy (MC median path not shown here)")
    plt.plot(eq_b.index, eq_b.values, label="BTC HODL")
    plt.yscale("log")
    plt.legend()
    plt.title("Equity Curves (log scale)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_regimes(idx, regimes, out_png):
    plt.figure(figsize=(10,1.6))
    reg = pd.Series(regimes, index=idx)
    color_map = {"bull":"#28a745","bear":"#dc3545","choppy":"#ffc107"}
    c = reg.map(color_map).fillna("#999999")
    plt.bar(reg.index, np.ones(len(reg)), width=1.0, color=c, align="center")
    plt.yticks([])
    plt.title("Regimes")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ---------------- Main

def main():
    p = argparse.ArgumentParser(description="Hybrid rotation with Bear asset (Inverse ETF or Binance BTC3S) + Monte Carlo")
    p.add_argument("--hybrid_csv", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out_prefix", required=True)

    p.add_argument("--inverse_etf_ticker", default=None, help="Yahoo Finance ticker (e.g., BITI). If omitted, uses Binance BTCDOWNUSDT")

    # Guard options (with B + C defaults ON)
    p.add_argument("--use_btc3s_guard", action="store_true", default=True,
                   help="Enable SMA/momentum gate + smarter stop + crash add-in")
    p.add_argument("--guard_sma_len", type=int, default=200)
    p.add_argument("--guard_mom_len", type=int, default=20)
    p.add_argument("--guard_mom_thresh", type=float, default=-0.05)
    p.add_argument("--guard_cooldown_days", type=int, default=1)

    # Smarter stop (B)
    p.add_argument("--stop_loss_cap", type=float, default=0.30, help="Cap for daily loss check (e.g., 0.30 = 30%)")
    p.add_argument("--atr_len", type=int, default=20, help="ATR proxy window on BTC")
    p.add_argument("--atr_mult", type=float, default=5.0, help="ATR multiple for dynamic daily loss cap")
    p.add_argument("--stop_strikes", type=int, default=2, help="Consecutive breach days to trigger stop")

    # Crash add-in (C)
    p.add_argument("--use_crash_addin", action="store_true", default=True, help="Also enter on fast 5d drawdown")
    p.add_argument("--crash_lookback_days", type=int, default=5)
    p.add_argument("--crash_thresh", type=float, default=-0.10)

    # Monte Carlo knobs
    p.add_argument("--mc_sims", type=int, default=1000)
    p.add_argument("--mc_max_delay_days", type=int, default=3)
    p.add_argument("--mc_delay_geom_p", type=float, default=0.5)
    p.add_argument("--mc_partial_fill_max_days", type=int, default=2)
    p.add_argument("--mc_miss_prob", type=float, default=0.01)
    p.add_argument("--mc_drought_prob", type=float, default=0.005)
    p.add_argument("--mc_drought_mean_days", type=int, default=3)
    p.add_argument("--mc_fee_bps_in", type=float, default=4)
    p.add_argument("--mc_fee_bps_out", type=float, default=4)
    p.add_argument("--mc_spread_bps", type=float, default=2)
    p.add_argument("--mc_slip_in_bps_mean", type=float, default=5)
    p.add_argument("--mc_slip_out_bps_mean", type=float, default=5)
    p.add_argument("--mc_slip_mult_bull", type=float, default=1.0)
    p.add_argument("--mc_slip_mult_bear", type=float, default=1.3)
    p.add_argument("--mc_slip_mult_choppy", type=float, default=1.15)
    p.add_argument("--mc_vol_slip_k", type=float, default=50)
    p.add_argument("--mc_ret_noise_sigma", type=float, default=0.001)
    p.add_argument("--mc_shock_prob", type=float, default=0.01)
    p.add_argument("--mc_shock_sigma", type=float, default=0.02)
    p.add_argument("--mc_shock_mean_days", type=int, default=1)
    p.add_argument("--mc_gap_weekend", action="store_true")
    p.add_argument("--mc_gap_sigma", type=float, default=0.01)

    args = p.parse_args()

    reg_df = load_regimes_csv(args.hybrid_csv)

    guard = GuardParams(
        use_guard = bool(args.use_btc3s_guard),
        sma_len = int(args.guard_sma_len),
        mom_len = int(args.guard_mom_len),
        mom_thresh = float(args.guard_mom_thresh),
        cooldown_days = int(args.guard_cooldown_days),
        stop_loss_cap = float(args.stop_loss_cap),
        atr_len = int(args.atr_len),
        atr_mult = float(args.atr_mult),
        stop_strikes = int(args.stop_strikes),
        use_crash_addin = bool(args.use_crash_addin),
        crash_lookback_days = int(args.crash_lookback_days),
        crash_thresh = float(args.crash_thresh)
    )

    idx, strat_logret, btc_logret, regimes, btc_close, bear_asset_logret, bear_label = build_returns(
        reg_df, args.start, args.end, guard, args.inverse_etf_ticker
    )

    mc = MCParams(
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
        gap_weekend=bool(args.mc_gap_weekend),
        gap_sigma=args.mc_gap_sigma
    )

    summary_mc, equity_wide, sim_lr = simulate_mc(idx, strat_logret, regimes, btc_logret, mc)

    prefix = args.out_prefix

    # Summary CSV
    summ_df = pd.DataFrame({
        "metric": ["CAGR","Vol","Sharpe","MaxDD","TotRet","Sims"],
        "median": [summary_mc["CAGR_med"], summary_mc["Vol_med"], summary_mc["Sharpe_med"],
                   summary_mc["MaxDD_med"], summary_mc["TotRet_med"], summary_mc["sims"]],
        "p10":    [summary_mc["CAGR_p10"], summary_mc["Vol_p10"], summary_mc["Sharpe_p10"],
                   summary_mc["MaxDD_p10"], summary_mc["TotRet_p10"], ""],
        "p90":    [summary_mc["CAGR_p90"], summary_mc["Vol_p90"], summary_mc["Sharpe_p90"],
                   summary_mc["MaxDD_p90"], summary_mc["TotRet_p90"], ""],
    })
    summ_df.to_csv(f"{prefix}_summary.csv", index=False)

    # Equity paths CSV
    equity_wide.to_csv(f"{prefix}_equity_paths.csv")

    # Regimes CSV
    reg_out = pd.DataFrame({
        "date": pd.DatetimeIndex(idx).date,
        "regime": pd.Series(regimes, index=idx).values,
        "btc_logret": _coerce_series_1d(btc_logret, index=idx).values,
        "bear_asset_logret": _coerce_series_1d(bear_asset_logret, index=idx).values,
        "strat_logret": _coerce_series_1d(strat_logret, index=idx).values
    })
    reg_out.to_csv(f"{prefix}_regimes.csv", index=False)

    # Plots
    plot_equity_vs_btc(idx, strat_logret, btc_logret, f"{prefix}_equity_vs_btc.png")
    plot_regimes(idx, regimes, f"{prefix}_regimes.png")

    # Bear windows
    bear_win = summarize_bear_windows(idx, regimes, btc_logret, bear_asset_logret, strat_logret)
    bear_win.to_csv(f"{prefix}_bear_windows_summary.csv", index=False)

    # Console summary
    print("\n=== Monte-Carlo summary (median [p10..p90]) ===")
    print(f"CAGR    : {summary_mc['CAGR_med']:+.2%}  [{summary_mc['CAGR_p10']:+.2%} .. {summary_mc['CAGR_p90']:+.2%}]")
    print(f"Vol     : {summary_mc['Vol_med']*100:.2f}%   [{summary_mc['Vol_p10']*100:.2f}% .. {summary_mc['Vol_p90']*100:.2f}%]")
    print(f"Sharpe  : {summary_mc['Sharpe_med']:.2f}   [{summary_mc['Sharpe_p10']:.2f} .. {summary_mc['Sharpe_p90']:.2f}]")
    print(f"MaxDD   : {summary_mc['MaxDD_med']:.2%}   [{summary_mc['MaxDD_p10']:.2%} .. {summary_mc['MaxDD_p90']:.2%}]")
    print(f"TotRet  : {summary_mc['TotRet_med']:+.2%}  [{summary_mc['TotRet_p10']:+.2%} .. {summary_mc['TotRet_p90']:+.2%}]")
    print(f"Bear asset used: {('ETF('+args.inverse_etf_ticker+')' if args.inverse_etf_ticker else 'BTC3S(Binance)')} (guard/CASH logic {'ON' if guard.use_guard else 'OFF'}; crash add-in {'ON' if guard.use_crash_addin else 'OFF'})")
    print("\n[OK] Wrote:")
    print(f"  {prefix}_summary.csv")
    print(f"  {prefix}_equity_paths.csv")
    print(f"  {prefix}_regimes.csv")
    print(f"  {prefix}_equity_vs_btc.png")
    print(f"  {prefix}_regimes.png")
    print(f"  {prefix}_bear_windows_summary.csv")

if __name__ == "__main__":
    main()