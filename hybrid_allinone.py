#!/usr/bin/env python3
# hybrid_allinone.py — crypto (24/7) friendly: calendar-day windows
# One-stop WF + WF2 + Hybrid generator using causal rolling thresholds.
#
# Output:
#   - hybrid_out.csv  (date, close, wf_label, wf2_label, hybrid_label, pos)
#   - wf_out.txt, wf2_out.txt         (overall summaries)
#   - wf_train.txt, wf_test.txt       (WF summaries: train<=2018-12-31, test>=2019-01-01)
#   - wf2_train.txt, wf2_test.txt     (WF2 summaries: same split)
#   - hybrid_plot.png                 (Bull=green, Bear=red, Choppy=blue)
#
# Examples:
#   python3 hybrid_allinone.py --symbol BTC-USD --start 2014-01-01 --macro_win 90 --micro_win 30
#   python3 hybrid_allinone.py --csv my_prices.csv --date_col date --close_col close --macro_win 90 --micro_win 30

import argparse, sys, math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- helpers --------------------

def _ensure_series_1d(x: pd.Series | pd.DataFrame) -> pd.Series:
    """Guarantee a 1-D float Series (fixes '(n,1)' issues)."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            raise ValueError("Expected 1 column for close, got multiple.")
    return pd.to_numeric(pd.Series(x, index=x.index), errors="coerce")

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = _ensure_series_1d(series)
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return (100 - (100 / (1 + rs))).fillna(50.0)

def _bb_width(close: pd.Series, period: int = 20, nstd: float = 2.0) -> pd.Series:
    close = _ensure_series_1d(close)
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (ma + nstd*std - (ma - nstd*std)) / ma

def _drawdown_from_peak(close: pd.Series, lookback: int = 30) -> pd.Series:
    close = _ensure_series_1d(close)
    roll_max = close.rolling(lookback, min_periods=1).max()
    return (close / roll_max) - 1.0

def _drift(close: pd.Series, win: int = 10) -> pd.Series:
    close = _ensure_series_1d(close)
    return close.pct_change(win)

def _pctl_rolling(series: pd.Series, win: int, q: float) -> pd.Series:
    series = _ensure_series_1d(series)
    def fn(x):
        return np.nanpercentile(x[:-1], q*100) if x.size>1 else np.nan
    return series.rolling(win+1, min_periods=win+1).apply(fn, raw=True)

def _wf_classify(df, pctl_win_days, chop_q, drift_win, drift_bull_min, drift_bear_max,
                 dd_lookback, rsi_bull_min, rsi_bear_max, ma_slow_days, label_name):
    close = _ensure_series_1d(df['close'])
    rsi = _rsi(close, 14)
    bbw = _bb_width(close, 20, 2.0)
    drift = _drift(close, drift_win)
    dd = _drawdown_from_peak(close, dd_lookback)
    ma_slow = close.rolling(ma_slow_days).mean()
    bbw_chop_thresh = _pctl_rolling(bbw, pctl_win_days, chop_q)

    labels = []
    for i in range(len(df)):
        if np.isnan(bbw_chop_thresh.iat[i]) or i < ma_slow_days:
            labels.append("Choppy"); continue
        is_chop = bbw.iat[i] <= bbw_chop_thresh.iat[i]
        above_ma = close.iat[i] > ma_slow.iat[i]
        if is_chop:
            labels.append("Choppy")
        else:
            if above_ma and (drift.iat[i] >= drift_bull_min or rsi.iat[i] >= rsi_bull_min):
                labels.append("Bull")
            elif (not above_ma) and (drift.iat[i] <= drift_bear_max or rsi.iat[i] <= rsi_bear_max or dd.iat[i] <= -0.03):
                labels.append("Bear")
            else:
                labels.append("Bull" if drift.iat[i] >= 0 else "Bear")
    return pd.Series(labels, index=df.index, name=label_name)

def _summarize_periods(dates: pd.Series, close: pd.Series, labels: pd.Series, title: str) -> str:
    """Human-readable regime period summary."""
    lines = [f"{title}", "="*len(title), ""]
    labels = labels.fillna("Choppy")
    start_idx = 0
    n = len(labels)
    def period_return(s, e):
        c0 = close.iat[s]; c1 = close.iat[e]
        return (c1 / c0 - 1.0) * 100.0
    counter = 1
    for i in range(1, n+1):
        if i == n or labels.iat[i] != labels.iat[start_idx]:
            end_idx = i-1
            lab = str(labels.iat[start_idx])
            d0 = dates.iat[start_idx].strftime("%Y-%m-%d")
            d1 = dates.iat[end_idx].strftime("%Y-%m-%d")
            days = (dates.iat[end_idx] - dates.iat[start_idx]).days + 1
            ret = period_return(start_idx, end_idx)
            lines.append(f"[{counter:03d}] {lab}")
            lines.append(f"     Period : {d0} -> {d1}  ({days} days)")
            lines.append(f"     Return : {ret:+.2f}%")
            lines.append("")
            counter += 1
            start_idx = i
    return "\n".join(lines)

def _plot_hybrid(date_series, close_series, hybrid_labels, out_path="hybrid_plot.png"):
    """Single-axis plot of close with colored background per hybrid label.
       Colors: Bull=green, Bear=red, Choppy=blue."""
    from matplotlib.patches import Patch
    bull_color = "green"; bear_color = "red"; chop_color = "blue"
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.plot(date_series, close_series)
    current = None; seg_start = 0
    for i in range(len(hybrid_labels)+1):
        if i==len(hybrid_labels) or hybrid_labels[i] != current:
            if current is not None:
                x0 = date_series[seg_start]; x1 = date_series[i-1]
                if current == "Bull":
                    ax.axvspan(x0, x1, color=bull_color, alpha=0.12, linewidth=0)
                elif current == "Bear":
                    ax.axvspan(x0, x1, color=bear_color, alpha=0.12, linewidth=0)
                else:
                    ax.axvspan(x0, x1, color=chop_color, alpha=0.08, linewidth=0)
            if i < len(hybrid_labels):
                current = hybrid_labels[i]; seg_start = i
    ax.set_title("Price with Hybrid Regimes")
    ax.set_xlabel("Date"); ax.set_ylabel("Close")
    ax.legend(handles=[
        Patch(facecolor="green", alpha=0.12, label="Bull"),
        Patch(facecolor="red",   alpha=0.12, label="Bear"),
        Patch(facecolor="blue",  alpha=0.08, label="Choppy"),
    ], loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--start", default="2014-01-01")           # start early so train period exists
    ap.add_argument("--end", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--date_col", default="Date")
    ap.add_argument("--close_col", default="Close")
    # Crypto 24/7: windows in calendar days
    ap.add_argument("--macro_win", type=int, default=365, help="WF window (days) for macro thresholds")
    ap.add_argument("--micro_win", type=int, default=182, help="WF2 window (days) for micro thresholds")
    ap.add_argument("--ma_macro", type=int, default=200, help="slow MA days for macro")
    ap.add_argument("--ma_micro", type=int, default=150, help="slow MA days for micro")
    # Train/Test split
    ap.add_argument("--train_end", default="2018-12-31", help="Train cutoff date (inclusive)")
    ap.add_argument("--test_start", default="2019-01-01", help="Test start date (inclusive)")
    args = ap.parse_args()

    # Load data
    if args.csv:
        df = pd.read_csv(args.csv)
        df = df.rename(columns={args.date_col:"date", args.close_col:"close"})
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
    else:
        import yfinance as yf
        data = yf.download(args.symbol, start=args.start, end=args.end, interval="1d", auto_adjust=True, progress=False)
        if data.empty:
            print("No data downloaded. Try --csv.", file=sys.stderr); sys.exit(1)
        df = data.reset_index().rename(columns={"Date":"date","Close":"close"})
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index("date").sort_index()

    # Ensure close is 1-D Series
    df['close'] = _ensure_series_1d(df['close'])

    # Generate WF (macro) and WF2 (micro) using calendar-day window sizes
    wf = _wf_classify(df, args.macro_win, 0.20, 10, 0.01, -0.01, 20, 55.0, 45.0, args.ma_macro, "wf_label")
    wf2 = _wf_classify(df, args.micro_win, 0.30, 7, 0.003, -0.003, 15, 52.0, 48.0, args.ma_micro, "wf2_label")

    # Assemble output (force 1-D)
    dates = pd.to_datetime(df.index.copy())
    close_1d = np.asarray(df['close']).reshape(-1)
    wf_1d = np.asarray(wf).reshape(-1)
    wf2_1d = np.asarray(wf2).reshape(-1)
    out = pd.DataFrame({
        "date": dates,
        "close": close_1d,
        "wf_label": wf_1d,
        "wf2_label": wf2_1d
    }).reset_index(drop=True)

    # ---------- Hybrid (WF anchor + WF2 tactical overrides) ----------
    # Features
    out['cumret10'] = _drift(pd.Series(out['close'].values, index=out['date']), 10)
    out['rsi'] = _rsi(pd.Series(out['close'].values, index=out['date']), 14)
    out['dd20'] = _drawdown_from_peak(pd.Series(out['close'].values, index=out['date']), 20)

    # Looser thresholds so overrides actually trigger
    cond_mini_bear = (out['cumret10'] <= -0.02) | (out['rsi'] <= 40) | (out['dd20'] <= -0.02)
    cond_mini_bull = (out['cumret10'] >=  0.04) | (out['rsi'] >= 60) | (out['dd20'] >= -0.005)

    # 3-day rolling confirmations to reduce whipsaws
    def _streak3(cond):
        return cond.rolling(3).apply(lambda x: 1.0 if x.all() else 0.0, raw=False).astype(bool)

    streak_bear = _streak3(cond_mini_bear & (out['wf_label']=="Bull") & (out['wf2_label']=="Bear"))
    streak_bull = _streak3(cond_mini_bull & (out['wf_label']=="Bear") & (out['wf2_label']=="Bull"))

    # Build labels
    hybrid = []
    for i in range(len(out)):
        base = out['wf_label'].iat[i]
        overlay = out['wf2_label'].iat[i]
        label = base
        if streak_bear.iat[i]:
            label = "Bear"
        elif streak_bull.iat[i]:
            label = "Bull"
        elif base == "Choppy":
            # Aggressive resolution of Chop using overlay + context
            if overlay == "Bull" and (out['rsi'].iat[i] >= 55 or out['cumret10'].iat[i] >= 0.01):
                label = "Bull"
            elif overlay == "Bear" and (out['rsi'].iat[i] <= 45 or out['cumret10'].iat[i] <= -0.01):
                label = "Bear"
            else:
                label = "Choppy"
        hybrid.append(label)

    out['hybrid_label'] = hybrid

    # Position sizing: overrides → flat; base labels keep default exposure
    pos = []
    for i, lab in enumerate(hybrid):
        base = out['wf_label'].iat[i]
        overlay = out['wf2_label'].iat[i]
        if ((base=="Bull" and overlay=="Bear") or (base=="Bear" and overlay=="Bull")) and (lab != base):
            pos.append(0.0)     # fully hedged on overrides
        else:
            if lab=="Bull": pos.append(1.0)
            elif lab=="Bear": pos.append(-0.5)
            else: pos.append(0.2)
    out['pos'] = pos

    # Save CSV (date as string for portability)
    csv_out = out.copy()
    csv_out['date'] = pd.to_datetime(csv_out['date']).dt.strftime("%Y-%m-%d")
    csv_out[['date','close','wf_label','wf2_label','hybrid_label','pos']].to_csv("hybrid_out.csv", index=False)

    # Overall WF / WF2 summaries
    all_dates = pd.to_datetime(out['date'])
    wf_txt  = _summarize_periods(all_dates, pd.Series(out['close'].values), pd.Series(out['wf_label'].values),  "WF — Regime Periods (calendar-day windows)")
    wf2_txt = _summarize_periods(all_dates, pd.Series(out['close'].values), pd.Series(out['wf2_label'].values), "WF2 — Regime Periods (calendar-day windows)")
    with open("wf_out.txt","w") as f: f.write(wf_txt)
    with open("wf2_out.txt","w") as f: f.write(wf2_txt)

    # Train/Test split files (train <= 2018-12-31, test >= 2019-01-01 by default)
    train_end_dt = pd.to_datetime(args.train_end); test_start_dt = pd.to_datetime(args.test_start)
    mask_train = pd.to_datetime(out['date']) <= train_end_dt
    mask_test  = pd.to_datetime(out['date']) >= test_start_dt

    def _safe_summary(mask, title, col):
        d = pd.to_datetime(out['date'])[mask]
        c = pd.Series(out['close'].values)[mask].reset_index(drop=True)
        l = pd.Series(out[col].values)[mask].reset_index(drop=True)
        if len(d) == 0:
            return f"{title}\n" + "="*len(title) + "\n\n(No data in this range)\n"
        return _summarize_periods(d.reset_index(drop=True), c, l, title)

    wf_train_txt  = _safe_summary(mask_train, f"WF — Train (<= {train_end_dt.date()})", "wf_label")
    wf_test_txt   = _safe_summary(mask_test,  f"WF — Test (>= {test_start_dt.date()})", "wf_label")
    wf2_train_txt = _safe_summary(mask_train, f"WF2 — Train (<= {train_end_dt.date()})", "wf2_label")
    wf2_test_txt  = _safe_summary(mask_test,  f"WF2 — Test (>= {test_start_dt.date()})", "wf2_label")

    with open("wf_train.txt","w") as f: f.write(wf_train_txt)
    with open("wf_test.txt","w") as f: f.write(wf_test_txt)
    with open("wf2_train.txt","w") as f: f.write(wf2_train_txt)
    with open("wf2_test.txt","w") as f: f.write(wf2_test_txt)

    # Visual
    _plot_hybrid(list(all_dates), list(out['close'].values), list(out['hybrid_label'].values), out_path="hybrid_plot.png")

    print(f"✅ Saved hybrid_out.csv (rows: {len(out)})")
    print("✅ Saved wf_out.txt, wf2_out.txt")
    print("✅ Saved wf_train.txt, wf_test.txt, wf2_train.txt, wf2_test.txt")
    print("✅ Saved hybrid_plot.png")
    print("Tip: use --macro_win 90 --micro_win 30 for your preferred windows.")

if __name__ == "__main__":
    main()