#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wf2.py — Walk-forward HMM regime classifier for BTC-USD

Key:
- Initial train: 2014-01-01..2018-12-31
- Walk-forward decode on TEST 2019+ with re-train every --walk_days (expanding window)
- State→Label mapping from TRAIN forward returns (H_MAP days)
- Causal decode (no future info from TEST segment used for a given block)
- Adaptive guardrails, strict choppy detection + hysteresis, causal min-persist

Outputs:
- CSV with columns: ts, close, LogReturn, Label_rt, Why_rt, state
- TXT with contiguous periods and returns
- Optional plot

Run:
python3 wf2.py --walk_days 90 --test_start 2019-01-01 --test_end 2025-12-31
"""

import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler

# -------------------
# Defaults / config
# -------------------
TRAIN_START = "2014-01-01"
TRAIN_END   = "2018-12-31"
TEST_START  = "2019-01-01"
TEST_END    = "2025-12-31"  # or None
N_STATES    = 3
SEED        = 42

MAP_USING_FORWARD = True
H_MAP             = 5      # forward map horizon (days)

# Decoding stickiness (causal)
BASE_KAPPA          = 6.0
ALPHA_CHOPPY        = 1.0
ALPHA_TREND_LOOSEN  = 1.0
DEBOUNCE_MIN_RUN    = 10
MIN_PERSIST_DAYS    = 8
GUARD_MIN_HOLD      = 5

# Guardrails (base; adaptive later)
GUARD_ADX_MIN_BASE  = 20.0
RSI_BULL_BASE       = 60.0
RSI_BEAR_BASE       = 45.0
DRIFT_OVR_ABS_BASE  = 0.002
DRIFT_OVR_WIN       = 10
DRIFT_OVR_FRAC      = 0.7
RANGE_ADX_MAX       = 18.0
RANGE_DRIFT_MAXABS  = 0.001
RANGE_DWIDTH_PCTL   = 0.30
VOL_PCTRANK_WIN     = 120
CMF_LEN             = 20
OBV_SLOPE_WIN       = 10
VOL_MIN_PCT         = 0.40
CHOPPY_ROLL_DAYS    = 120
CHOPPY_PCTL         = 0.25
CHOP_ENTER_K        = 3
CHOP_EXIT_K         = 2

FEATS = [
    "RealVol20","ATRp","Drift_10","Drift_50","Drift_200","ADX14","KAMA_dev","MA200z",
    "DonchianW","DonchianPos","RSI14","MACD_H","MACD_H_slope3","BBWidth20",
    "OBV_slope10","CMF20","VolPctRank120"
]

np.set_printoptions(suppress=True, floatmode="fixed", linewidth=140)
pd.set_option("display.width", 160)

# -------------------
# Data (Yahoo Finance)
# -------------------
def fetch_btc_yf(start, end):
    import yfinance as yf
    df = yf.download("BTC-USD", start=start, end=end, interval="1d",
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("Yahoo Finance returned empty BTC-USD.")
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open","high","low","close","volume"]].replace([np.inf,-np.inf], np.nan).dropna()
    return df.sort_index()

# -------------------
# Indicators / features
# -------------------
def ema(x, span): return x.ewm(span=span, adjust=False, min_periods=span).mean()

def true_range(h,l,c):
    pc = c.shift(1)
    return pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def adx(h,l,c,length=14):
    up = h.diff(); down = -l.diff()
    upv = np.asarray(up).ravel(); dnv = np.asarray(down).ravel()
    plus_dm  = np.where((upv>dnv)&(upv>0), upv, 0.0)
    minus_dm = np.where((dnv>upv)&(dnv>0), dnv, 0.0)
    tr   = true_range(h,l,c)
    atr  = tr.rolling(length).mean()
    p_dm = pd.Series(plus_dm,  index=c.index)
    m_dm = pd.Series(minus_dm, index=c.index)
    p_di = 100 * p_dm.rolling(length).mean() / atr
    m_di = 100 * m_dm.rolling(length).mean() / atr
    dx   = (100*(p_di-m_di).abs()/(p_di+m_di)).replace([np.inf,-np.inf], np.nan)
    return dx.rolling(length).mean()

def kama(price, er_len=10, fast=2, slow=30):
    if isinstance(price, pd.DataFrame): price = price.iloc[:, 0]
    price = pd.Series(np.asarray(price).ravel(), index=price.index)
    change = price.diff(er_len).abs()
    vol    = price.diff().abs().rolling(er_len).sum()
    er     = (change/vol).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    sc     = (er*(2/(fast+1)-2/(slow+1)) + 2/(slow+1))**2
    out    = pd.Series(index=price.index, dtype=float)
    out.iloc[0] = price.iloc[0]
    for i in range(1, len(price)):
        val = sc.iloc[i]
        sci = float(val) if np.isfinite(val) else 0.0
        out.iloc[i] = out.iloc[i-1] + sci * (price.iloc[i] - out.iloc[i-1])
    return out

def donchian(h,l,n=20):
    upper = h.rolling(n).max(); lower = l.rolling(n).min()
    width = (upper-lower).div(lower.replace(0, np.nan)).fillna(0.0)
    rng   = (upper-lower).replace(0, np.nan)
    pos   = (((h+l)/2)-lower).div(rng).clip(0,1).fillna(0.5)
    return upper, lower, width, pos

def rsi(price, length=14):
    delta = price.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    roll_dn = dn.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(price, fast=12, slow=26, signal=9):
    ema_fast = ema(price, fast); ema_slow = ema(price, slow)
    line = ema_fast - ema_slow
    sig  = ema(line, signal)
    hist = line - sig
    slope3 = hist.diff().rolling(3).mean()
    return line, sig, hist, slope3

def bollinger_width(price, n=20, k=2.0):
    ma = price.rolling(n).mean(); sd = price.rolling(n).std()
    upper = ma + k*sd; lower = ma - k*sd
    width = (upper - lower) / (ma.replace(0, np.nan))
    return width.fillna(0.0)

def obv(close, volume):
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume).cumsum()

def obv_slope(x, win=10):
    idx = np.arange(win, dtype=float)
    def _s(y):
        if np.any(~np.isfinite(y)): return np.nan
        xbar = idx.mean(); ybar = y.mean()
        num = np.sum((idx - xbar) * (y - ybar)); den = np.sum((idx - xbar)**2) + 1e-12
        return num / den
    return pd.Series(x).rolling(win).apply(_s, raw=True)

def cmf(high, low, close, volume, n=20):
    c = close if isinstance(close, pd.Series) else pd.Series(np.asarray(close).ravel())
    idx = c.index
    h = high if isinstance(high, pd.Series) else pd.Series(np.asarray(high).ravel(), index=idx)
    l = low  if isinstance(low,  pd.Series) else pd.Series(np.asarray(low).ravel(),  index=idx)
    v = volume if isinstance(volume, pd.Series) else pd.Series(np.asarray(volume).ravel(), index=idx)
    denom = (h - l).replace(0, np.nan)
    mfm = (((c - l) - (h - c)) / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mfv = (mfm.astype(float) * v.astype(float))
    num = mfv.rolling(n).sum()
    den = v.rolling(n).sum().replace(0, np.nan)
    return (num / den).fillna(0.0)

def pct_rank(series, win=120):
    return series.rolling(win, min_periods=max(20, int(win*0.25)))\
                 .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

def build_features(df):
    # Force 1-D Series to avoid DataFrame assignment errors
    px  = pd.Series(np.asarray(df["close"]).ravel(),  index=df.index)
    hi  = pd.Series(np.asarray(df["high"]).ravel(),   index=df.index)
    lo  = pd.Series(np.asarray(df["low"]).ravel(),    index=df.index)
    vol = pd.Series(np.asarray(df["volume"]).ravel(), index=df.index)

    out = pd.DataFrame(index=df.index)
    out["close"]     = px
    out["LogReturn"] = np.log(px/px.shift(1))

    # ATR% as Series
    tr_ser = true_range(hi, lo, px)                        # Series
    atr    = tr_ser.rolling(14, min_periods=1).mean()       # Series
    out["ATRp"] = (atr / px.replace(0, np.nan)).fillna(0.0)

    # Rest
    out["RealVol20"] = out["LogReturn"].rolling(20).std().fillna(0.0)
    ew5, ew20, ew50, ew200 = ema(px,5), ema(px,20), ema(px,50), ema(px,200)
    out["EWMA5"]=ew5; out["EWMA20"]=ew20; out["EWMA50"]=ew50; out["EWMA200"]=ew200

    out["Drift_10"]  = (ew5  - ew20).div(ew20).replace([np.inf,-np.inf], 0).fillna(0.0)
    out["Drift_50"]  = (ew20 - ew50).div(ew50).replace([np.inf,-np.inf], 0).fillna(0.0)
    out["Drift_200"] = (ew50 - ew200).div(ew200).replace([np.inf,-np.inf], 0).fillna(0.0)

    out["ADX14"]   = adx(hi, lo, px, 14).fillna(0.0)
    kser           = kama(px)
    out["KAMA_dev"]= (px - kser).div(kser.replace(0,np.nan)).fillna(0.0)
    std200         = px.rolling(200).std().replace(0,np.nan)
    out["MA200z"]  = (px - ew200).div(std200).fillna(0.0)

    _,_,dwidth,dpos  = donchian(hi,lo,20)
    out["DonchianW"]   = dwidth.fillna(0.0)
    out["DonchianPos"] = dpos.fillna(0.5)

    out["RSI14"]       = rsi(px, 14).fillna(50.0)
    _, _, macd_hist, macd_slope3 = macd(px, 12, 26, 9)
    out["MACD_H"]        = macd_hist.fillna(0.0)
    out["MACD_H_slope3"] = macd_slope3.fillna(0.0)
    out["BBWidth20"]     = bollinger_width(px, 20, 2.0)

    obv_s = obv(px, vol)
    out["OBV_slope10"]   = obv_slope(obv_s, OBV_SLOPE_WIN)
    out["CMF20"]         = cmf(hi, lo, px, vol, CMF_LEN)
    out["VolPctRank120"] = pct_rank(vol, VOL_PCTRANK_WIN).fillna(0.0)

    # Vol percentile for adaptive thresholds
    out["VolPctl252"]    = pct_rank(out["RealVol20"], 252).clip(0,1).fillna(0.0)

    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    # keep last ~200+ to reduce warmup if huge
    if len(out) > 250: out = out.iloc[200:]
    return out

# -------------------
# HMM & causal decoder
# -------------------
def fit_hmm(X, n_states=N_STATES, seed=SEED):
    hmm = GaussianHMM(n_components=n_states, covariance_type="diag",
                      n_iter=200, random_state=seed, tol=1e-4)
    hmm.fit(X); return hmm

def _diag_var(cov_s):
    cov_s = np.asarray(cov_s)
    if cov_s.ndim == 1: var = cov_s
    elif cov_s.ndim == 2: var = np.diag(cov_s)
    else:
        var = np.squeeze(cov_s)
        if var.ndim == 2: var = np.diag(var)
    return np.clip(var.astype(float), 1e-8, np.inf)

def sticky_transmat(A, kappa):
    A = np.asarray(A, dtype=float).copy()
    S = A.shape[0]
    if S == 0: return A
    if (A <= 0).any() or not np.isfinite(A).all():
        A = np.ones_like(A) / S
    A = A + np.eye(S)*kappa
    A = (A.T / A.sum(axis=1)).T
    return A

def _minmax(x):
    x = np.asarray(x, dtype=float)
    a=np.nanmin(x); b=np.nanmax(x)
    if not np.isfinite(a) or not np.isfinite(b) or b-a<1e-12: return np.zeros_like(x)
    return (x-a)/(b-a)

def score_blocks(df):
    choppy_score = np.clip(np.nanmean(np.vstack([
        1-_minmax(df["LogReturn"].abs().values),
        1-_minmax(df["ADX14"].values),
        1-_minmax(df["DonchianW"].values)
    ]), axis=0),0,1)
    trend_score = np.clip(np.nanmean(np.vstack([
        _minmax(np.abs(df["Drift_10"].values)),
        _minmax(np.abs(df["Drift_50"].values)),
        _minmax(df["KAMA_dev"].values),
        _minmax(df["ADX14"].values),
        _minmax(df["DonchianW"].values),
    ]), axis=0),0,1)
    return choppy_score, trend_score

def decode_causal_dynamic(hmm, X, choppy_score, trend_score):
    S, T = hmm.n_components, X.shape[0]
    logprob_t = np.zeros((T,S))
    for s in range(S):
        mean = np.asarray(hmm.means_[s]).reshape(-1)
        var  = _diag_var(hmm.covars_[s]).reshape(-1)
        diff = X - mean
        const = np.sum(np.log(2*np.pi*var))
        quad  = np.sum((diff**2)/var, axis=1)
        logprob_t[:, s] = -0.5*(const + quad)
    log_alpha = np.zeros((T,S))
    log_pi    = np.log(np.maximum(hmm.startprob_, 1e-12))
    log_alpha[0] = log_pi + logprob_t[0]
    backptr = np.zeros((T,S), dtype=int)
    for t in range(1,T):
        kappa_t = BASE_KAPPA * (1 + ALPHA_CHOPPY*choppy_score[t]) * (1 - ALPHA_TREND_LOOSEN*trend_score[t])
        kappa_t = max(0.0, float(kappa_t))
        A_t     = sticky_transmat(hmm.transmat_, kappa_t)
        log_A_t = np.log(np.maximum(A_t, 1e-12))
        prev    = log_alpha[t-1][:,None] + log_A_t
        backptr[t] = np.argmax(prev, axis=0)
        log_alpha[t] = prev.max(axis=0) + logprob_t[t]
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(log_alpha[-1])
    for t in range(T-2,-1,-1):
        states[t] = backptr[t+1, states[t+1]]
    return states

# -------------------
# State→Label mapping
# -------------------
def state_label_mapper_from_train_forward(posteriors, close_series, h, state_count):
    close_1d = close_series.iloc[:, 0] if isinstance(close_series, pd.DataFrame) else close_series
    fwd = np.log(close_1d.shift(-h) / close_1d)

    P_full = np.asarray(posteriors, dtype=float)
    valid = fwd.notna().to_numpy(dtype=bool).reshape(-1)
    P = P_full[valid]
    y = fwd.to_numpy(dtype=float).reshape(-1)[valid]

    row_ok = (np.isfinite(y) & np.all(np.isfinite(P), axis=1)).astype(bool)
    if not np.any(row_ok):
        raise ValueError("No finite rows for forward-mapping.")
    P = P[row_ok]; y = y[row_ok]

    mu_f = {}
    for s in range(int(state_count)):
        w = P[:, s]
        denom = float(w.sum()) + 1e-12
        mu_f[s] = float(np.dot(w, y) / denom)

    ranked = sorted(mu_f.items(), key=lambda kv: kv[1])  # low → high
    s_bear, _ = ranked[0]
    s_mid,  _ = ranked[len(ranked)//2]
    s_bull, _ = ranked[-1]
    mapping = {s_bear: "Bear", s_mid: "Choppy", s_bull: "Bull"}
    return lambda s: mapping[int(s)]

def apply_hysteresis(bool_series, k_enter=CHOP_ENTER_K, k_exit=CHOP_EXIT_K):
    raw = bool_series.astype(bool).values
    out = np.zeros_like(raw, dtype=bool)
    in_chop = False; below_run = 0; above_run = 0
    for i, flag in enumerate(raw):
        if not in_chop:
            if flag:
                below_run += 1
                if below_run >= k_enter:
                    in_chop = True; out[i] = True
            else:
                below_run = 0
        else:
            if not flag:
                above_run += 1
                if above_run >= k_exit:
                    in_chop = False; out[i] = False; below_run = 0
                else:
                    out[i] = True
            else:
                out[i] = True; above_run = 0
    return pd.Series(out, index=bool_series.index)

# ---------- Adaptive guardrails ----------
def trend_guardrail_adaptive(df):
    volp = df["VolPctl252"].clip(0,1).fillna(0.0)
    adx_min_dyn  = GUARD_ADX_MIN_BASE + 10.0*volp
    rsi_bull_dyn = RSI_BULL_BASE  + 5.0*volp
    rsi_bear_dyn = RSI_BEAR_BASE  - 5.0*volp
    drift_abs_dyn= DRIFT_OVR_ABS_BASE * (1.0 + 0.75*volp)

    df = df.copy()
    df["ADXminDyn"]   = adx_min_dyn
    df["RSIbullDyn"]  = rsi_bull_dyn
    df["RSIbearDyn"]  = rsi_bear_dyn
    df["DriftAbsDyn"] = drift_abs_dyn

    adx_ok   = df["ADX14"] >= df["ADXminDyn"]
    cs = df["close"]; cs = cs.iloc[:, 0] if isinstance(cs, pd.DataFrame) else cs
    ew = df["EWMA200"];  ew = ew.iloc[:, 0] if isinstance(ew, pd.DataFrame) else ew
    above    = cs > ew
    below    = cs < ew
    rsi_ok_bull = df["RSI14"] >= df["RSIbullDyn"]
    rsi_ok_bear = df["RSI14"] <= df["RSIbearDyn"]
    macd_up  = (df["MACD_H"] > 0) & (df["MACD_H_slope3"] > 0)
    macd_dn  = (df["MACD_H"] < 0) & (df["MACD_H_slope3"] < 0)
    cmf = df["CMF20"]; obv_s = df["OBV_slope10"]; volp120 = df["VolPctRank120"]
    bull_votes = (cmf > 0).astype(int) + (obv_s > 0).astype(int) + (volp120 >= VOL_MIN_PCT).astype(int)
    bear_votes = (cmf < 0).astype(int) + (obv_s < 0).astype(int) + (volp120 >= VOL_MIN_PCT).astype(int)
    bull_vol_ok = bull_votes >= 2
    bear_vol_ok = bear_votes >= 2

    bf = adx_ok & above & rsi_ok_bull & macd_up  & bull_vol_ok
    be = adx_ok & below & rsi_ok_bear & macd_dn  & bear_vol_ok

    # causal min-hold for forced signals
    bf = bf.copy(); be = be.copy()
    last=None; hold=0
    for i in range(len(df)):
        if last=="Bull":
            if hold < GUARD_MIN_HOLD-1 and not bf.iloc[i]:
                bf.iloc[i]=True; be.iloc[i]=False; hold+=1; continue
        if last=="Bear":
            if hold < GUARD_MIN_HOLD-1 and not be.iloc[i]:
                be.iloc[i]=True; bf.iloc[i]=False; hold+=1; continue
        if bf.iloc[i] and not be.iloc[i]: last,hold="Bull",1
        elif be.iloc[i] and not bf.iloc[i]: last,hold="Bear",1
        else: last,hold=None,0

    return bf, be, df["DriftAbsDyn"]

def causal_lowvol_flag(df):
    rv = df["RealVol20"]; bb = df["BBWidth20"]
    rv_th = rv.rolling(CHOPPY_ROLL_DAYS, min_periods=max(20, int(CHOPPY_ROLL_DAYS*0.25))).quantile(CHOPPY_PCTL).shift(1)
    bb_th = bb.rolling(CHOPPY_ROLL_DAYS, min_periods=max(20, int(CHOPPY_ROLL_DAYS*0.25))).quantile(CHOPPY_PCTL).shift(1)
    raw = (rv <= rv_th.fillna(np.inf)) & (bb <= bb_th.fillna(np.inf))
    return raw.fillna(False)

def range_aware_choppy_strict(df):
    adx_low  = df["ADX14"] < RANGE_ADX_MAX
    drift_sm = df["Drift_10"].abs() < RANGE_DRIFT_MAXABS
    dwidth   = df["DonchianW"]
    dwidth_pct = dwidth.rolling(120, min_periods=40)\
                       .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x)>0 else np.nan)
    narrow   = (dwidth_pct <= RANGE_DWIDTH_PCTL).fillna(False)
    return (adx_low & drift_sm & narrow)

# ---------- Build labels ----------
def build_labels_rt(df, states, label_by_state):
    base = np.array([label_by_state(s) for s in states], dtype=object)

    # Adaptive guardrails
    bull_force, bear_force, drift_abs_dyn = trend_guardrail_adaptive(df)
    forced = np.where(bull_force, "Bull", np.where(bear_force, "Bear", None))
    after_guard = np.where(forced==None, base, forced)

    # Choppy detection
    lowvol   = causal_lowvol_flag(df)
    range_ch = range_aware_choppy_strict(df)
    range_ch = range_ch & (base=="Choppy") & (~(bull_force | bear_force))
    chop_hyst = apply_hysteresis(lowvol, k_enter=CHOP_ENTER_K, k_exit=CHOP_EXIT_K)
    choppy_any= (chop_hyst | range_ch).astype(bool)

    # Drift override (vol-adaptive)
    drift = df["Drift_10"].fillna(0.0)
    strong = (drift.abs() > drift_abs_dyn).astype(int)
    strong_frac = strong.rolling(DRIFT_OVR_WIN, min_periods=1).mean().fillna(0.0)

    cs = df["close"]; cs = cs.iloc[:,0] if isinstance(cs, pd.DataFrame) else cs
    ew50 = df["EWMA50"]; ew50 = ew50.iloc[:,0] if isinstance(ew50, pd.DataFrame) else ew50

    proposed=[]; reason=[]
    for i,(lab_g,is_chop) in enumerate(zip(after_guard, choppy_any.values)):
        if bull_force.iloc[i] or bear_force.iloc[i]:
            side="above" if bull_force.iloc[i] else "below"
            proposed.append(lab_g)
            because=f"Guardrail(adaptive): ADX≥dyn, MA200 {side}, RSI dyn OK, MACD slope OK, vol-confirmed"
            reason.append(because); continue

        if is_chop:
            adx_i = float(df["ADX14"].iloc[i])
            frac_req = 0.80 if adx_i<20.0 else DRIFT_OVR_FRAC
            vol_ok   = (df["VolPctRank120"].iloc[i] >= VOL_MIN_PCT)
            cmf_ok_bull = (df["CMF20"].iloc[i] > 0)
            cmf_ok_bear = (df["CMF20"].iloc[i] < 0)
            above50 = bool(cs.iloc[i] > ew50.iloc[i])
            if (strong_frac.iloc[i] >= frac_req) and (adx_i >= 18.0) and vol_ok:
                if drift.iloc[i] > 0 and above50 and cmf_ok_bull:
                    proposed.append("Bull"); reason.append("Choppy→Bull by drift (adaptive)"); continue
                if drift.iloc[i] < 0 and (not above50) and cmf_ok_bear:
                    proposed.append("Bear"); reason.append("Choppy→Bear by drift (adaptive)"); continue
            proposed.append("Choppy"); reason.append("Range/Low-vol (strict)"); continue

        if   lab_g == "Bull": reason.append("HMM→Bull (train-frozen)")
        elif lab_g == "Bear": reason.append("HMM→Bear (train-frozen)")
        else:                  reason.append("HMM→Choppy (train-frozen)")
        proposed.append(lab_g)

    # Causal persistence (no weekly gate)
    out=[]; why=[]
    last_label=None
    pending=None; pend_count=0

    for i, (p, r) in enumerate(zip(proposed, reason)):
        forced_now = (p in ("Bull","Bear")) and ( (bull_force.iloc[i] and p=="Bull") or (bear_force.iloc[i] and p=="Bear") )

        if last_label is None:
            out.append(p); why.append(r)
            last_label=p; pending=None; pend_count=0
            continue

        if p == last_label:
            out.append(last_label); why.append("Stay " + last_label + " | " + r)
            pending=None; pend_count=0
            continue

        if pending is None or pending != p:
            pending = p; pend_count = 1
        else:
            pend_count += 1

        if forced_now or (pend_count >= MIN_PERSIST_DAYS):
            last_label = p
            out.append(last_label); why.append(("FORCED " if forced_now else "Flip")+"→"+last_label+f" (persist {pend_count}d)")
            pending=None; pend_count=0
        else:
            out.append(last_label); why.append(f"Hold {last_label} (pending {pending} {pend_count}/{MIN_PERSIST_DAYS})")

    lbl = df[["close","LogReturn"]].copy()
    lbl["Label_rt"] = out
    lbl["Why_rt"]   = why
    lbl["state"]    = states
    return lbl

def contiguous_periods(df_lbl, label_col="Label_rt"):
    rows=[]
    if df_lbl.empty: return rows
    lab = df_lbl[label_col].values
    idx = df_lbl.index
    start = idx[0]; cur = lab[0]
    for i in range(1,len(df_lbl)):
        if lab[i] != cur:
            seg = df_lbl.iloc[:i].loc[start:idx[i-1]]
            seg_ret = float(np.exp(seg["LogReturn"].sum()) - 1.0)
            rows.append((cur, start, idx[i-1], len(seg), seg_ret))
            start = idx[i]; cur = lab[i]
    seg = df_lbl.loc[start:idx[-1]]
    seg_ret = float(np.exp(seg["LogReturn"].sum()) - 1.0)
    rows.append((cur, start, idx[-1], len(seg), seg_ret))
    return rows

def write_periods_txt(path, periods, title):
    with open(path, "w") as f:
        f.write(f"{title}\n")
        f.write("="*len(title) + "\n\n")
        for i,(lab, start, end, n, seg_ret) in enumerate(periods,1):
            f.write(f"[{i:03d}] {lab}\n")
            f.write(f"     Period : {start.date()} -> {end.date()}  ({n} days)\n")
            f.write(f"     Return : {seg_ret:+.2%}\n\n")

# -------------------
# Walk-forward decode
# -------------------
def walk_forward_decode(full_feats, walk_days=90, init_train_start=TRAIN_START,
                        init_train_end=TRAIN_END, test_start=TEST_START, test_end=TEST_END,
                        out_csv="btc_usd_regimes_walkforward.csv",
                        txt_train="regime_periods_train_2014_2018.txt",
                        txt_test="regime_periods_test_2019plus_walkforward.txt",
                        do_plot=True):
    # Split initial train / test slices (index-aligned)
    feats = full_feats.copy()
    tr0 = feats.loc[(feats.index >= pd.Timestamp(init_train_start, tz="UTC")) &
                    (feats.index <= pd.Timestamp(init_train_end,   tz="UTC"))].copy()
    te  = feats.loc[(feats.index >= pd.Timestamp(test_start,    tz="UTC")) &
                    (feats.index <= (pd.Timestamp(test_end, tz="UTC") if test_end else feats.index.max()))].copy()

    if len(tr0)==0: raise RuntimeError("Empty initial train slice.")
    if len(te)==0:  raise RuntimeError("Empty test slice.")

    # Fit once on initial train for train reporting
    scaler_tr = RobustScaler()
    X_tr = scaler_tr.fit_transform(tr0[FEATS].values)
    hmm_tr = fit_hmm(X_tr, n_states=N_STATES, seed=SEED)
    post_tr = hmm_tr.predict_proba(X_tr)
    label_by_state_tr = state_label_mapper_from_train_forward(post_tr, tr0["close"], h=H_MAP, state_count=N_STATES)
    ch_tr, td_tr = score_blocks(tr0)
    st_tr = hmm_tr.predict(X_tr)  # offline prettifier optional
    st_tr = st_tr  # keep as is
    lbl_tr = build_labels_rt(tr0, st_tr, label_by_state_tr)

    # Walk-forward over TEST
    all_blocks = []
    dates_te = te.index
    if walk_days <= 0: walk_days = 90
    block_starts = [dates_te[0]]
    # create block boundaries
    last = dates_te[0]
    for dt in dates_te[1:]:
        if (dt - last).days >= walk_days:
            block_starts.append(dt)
            last = dt
    # ensure last block covers to end
    if block_starts[-1] != dates_te[0] and block_starts[-1] > dates_te[-1]:
        block_starts[-1] = dates_te[-1]

    # Iterate blocks
    for i, bstart in enumerate(block_starts):
        bend = dates_te[-1] if i == len(block_starts)-1 else (block_starts[i+1] - pd.Timedelta(days=1))
        block_mask = (te.index >= bstart) & (te.index <= bend)
        blk = te.loc[block_mask].copy()
        if blk.empty: continue

        # Expanding train: all data up to block start - 1 day
        train_mask = feats.index <= (bstart - pd.Timedelta(days=1))
        tr_i = feats.loc[train_mask].copy()
        # keep robust minimum size
        if len(tr_i) < max(250, len(FEATS)*5):
            raise RuntimeError(f"Train slice too short at block {bstart.date()} (n={len(tr_i)}).")

        scaler = RobustScaler()
        X_train = scaler.fit_transform(tr_i[FEATS].values)
        hmm = fit_hmm(X_train, n_states=N_STATES, seed=SEED)
        post = hmm.predict_proba(X_train)
        label_by_state = state_label_mapper_from_train_forward(post, tr_i["close"], h=H_MAP, state_count=N_STATES)

        # Transform block with current scaler
        X_blk = scaler.transform(blk[FEATS].values)
        ch_b, td_b = score_blocks(blk)
        st_blk = decode_causal_dynamic(hmm, X_blk, ch_b, td_b)
        lbl_blk = build_labels_rt(blk, st_blk, label_by_state)
        all_blocks.append(lbl_blk)

        print(f"[WF] Block {bstart.date()} → {bend.date()}  (train rows={len(tr_i)}, block rows={len(blk)})")

    lbl_te = pd.concat(all_blocks).sort_index()

    # Save outputs
    out = lbl_te.copy()
    out.insert(0, "ts", out.index)
    out.to_csv(out_csv, index=False)
    print(f"[OK] Saved {out_csv} (rows={len(out)})")

    # Write period summaries
    periods_tr = contiguous_periods(lbl_tr, "Label_rt")
    periods_te = contiguous_periods(lbl_te, "Label_rt")
    write_periods_txt(txt_train, periods_tr, "BTC-USD — Regime Periods (TRAIN 2014–2018)")
    write_periods_txt(txt_test,  periods_te,  "BTC-USD — Regime Periods (TEST 2019+ Walk-Forward)")
    print(f"[OK] Wrote: {txt_train}")
    print(f"[OK] Wrote: {txt_test}")

    # Plot TEST with regimes
    if do_plot:
        def segs_from_labels(idx, labs):
            segs=[]; cur=labs[0]; start=idx[0]
            for i in range(1,len(labs)):
                if labs[i]!=cur:
                    segs.append((start, idx[i-1], cur))
                    start=idx[i]; cur=labs[i]
            segs.append((start, idx[-1], cur)); return segs

        colors = {"Bull":"#2ecc71","Bear":"#e74c3c","Choppy":"#3498db"}
        ts = lbl_te.index
        labs = lbl_te["Label_rt"].values
        closes = feats.loc[ts, "close"]
        fig, ax = plt.subplots(figsize=(13,6))
        ax.plot(ts, closes, lw=1.6, alpha=0.95)
        for a,b,lab in segs_from_labels(list(ts), list(labs)):
            ax.axvspan(a, b, color=colors.get(lab,"#ccc"), alpha=0.18, linewidth=0)
        ax.set_yscale("log"); ax.grid(True, alpha=0.25)
        ax.set_title(f"BTC-USD — Walk-Forward Regimes (Train≤blockStart; H={H_MAP}d forward map)")
        ax.set_xlabel("Date"); ax.set_ylabel("Price (log scale)")
        ax.legend(handles=[Patch(facecolor=colors["Bull"], alpha=0.35, label="Bull"),
                           Patch(facecolor=colors["Bear"], alpha=0.35, label="Bear"),
                           Patch(facecolor=colors["Choppy"], alpha=0.35, label="Choppy")],
                  loc="upper left")
        plt.tight_layout(); plt.show()

    return lbl_tr, lbl_te

# -------------------
# Main
# -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_start", default=TRAIN_START)
    ap.add_argument("--train_end",   default=TRAIN_END)
    ap.add_argument("--test_start",  default=TEST_START)
    ap.add_argument("--test_end",    default=TEST_END)
    ap.add_argument("--walk_days",   type=int, default=90)
    ap.add_argument("--plot",        action="store_true")
    ap.add_argument("--out_csv",     default="btc_usd_regimes_2019plus_walkforward.csv")
    ap.add_argument("--txt_train",   default="regime_periods_train_2014_2018.txt")
    ap.add_argument("--txt_test",    default="regime_periods_test_2019plus_walkforward.txt")
    args = ap.parse_args()

    # Pull data full span needed
    full = fetch_btc_yf("2014-01-01", args.test_end or "2025-12-31")
    feats = build_features(full)

    # Run walk-forward
    _lbl_tr, _lbl_te = walk_forward_decode(
        full_feats=feats,
        walk_days=int(args.walk_days),
        init_train_start=args.train_start,
        init_train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        out_csv=args.out_csv,
        txt_train=args.txt_train,
        txt_test=args.txt_test,
        do_plot=bool(args.plot)
    )

if __name__ == "__main__":
    main()
