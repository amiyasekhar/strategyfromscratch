# BTC/USDT Regime Classification + Plot (single cell)
# If needed, uncomment installs:
# !pip -q install hmmlearn requests pandas numpy scikit-learn matplotlib

import time, numpy as np, pandas as pd, requests, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler

# -----------------------
# Config (tunable knobs)
# -----------------------
SYMBOL            = "BTCUSDT"
INTERVAL          = "1d"
TRAIN_START       = "2019-01-01"
TRAIN_END         = "2023-12-31"
TEST_START        = "2024-01-01"
TEST_END          = None  # e.g., "2025-09-05" or None for latest

# Labeling + smoothing knobs
NEAR_ZERO_BAND    = 0.0010   # narrower band => fewer "Choppy"
DEBOUNCE_MIN_RUN  = 5        # min consecutive days per state
BASE_KAPPA        = 6.0      # baseline stickiness
ALPHA_CHOPPY      = 1.0      # stickier in chop
ALPHA_TREND_LOOSEN= 1.0      # loosen faster when trends strengthen

SEED              = 42
START_BAL         = 1.0

CSV_OUT           = "btcusdt_daily_regimes.csv"
TXT_TRAIN_OUT     = "regime_periods_train_2019_2023.txt"
TXT_TEST_OUT      = "regime_periods_test_2024plus.txt"

np.set_printoptions(suppress=True, floatmode="fixed", linewidth=140)

# -----------------------
# Fetch + features
# -----------------------
def _to_ms(ts_str):
    if ts_str is None: return None
    return int(pd.Timestamp(ts_str, tz="UTC").timestamp() * 1000)

def binance_klines(symbol, interval, start=None, end=None, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    if start: params["startTime"] = _to_ms(start)
    if end:   params["endTime"]   = _to_ms(end)
    out = []
    print("=== Fetching daily OHLCV & building features (paginated) ===")
    while True:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data: break
        out.extend(data)
        last_open_ms = data[-1][0]
        params["startTime"] = last_open_ms + 1
        if end and last_open_ms >= _to_ms(end): break
        time.sleep(0.1)
        if len(data) < params["limit"]: break
    if not out:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"]).set_index(pd.Index([], name="ts"))
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(out, columns=cols)
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df = df[["ts","open","high","low","close","volume"]].set_index("ts").sort_index()
    return df

def ema(x, span):
    return x.ewm(span=span, adjust=False, min_periods=span).mean()

def true_range(h,l,c):
    pc = c.shift(1)
    return pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def adx(h,l,c,length=14):
    up = h.diff(); down = -l.diff()
    plus_dm  = np.where((up>down)&(up>0), up, 0.0)
    minus_dm = np.where((down>up)&(down>0), down, 0.0)
    tr   = true_range(h,l,c)
    atr  = tr.rolling(length).mean()
    p_di = pd.Series(100*pd.Series(plus_dm,  index=c.index).rolling(length).mean()/atr, index=c.index)
    m_di = pd.Series(100*pd.Series(minus_dm, index=c.index).rolling(length).mean()/atr, index=c.index)
    dx   = (100*(p_di-m_di).abs()/(p_di+m_di)).replace([np.inf,-np.inf], np.nan)
    return dx.rolling(length).mean()

def kama(price, er_len=10, fast=2, slow=30):
    change = price.diff(er_len).abs()
    vol    = price.diff().abs().rolling(er_len).sum()
    er     = (change/vol).replace([np.inf,-np.inf], np.nan).fillna(0)
    sc     = (er*(2/(fast+1)-2/(slow+1)) + 2/(slow+1))**2
    out    = pd.Series(index=price.index, dtype=float)
    out.iloc[0] = price.iloc[0]
    for i in range(1,len(price)):
        sci = sc.iloc[i] if np.isfinite(sc.iloc[i]) else 0.0
        out.iloc[i] = out.iloc[i-1] + sci*(price.iloc[i]-out.iloc[i-1])
    return out

def donchian(h,l,n=20):
    upper = h.rolling(n).max(); lower = l.rolling(n).min()
    width = (upper-lower).div(lower.replace(0, np.nan)).fillna(0.0)
    rng   = (upper-lower).replace(0, np.nan)
    pos   = (((h+l)/2)-lower).div(rng).clip(0,1).fillna(0.5)
    return upper, lower, width, pos

def build_features(df):
    px, hi, lo = df["close"], df["high"], df["low"]
    df["LogReturn"] = np.log(px/px.shift(1))
    df["ATRp"]      = (true_range(hi,lo,px).rolling(14).mean()/px).fillna(0.0)
    df["RealVol20"] = df["LogReturn"].rolling(20).std().fillna(0.0)
    df["EWMA5"]  = ema(px,5);   df["EWMA20"] = ema(px,20)
    df["EWMA50"] = ema(px,50);  df["EWMA200"] = ema(px,200)
    df["Drift_10"]  = (df["EWMA5"]  - df["EWMA20"]).div(df["EWMA20"]).replace([np.inf,-np.inf], 0).fillna(0.0)
    df["Drift_50"]  = (df["EWMA20"] - df["EWMA50"]).div(df["EWMA50"]).replace([np.inf,-np.inf], 0).fillna(0.0)
    df["Drift_200"] = (df["EWMA50"] - df["EWMA200"]).div(df["EWMA200"]).replace([np.inf,-np.inf], 0).fillna(0.0)
    df["ADX14"]     = adx(hi,lo,px,14).fillna(0.0)
    df["KAMA"]      = kama(px)
    df["KAMA_dev"]  = (px-df["KAMA"]).div(df["KAMA"].replace(0, np.nan)).fillna(0.0)
    df["MA200z"]    = (px - df["EWMA200"]).div(px.rolling(200).std().replace(0, np.nan)).fillna(0.0)
    _,_,dwidth,dpos = donchian(hi,lo,20)
    df["DonchianW"]   = dwidth.fillna(0.0)
    df["DonchianPos"] = dpos.fillna(0.5)
    return df.dropna().copy()

def split_train_test(df):
    tr = df.loc[(df.index>=pd.Timestamp(TRAIN_START, tz="UTC"))&(df.index<=pd.Timestamp(TRAIN_END, tz="UTC"))].copy()
    te = df.loc[(df.index>=pd.Timestamp(TEST_START,  tz="UTC"))].copy() if TEST_START else df.iloc[0:0].copy()
    if TEST_END: te = te.loc[te.index<=pd.Timestamp(TEST_END, tz="UTC")]
    print(f"Train: {tr.index.min()} -> {tr.index.max()} (n={len(tr)})")
    print(f"Test : {te.index.min()} -> {te.index.max()} (n={len(te)})")
    return tr, te

# -----------------------
# HMM + decoding helpers
# -----------------------
def fit_hmm(X, n_states=3, seed=SEED):
    hmm = GaussianHMM(n_components=n_states, covariance_type="diag",
                      n_iter=200, random_state=seed, tol=1e-4)
    hmm.fit(X)
    return hmm

def sticky_transmat(A, kappa):
    A = np.asarray(A, dtype=float).copy()
    S = A.shape[0]
    if S == 0: return A
    if (A <= 0).any() or not np.isfinite(A).all():
        A = np.ones_like(A) / S
    A = A + np.eye(S)*kappa
    A = (A.T / A.sum(axis=1)).T
    return A

def _diag_var(cov_s):
    cov_s = np.asarray(cov_s)
    if cov_s.ndim == 1:
        var = cov_s
    elif cov_s.ndim == 2:
        var = np.diag(cov_s)
    else:
        var = np.squeeze(cov_s)
        if var.ndim == 2: var = np.diag(var)
    return np.clip(var.astype(float), 1e-8, np.inf)

def decode_causal_dynamic(hmm, X, choppy_score, trend_score):
    S, T, D = hmm.n_components, X.shape[0], X.shape[1]
    # emission log-probs (diag Gaussian)
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
        prev    = log_alpha[t-1][:,None] + log_A_t   # (S,S)
        backptr[t] = np.argmax(prev, axis=0)
        log_alpha[t] = prev.max(axis=0) + logprob_t[t]

    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(log_alpha[-1])
    for t in range(T-2,-1,-1):
        states[t] = backptr[t+1, states[t+1]]
    return states

def debounce_states(states, min_run):
    if len(states)==0: return states
    s = states.copy(); i=0
    while i<len(s):
        j=i
        while j+1<len(s) and s[j+1]==s[i]: j+=1
        run = j-i+1
        if run<min_run:
            if i>0: s[i:j+1]=s[i-1]
            elif j+1<len(s): s[i:j+1]=s[j+1]
        i=j+1
    return s

def state_to_label_map(mu_by_state, near_zero_band=NEAR_ZERO_BAND):
    ranked = sorted(mu_by_state.items(), key=lambda kv: kv[1])  # low->high
    lo_s, lo_mu = ranked[0]
    mid_s, mid_mu = ranked[1]
    hi_s, hi_mu = ranked[2]
    def label_of_state(s):
        mu = mu_by_state[s]
        if abs(mu-mid_mu) <= near_zero_band: return "Choppy"
        if s==lo_s: return "Bear"
        if s==hi_s: return "Bull"
        return "Choppy"
    return label_of_state, (lo_s, mid_s, hi_s), (lo_mu, mid_mu, hi_mu)

def contiguous_periods(df_lbl, label_col="Label", why_col="Why"):
    rows=[]
    if df_lbl.empty: return rows
    cur  = df_lbl[label_col].iloc[0]
    why  = df_lbl[why_col].iloc[0]
    start= df_lbl.index[0]
    for t in range(1,len(df_lbl)):
        if df_lbl[label_col].iloc[t]!=cur:
            end = df_lbl.index[t-1]
            seg = df_lbl.loc[start:end]
            seg_ret = float(np.exp(seg["LogReturn"].sum())-1.0)
            rows.append((cur, start, end, len(seg), seg_ret, float(seg["LogReturn"].mean()), float(seg["LogReturn"].std()), why))
            cur  = df_lbl[label_col].iloc[t]
            why  = df_lbl[why_col].iloc[t]
            start= df_lbl.index[t]
    end = df_lbl.index[-1]; seg = df_lbl.loc[start:end]
    rows.append((cur, start, end, len(seg), float(np.exp(seg["LogReturn"].sum())-1.0), float(seg["LogReturn"].mean()), float(seg["LogReturn"].std()), why))
    return rows

# -----------------------
# Backtests + metrics
# -----------------------
def backtest_long_only(df_lbl, label_col="Label"):
    bal=START_BAL; pos=0.0; eq=[]; trades=0
    for _,row in df_lbl.iterrows():
        px=row["close"]; want=(row[label_col]=="Bull")
        if want and pos==0.0:
            pos=bal/px; bal=0.0; trades+=1
        elif (not want) and pos>0.0:
            bal=pos*px; pos=0.0; trades+=1
        eq.append(bal+pos*px)
    if pos>0:
        bal=pos*df_lbl["close"].iloc[-1]; pos=0.0
    eq=np.array(eq); rets=np.diff(np.log(eq+1e-12), prepend=np.log(eq[0]+1e-12))
    sharpe=float((np.mean(rets)/(np.std(rets)+1e-12))*np.sqrt(365))
    return {"days":len(df_lbl),"trades":trades,"total_return_strat":float(bal/START_BAL),
            "total_return_bh":float(df_lbl['close'].iloc[-1]/df_lbl['close'].iloc[0]),"sharpe":sharpe}

def per_label_stats(df_lbl):
    g=df_lbl.groupby("Label")["LogReturn"]
    out=pd.DataFrame({"mean":g.mean(),"std":g.std(),"count":g.size()})
    out["ann_IR"]= (out["mean"]/out["std"]).replace([np.inf,-np.inf], np.nan)*np.sqrt(365)
    return out.fillna(0.0)

def da_bull_bear(df_lbl):
    lbl = df_lbl.copy()
    lbl["ret_sign"]=np.where(lbl["LogReturn"]>0, "+", "-")
    return float( np.mean( ((lbl["Label"]=="Bull") & (lbl["ret_sign"]=="+")) |
                           ((lbl["Label"]=="Bear") & (lbl["ret_sign"]=="-")) ) )

def three_way_accuracy(df_lbl, band=NEAR_ZERO_BAND):
    small = df_lbl["LogReturn"].abs() <= band
    up    = df_lbl["LogReturn"] > 0
    dn    = df_lbl["LogReturn"] < 0
    pred  = df_lbl["Label"]
    correct = (((pred=="Bull")&up) | ((pred=="Bear")&dn) | ((pred=="Choppy")&small))
    return float(correct.mean())

# -----------------------
# Run HMM + label
# -----------------------
df_all = binance_klines(SYMBOL, INTERVAL, start=TRAIN_START, end=TEST_END)
df_all = build_features(df_all)
dtr, dte = split_train_test(df_all)

FEATS = ["RealVol20","ATRp","Drift_10","Drift_50","Drift_200","ADX14","KAMA_dev","MA200z","DonchianW","DonchianPos"]
scaler = RobustScaler()
X_tr = scaler.fit_transform(dtr[FEATS].values)
X_te = scaler.transform(dte[FEATS].values)

print("=== Fitting HMM (train) ===")
hmm = fit_hmm(X_tr, n_states=3, seed=SEED)

# State means on TRAIN via soft responsibilities
post = hmm.predict_proba(X_tr)
mu_state = {}
for s in range(hmm.n_components):
    w = post[:,s]
    mu_state[s] = float(np.sum(w*dtr["LogReturn"].values)/(w.sum()+1e-12))

label_of_state, mu_states, mu_vals = state_to_label_map(mu_state, NEAR_ZERO_BAND)

# Dynamic scores (TEST)
def _minmax(x):
    x = np.asarray(x, dtype=float)
    a=np.nanmin(x); b=np.nanmax(x)
    if not np.isfinite(a) or not np.isfinite(b) or b-a<1e-12:
        return np.zeros_like(x)
    return (x-a)/(b-a)

choppy_score = np.nanmean(np.vstack([
    1-_minmax(dte["LogReturn"].abs().values),
    1-_minmax(dte["ADX14"].values),
    1-_minmax(dte["DonchianW"].values)
]), axis=0)
choppy_score = np.clip(choppy_score,0,1)

trend_score = np.nanmean(np.vstack([
    _minmax(np.abs(dte["Drift_10"].values)),
    _minmax(np.abs(dte["Drift_50"].values)),
    _minmax(dte["KAMA_dev"].values),
    _minmax(dte["ADX14"].values),
    _minmax(dte["DonchianW"].values),
]), axis=0)
trend_score = np.clip(trend_score,0,1)

print("=== Decoding TEST (causal, adaptive) ===")
states_te = decode_causal_dynamic(hmm, X_te, choppy_score, trend_score)
states_te = debounce_states(states_te, DEBOUNCE_MIN_RUN)
states_tr = debounce_states(hmm.predict(X_tr), DEBOUNCE_MIN_RUN)

# Labels & Why
lo_s, mid_s, hi_s = mu_states
lo_mu, mid_mu, hi_mu = mu_vals

def day_why(s):
    if abs(mu_state[s]-mid_mu) <= NEAR_ZERO_BAND: return "Near-zero => Choppy"
    if s==hi_s: return f"State {s} (mu_train={mu_state[s]:+.6f}) => Bull"
    if s==lo_s: return f"State {s} (mu_train={mu_state[s]:+.6f}) => Bear"
    return "Near-zero => Choppy"

lbl_tr = pd.DataFrame(index=dtr.index)
lbl_tr["state"]=states_tr
lbl_tr["Label"]=[label_of_state(s) for s in states_tr]
lbl_tr["Why"]  =[day_why(s) for s in states_tr]
lbl_tr["LogReturn"]=dtr["LogReturn"]
lbl_tr["close"]=dtr["close"]

lbl_te = pd.DataFrame(index=dte.index)
lbl_te["state"]=states_te
lbl_te["Label"]=[label_of_state(s) for s in states_te]
lbl_te["Why"]  =[day_why(s) for s in states_te]
lbl_te["LogReturn"]=dte["LogReturn"]
lbl_te["close"]=dte["close"]

# Periods
def print_periods_header(title, periods, mu_tuple):
    lo_s, mid_s, hi_s = mu_tuple[0]
    lo_mu, mid_mu, hi_mu = mu_tuple[1]
    bull_days = sum(p[3] for p in periods if p[0]=="Bull")
    bear_days = sum(p[3] for p in periods if p[0]=="Bear")
    ch_days   = sum(p[3] for p in periods if p[0]=="Choppy")
    print(title)
    print("="*len(title))
    if bull_days: print(f"  - Bull: {bull_days} days")
    if bear_days: print(f"  - Bear: {bear_days} days")
    if ch_days:   print(f"  - Choppy: {ch_days} days")
    print("")
    print("TRAIN state means (mu daily, raw returns, 2019-2023):")
    print(f"  State {lo_s}: mu={lo_mu:+.6f}")
    print(f"  State {mid_s}: mu={mid_mu:+.6f}")
    print(f"  State {hi_s}: mu={hi_mu:+.6f}")
    print(f"  Near-zero band: +/-{NEAR_ZERO_BAND:.6f}\n")

train_periods = contiguous_periods(lbl_tr, "Label", "Why")
test_periods  = contiguous_periods(lbl_te, "Label", "Why")

print("")
print_periods_header("BTC/USDT — Regime Classification (SEEN: 2019-2023)", train_periods, (mu_states, mu_vals))
print(f"Saved detailed periods: {TXT_TRAIN_OUT}\n")
print_periods_header("BTC/USDT — Regime Classification (UNSEEN: 2024+)", test_periods, (mu_states, mu_vals))
print(f"Saved detailed periods: {TXT_TEST_OUT}\n")

# Backtests + stats
print("=== Backtests (Choppy=flat; long-only in Bull) ===")
bt_tr=backtest_long_only(lbl_tr); bt_te=backtest_long_only(lbl_te)
print("SEEN  (2019-2023):", {k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in bt_tr.items()})
print("UNSEEN (2024+):   ", {k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in bt_te.items()}, "\n")

print("=== SEEN (2019-2023) ===\n")
stats_tr=per_label_stats(lbl_tr); print(stats_tr, "\n")
print(f"2-way directional accuracy (Bull↑, Bear↓): {da_bull_bear(lbl_tr):.3f}")
print(f"3-way accuracy (Bull↑, Bear↓, Choppy≈0):   {three_way_accuracy(lbl_tr):.3f}\n")
print("Label vs realized sign of returns:")
print(pd.crosstab(lbl_tr["Label"], np.where(lbl_tr["LogReturn"]>0,"+","-")), "\n")

print("=== UNSEEN (2024+) ===\n")
stats_te=per_label_stats(lbl_te); print(stats_te, "\n")
print(f"2-way directional accuracy (Bull↑, Bear↓): {da_bull_bear(lbl_te):.3f}")
print(f"3-way accuracy (Bull↑, Bear↓, Choppy≈0):   {three_way_accuracy(lbl_te):.3f}\n")
print("Label vs realized sign of returns:")
print(pd.crosstab(lbl_te["Label"], np.where(lbl_te["LogReturn"]>0,"+","-")), "\n")

print("=== Last 20 days (UNSEEN) — Close, LogReturn, Label ===")
print(lbl_te[["close","LogReturn","Label"]].tail(20), "\n")

# Save CSV and period files
df_out = pd.concat([lbl_tr.assign(split="train"), lbl_te.assign(split="test")]).reset_index().rename(columns={"index":"ts"})
df_out.to_csv(CSV_OUT, index=False); print(f"Saved: {CSV_OUT}")

def write_periods_txt(path, periods):
    with open(path,"w") as f:
        for i,(lab, start, end, n, seg_ret, mu_seg, sig_seg, why) in enumerate(periods,1):
            sig_seg = 0.0 if (sig_seg is None or np.isnan(sig_seg)) else float(sig_seg)
            f.write(f"[{i:03d}] {lab}\n")
            f.write(f"     Period : {start.date()} -> {end.date()}  ({n} days)\n")
            f.write(f"     Return : {seg_ret:+.2%}  (mu_segment={mu_seg:+.6f}, sigma_segment={sig_seg:.6f})\n")
            f.write(f"     Why    : {why}\n\n")

write_periods_txt(TXT_TRAIN_OUT, train_periods); print("Wrote:", TXT_TRAIN_OUT)
write_periods_txt(TXT_TEST_OUT,  test_periods);  print("Wrote:", TXT_TEST_OUT)

# -----------------------
# Plot BTC price with regime highlights (Bull=green, Bear=red, Choppy=blue)
# -----------------------
def _contiguous_segments_plot(df, label_col="Label", ts_col="ts"):
    if df.empty:
        return []
    segs=[]
    cur_label=df[label_col].iloc[0]
    start=df[ts_col].iloc[0]
    for i in range(1,len(df)):
        if df[label_col].iloc[i] != cur_label:
            end = df[ts_col].iloc[i-1]
            segs.append((start, end, cur_label))
            cur_label = df[label_col].iloc[i]
            start = df[ts_col].iloc[i]
    segs.append((start, df[ts_col].iloc[-1], cur_label))
    return segs

# You can set USE_SPLIT = None / "train" / "test"
USE_SPLIT = None
plot_df = df_out.copy()
if USE_SPLIT:
    plot_df = plot_df[plot_df["split"].str.lower()==USE_SPLIT.lower()].copy()
plot_df = plot_df.sort_values("ts").reset_index(drop=True)

colors = {"Bull": "#2ecc71", "Bear": "#e74c3c", "Choppy": "#3498db"}
segs = _contiguous_segments_plot(plot_df, "Label", "ts")

fig, ax = plt.subplots(figsize=(13,6))
ax.plot(plot_df["ts"], plot_df["close"], lw=1.6, label="BTC/USDT Close", alpha=0.95)

for (start, end, lab) in segs:
    ax.axvspan(start, end, color=colors.get(lab, "#cccccc"), alpha=0.18, linewidth=0)

ax.set_title("BTC/USDT — Regime Highlights (Bull=green, Bear=red, Choppy=blue)")
ax.set_xlabel("Date"); ax.set_ylabel("Price (log scale)")
ax.set_yscale("log"); ax.grid(True, alpha=0.25)

patches = [Patch(facecolor=colors["Bull"], alpha=0.35, label="Bull"),
           Patch(facecolor=colors["Bear"], alpha=0.35, label="Bear"),
           Patch(facecolor=colors["Choppy"], alpha=0.35, label="Choppy")]
ax.legend(handles=patches, loc="upper left")

plt.tight_layout()
plt.show()

print("\nNotes:")
print(f"- Near-zero band +/-{NEAR_ZERO_BAND} => mid-mean state becomes 'Choppy' when close to the middle state mean.")
print(f"- Debounce = {DEBOUNCE_MIN_RUN} days (prevents flips on short bursts).")
print("- Dynamic stickiness: higher when choppy (low ADX / narrow Donchian / small moves), lower when trends strengthen.")
print("- Emission step uses robust diag-Gaussian logpdf; cov diagonals are clipped to avoid zero-variance issues.")