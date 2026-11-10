"""Kronos simple app v3 - একটা ক্যান্ডেল, ২০ বার অনুমান, দুইবার যাচাই।
এই ফাইলটা Kronos Fork-এর মূল ফোল্ডারে (model ফোল্ডারের পাশে) kronos_app.py নামে রাখুন।"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yfinance as yf

st.set_page_config(page_title="Kronos পরের ক্যান্ডেল", layout="wide")

IST = "Asia/Kolkata"
MODELS = {
    "Kronos-mini (হালকা, দ্রুত)": ("NeoQuasar/Kronos-Tokenizer-2k", "NeoQuasar/Kronos-mini", 2048),
    "Kronos-small (একটু ভালো, ভারী)": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-small", 512),
}
SEED_A, SEED_B = 42, 1042


@st.cache_resource(show_spinner="Kronos মডেল লোড হচ্ছে (প্রথমবার সময় লাগবে)...")
def load_predictor(key):
    from model import Kronos, KronosPredictor, KronosTokenizer

    tok_id, mod_id, ctx = MODELS[key]
    tokenizer = KronosTokenizer.from_pretrained(tok_id)
    model = Kronos.from_pretrained(mod_id)
    return KronosPredictor(model, tokenizer, device="cpu", max_context=ctx)


@st.cache_data(ttl=1800, show_spinner=False)
def get_daily(ticker):
    d = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
    idx = pd.DatetimeIndex(d.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    d.index = idx.normalize()
    return d


def to_weekly(d):
    return d.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()


def run_many(pred, x, tf, runs, temp, seed, label):
    """পরের ১টা ক্যান্ডেলের জন্য `runs` বার অনুমান। ফল: runs x 4 (O,H,L,C)"""
    x = x.tail(400)
    xdf = pd.DataFrame({
        "open": x.Open.values, "high": x.High.values, "low": x.Low.values,
        "close": x.Close.values, "volume": x.Volume.values,
        "amount": (x.Close * x.Volume).values,
    })
    xts = pd.Series(x.index)
    start = x.index[-1] + pd.Timedelta(days=1)
    fut = pd.bdate_range(start, periods=1) if tf == "Daily" else pd.date_range(start, periods=1, freq="W-FRI")
    yts = pd.Series(fut)
    outs = []
    bar = st.progress(0.0, text=label)
    for i in range(runs):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        p = pred.predict(df=xdf, x_timestamp=xts, y_timestamp=yts, pred_len=1,
                         T=temp, top_p=0.9, sample_count=1)
        outs.append(p[["open", "high", "low", "close"]].values[0].astype(float))
        bar.progress((i + 1) / runs, text=f"{label} ({i + 1}/{runs})")
    bar.empty()
    return fut, np.array(outs)


def draw(x, fut, arr, n_hist=40):
    mean = arr.mean(axis=0)
    lo, hi = np.percentile(arr[:, 3], 10), np.percentile(arr[:, 3], 90)
    h = x.tail(n_hist)
    fig, ax = plt.subplots(figsize=(11, 5))

    def candle(xp, o, hh, l, c, alpha):
        col = "#16a34a" if c >= o else "#dc2626"
        ax.plot([xp, xp], [l, hh], color=col, lw=1, alpha=alpha)
        ax.add_patch(plt.Rectangle((xp - 0.3, min(o, c)), 0.6, max(abs(c - o), 1e-9), color=col, alpha=alpha))

    for i, (o, hh, l, c) in enumerate(zip(h.Open, h.High, h.Low, h.Close)):
        candle(i, o, hh, l, c, 1.0)
    xf = len(h)
    candle(xf, mean[0], mean[1], mean[2], mean[3], 0.5)
    ax.fill_between([xf - 0.4, xf + 0.4], lo, hi, color="#2563eb", alpha=0.2, label="২০ অনুমানের Close (১০%–৯০%)")
    ax.axvline(len(h) - 0.5, ls="--", color="gray")
    dates = list(h.index) + list(fut)
    ticks = np.linspace(0, len(dates) - 1, 6).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[i].strftime("%d %b %y") for i in ticks], fontsize=8)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(-1, len(h) + 1)
    return fig


# ------------------------------------------------------------------ UI
st.title("Kronos - পরের ক্যান্ডেল কোন দিকে?")
st.caption("এক ক্যান্ডেলের জন্য মডেল অনেকবার অনুমান করে। কতবার ওপরে বলল সেটা গুনে দেখায়। এটা আন্দাজ, নিশ্চিত নয়।")

with st.sidebar:
    exch = st.radio("এক্সচেঞ্জ", ["NSE", "BSE"], horizontal=True)
    tf = st.radio("টাইমফ্রেম", ["Weekly", "Daily"], horizontal=True)
    model_key = st.selectbox("মডেল", list(MODELS))
    runs = st.slider("কতবার অনুমান (প্রতি পরীক্ষায়)", 10, 40, 20)
    temp = st.slider("Randomness (কম = বেশি স্থির)", 0.3, 1.0, 0.6, 0.1)
    complete_only = st.checkbox("চলমান অসম্পূর্ণ ক্যান্ডেল বাদ দিন", value=True)

name = st.text_input("স্টকের নাম (NSE: RELIANCE | BSE: স্ক্রিপ কোড যেমন 500325)", "RELIANCE").strip().upper()
go = st.button("Forecast", type="primary")

if go and name:
    ticker = name if "." in name else name + (".NS" if exch == "NSE" else ".BO")
    try:
        daily = get_daily(ticker)
    except Exception as e:
        st.error(f"{ticker}-এর ডেটা আনা যায়নি: {e}")
        st.stop()
    if len(daily) < 150:
        st.error(f"{ticker}-এর ডেটা যথেষ্ট নেই। নাম বা এক্সচেঞ্জ ঠিক আছে কি না দেখুন।")
        st.stop()

    x = daily if tf == "Daily" else to_weekly(daily)
    now = pd.Timestamp.now(tz=IST)
    today = now.tz_localize(None).normalize()
    mkt_open = now.weekday() < 5 and (now.hour * 60 + now.minute) < 15 * 60 + 45
    last = x.index[-1]
    if complete_only and (last > today or (last == today and mkt_open)):
        x = x.iloc[:-1]
    if len(x) < 100:
        st.error("ক্যান্ডেল যথেষ্ট নেই।")
        st.stop()

    pred = load_predictor(model_key)
    fut, arrA = run_many(pred, x, tf, runs, temp, SEED_A, "পরীক্ষা ১")
    _, arrB = run_many(pred, x, tf, runs, temp, SEED_B, "পরীক্ষা ২")

    last_close = float(x.Close.iat[-1])
    upA = int((arrA[:, 3] > last_close).sum())
    upB = int((arrB[:, 3] > last_close).sum())
    pA, pB = upA / runs, upB / runs
    allarr = np.vstack([arrA, arrB])
    p = (upA + upB) / (2 * runs)
    agree = abs(pA - pB) <= 0.20

    st.subheader(f"{name} ({'সাপ্তাহিক' if tf == 'Weekly' else 'দৈনিক'}) - শেষ ক্যান্ডেল {x.index[-1].date()}, Close {last_close:.2f}")

    c1, c2 = st.columns(2)
    c1.metric("পরীক্ষা ১: কতবার ওপরে", f"{upA} / {runs}")
    c2.metric("পরীক্ষা ২: কতবার ওপরে", f"{upB} / {runs}")

    if not agree:
        st.warning("দুই পরীক্ষার ফল মেলেনি। এই সংকেত ধরবেন না (মডেল নিজেই নিশ্চিত নয়)।")
    elif p >= 0.75:
        st.success(f"ওপরের ঝোঁক স্পষ্ট: মোট {2 * runs} বারের {upA + upB} বার ওপরে ({p * 100:.0f}%)।")
    elif p >= 0.60:
        st.success(f"সামান্য ওপরের ঝোঁক: {p * 100:.0f}% বার ওপরে।")
    elif p > 0.40:
        st.info(f"কোনো স্পষ্ট ঝোঁক নেই ({p * 100:.0f}% বার ওপরে)। এই স্টকে ট্রেড ভাববেন না।")
    elif p > 0.25:
        st.error(f"সামান্য নিচের ঝোঁক: মাত্র {p * 100:.0f}% বার ওপরে।")
    else:
        st.error(f"নিচের ঝোঁক স্পষ্ট: মাত্র {p * 100:.0f}% বার ওপরে।")

    med = float(np.median(allarr[:, 3]))
    st.write(
        f"অনুমানের মাঝামাঝি Close: **{med:.2f}** ({(med / last_close - 1) * 100:+.2f}%)। "
        f"বেশিরভাগ অনুমান ছিল {np.percentile(allarr[:, 3], 10):.2f} থেকে {np.percentile(allarr[:, 3], 90):.2f}-এর মধ্যে।"
    )

    st.pyplot(draw(x, fut, allarr))

    m = allarr.mean(axis=0)
    st.dataframe(
        pd.DataFrame([{"তারিখ": fut[0].date(), "Open": round(m[0], 2), "High": round(m[1], 2),
                       "Low": round(m[2], 2), "Close": round(m[3], 2)}]),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "মনে রাখবেন: 'কতবার ওপরে' মানে মডেলের নিজেদের মধ্যে মিল, বাজারের গ্যারান্টি নয়। "
        "সাপোর্টে প্যাটার্নের সাথে মিলিয়ে দেখুন আর স্টপ-লস রাখুন।"
    )
