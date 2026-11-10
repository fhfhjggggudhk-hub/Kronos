"""Kronos forecast app. Put this file in the ROOT of your Kronos fork
(next to the `model` folder) and deploy it on Streamlit."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Kronos Forecast", layout="wide")

IST = "Asia/Kolkata"
MODELS = {
    "Kronos-mini (fast, light)": ("NeoQuasar/Kronos-Tokenizer-2k", "NeoQuasar/Kronos-mini", 2048),
    "Kronos-small (better, heavier)": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-small", 512),
}


@st.cache_resource(show_spinner="Loading Kronos model (first time takes a while)...")
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


def run_forecast(pred, x, tf, n, runs):
    x = x.tail(400)
    xdf = pd.DataFrame({
        "open": x.Open.values, "high": x.High.values, "low": x.Low.values,
        "close": x.Close.values, "volume": x.Volume.values,
        "amount": (x.Close * x.Volume).values,
    })
    xts = pd.Series(x.index)
    start = x.index[-1] + pd.Timedelta(days=1)
    fut = pd.bdate_range(start, periods=n) if tf == "Daily" else pd.date_range(start, periods=n, freq="W-FRI")
    yts = pd.Series(fut)
    outs = []
    bar = st.progress(0.0, text="Running Kronos...")
    for i in range(runs):
        p = pred.predict(df=xdf, x_timestamp=xts, y_timestamp=yts, pred_len=n,
                         T=1.0, top_p=0.9, sample_count=1)
        outs.append(p[["open", "high", "low", "close"]].values.astype(float))
        bar.progress((i + 1) / runs, text=f"Run {i + 1}/{runs}")
    bar.empty()
    return fut, np.stack(outs)  # shape: runs x n x 4 (O,H,L,C)


def draw(x, fut, arr, n_hist=60):
    n = len(fut)
    mean = arr.mean(axis=0)
    lo = np.percentile(arr[:, :, 3], 10, axis=0)
    hi = np.percentile(arr[:, :, 3], 90, axis=0)
    h = x.tail(n_hist)
    fig, ax = plt.subplots(figsize=(11, 5))

    def candles(xs, O, H, L, C, alpha):
        for i in range(len(xs)):
            col = "#16a34a" if C[i] >= O[i] else "#dc2626"
            ax.plot([xs[i], xs[i]], [L[i], H[i]], color=col, lw=1, alpha=alpha)
            ax.add_patch(plt.Rectangle((xs[i] - 0.3, min(O[i], C[i])), 0.6,
                                       max(abs(C[i] - O[i]), 1e-9), color=col, alpha=alpha))

    xs_h = np.arange(len(h))
    xs_f = np.arange(len(h), len(h) + n)
    candles(xs_h, h.Open.values, h.High.values, h.Low.values, h.Close.values, 1.0)
    candles(xs_f, mean[:, 0], mean[:, 1], mean[:, 2], mean[:, 3], 0.55)
    ax.fill_between(xs_f, lo, hi, color="#2563eb", alpha=0.18, label="Forecast close: 10-90% range")
    ax.axvline(len(h) - 0.5, ls="--", color="gray")
    dates = list(h.index) + list(fut)
    ticks = np.linspace(0, len(dates) - 1, 7).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[i].strftime("%d %b %y") for i in ticks], fontsize=8)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(-1, len(h) + n)
    return fig


# ------------------------------------------------------------------ UI
st.title("Kronos - next candle forecast")

with st.sidebar:
    exch = st.radio("Exchange", ["NSE", "BSE"], horizontal=True)
    tf = st.radio("Timeframe", ["Daily", "Weekly"], horizontal=True)
    model_key = st.selectbox("Model", list(MODELS))
    n = st.slider("Candles to forecast", 1, 20, 5 if tf == "Daily" else 4)
    runs = st.slider("Number of runs (more = steadier, slower)", 1, 20, 5)
    complete_only = st.checkbox("Ignore running (incomplete) candle", value=True)

name = st.text_input("Stock name (NSE: RELIANCE | BSE: scrip code like 500325)", "RELIANCE").strip().upper()
go = st.button("Forecast", type="primary")

if go and name:
    ticker = name if "." in name else name + (".NS" if exch == "NSE" else ".BO")
    try:
        daily = get_daily(ticker)
    except Exception as e:
        st.error(f"Could not download data for {ticker}: {e}")
        st.stop()
    if len(daily) < 150:
        st.error(f"Not enough data for {ticker}. Check the stock name / exchange.")
        st.stop()

    x = daily if tf == "Daily" else to_weekly(daily)
    now = pd.Timestamp.now(tz=IST)
    today = now.tz_localize(None).normalize()
    mkt_open = now.weekday() < 5 and (now.hour * 60 + now.minute) < 15 * 60 + 45
    last = x.index[-1]
    if complete_only and (last > today or (last == today and mkt_open)):
        x = x.iloc[:-1]
    if len(x) < 100:
        st.error("Not enough candles for this timeframe.")
        st.stop()

    pred = load_predictor(model_key)
    fut, arr = run_forecast(pred, x, tf, n, runs)

    last_close = float(x.Close.iat[-1])
    mean = arr.mean(axis=0)
    c1 = float(mean[0, 3])
    up1 = float((arr[:, 0, 3] > last_close).mean() * 100)

    st.subheader(f"{name} ({tf}) - last candle {x.index[-1].date()}, close {last_close:.2f}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Next candle: forecast close", f"{c1:.2f}", f"{(c1 / last_close - 1) * 100:+.2f}%")
    m2.metric(f"Runs with close above {last_close:.2f}", f"{up1:.0f}%")
    m3.metric("Final candle: forecast close", f"{mean[-1, 3]:.2f}",
              f"{(mean[-1, 3] / last_close - 1) * 100:+.2f}%")

    st.pyplot(draw(x, fut, arr))

    rows = []
    for i in range(len(fut)):
        rows.append({
            "Date": fut[i].date(),
            "Open": round(mean[i, 0], 2), "High": round(mean[i, 1], 2),
            "Low": round(mean[i, 2], 2), "Close": round(mean[i, 3], 2),
            "Change vs last close %": round((mean[i, 3] / last_close - 1) * 100, 2),
            "Green-candle runs %": round(float((arr[:, i, 3] > arr[:, i, 0]).mean() * 100)),
            "Close above last close %": round(float((arr[:, i, 3] > last_close).mean() * 100)),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Faded candles = average of all runs. The percentages show how many runs agreed, "
        "not a real probability. This is an estimate from a model, not advice or a guarantee. "
        "Daily forecast dates skip weekends only, not market holidays."
    )
