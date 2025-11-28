import os
import math
import json
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="CPSC 4640 · Exposure Bias Dashboard",
    layout="wide",
    page_icon="📈",
)

# --------------------------
# Data loading
# --------------------------

def load_local_otto_jsonl(path: str, max_sessions: int = 5000) -> pd.DataFrame:
    rows: List[Dict] = []
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r") as f:
        for si, line in enumerate(f):
            if si >= max_sessions:
                break
            try:
                obj = json.loads(line)
                session_id = obj.get("session") or obj.get("session_id") or si
                for ev in obj.get("events", []):
                    etype = ev.get("type", "click")
                    if etype == "clicks":
                        etype = "click"
                    if etype == "carts":
                        etype = "cart"
                    if etype == "orders":
                        etype = "order"
                    rows.append({
                        "session": session_id,
                        "aid": int(ev.get("aid", -1)),
                        "ts": int(ev.get("ts", 0)),
                        "type": etype,
                    })
            except Exception:
                continue
    return pd.DataFrame(rows)


def load_dataset(max_sessions: int) -> Tuple[pd.DataFrame, np.ndarray, str]:
    candidate_paths = [
        os.path.join("data", "otto-recsys-train.jsonl"),
        os.path.join("data", "train.jsonl"),
        os.path.join("data", "otto-recsys-test.jsonl"),
        os.path.join("..", "data", "otto-recsys-train.jsonl"),
        os.path.join("..", "data", "train.jsonl"),
        os.path.join("..", "data", "otto-recsys-test.jsonl"),
    ]
    available_paths = [p for p in candidate_paths if os.path.exists(p) and os.path.getsize(p) > 0]

    if available_paths:
        # Distribute the cap across files
        per_file = max(50, max_sessions // len(available_paths))
        dfs = [load_local_otto_jsonl(p, max_sessions=per_file) for p in available_paths]
        df = pd.concat(dfs, ignore_index=True).dropna(subset=["aid", "type"])
        source = available_paths[0]
    else:
        # Synthetic fallback (same shape as notebook)
        num_items = 500
        num_sessions = max_sessions
        item_ids = np.arange(1, num_items + 1)
        popularity = np.random.zipf(1.2, size=num_items)
        popularity = popularity / popularity.sum()
        rows: List[Dict] = []
        for s in range(num_sessions):
            session_len = np.random.randint(3, 15)
            seen = np.random.choice(item_ids, size=session_len, replace=True, p=popularity)
            tstamp0 = np.random.randint(1_700_000_000, 1_700_500_000)
            for k, iid in enumerate(seen):
                event_type = np.random.choice(["click", "cart", "order"], p=[0.85, 0.1, 0.05])
                rows.append({
                    "session": s,
                    "aid": int(iid),
                    "ts": tstamp0 + int(k * np.random.randint(1, 30)),
                    "type": event_type
                })
        df = pd.DataFrame(rows)
        source = "synthetic"

    item_ids = np.array(sorted(df["aid"].unique()))
    return df, item_ids, source


# --------------------------
# Metrics
# --------------------------

def gini_from_counts(counts: np.ndarray) -> float:
    x = np.asarray(counts, dtype=float)
    if x.size == 0:
        return float("nan")
    mean_x = np.mean(x)
    if mean_x == 0:
        return 0.0
    diffsum = np.abs(x[:, None] - x[None, :]).sum()
    n = x.size
    return float(diffsum / (2 * n * n * mean_x))


def hhi_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return float("nan")
    shares = counts / total
    return float(np.sum(np.square(shares)))


def exposure_counts(df: pd.DataFrame, event: str) -> pd.Series:
    sub = df[df["type"] == event]
    return sub.groupby("aid").size()


def head_tail_split(counts: pd.Series, head_q: float = 0.10) -> Tuple[set, set]:
    counts_sorted = counts.sort_values(ascending=False)
    n_head = max(1, int(math.ceil(head_q * len(counts_sorted))))
    head_ids = set(counts_sorted.index[:n_head])
    tail_ids = set(counts_sorted.index[n_head:])
    return head_ids, tail_ids


def compute_stage_metrics(df: pd.DataFrame, item_ids: np.ndarray, events: List[str], head_q: float) -> pd.DataFrame:
    rows: List[Dict] = []
    for event in events:
        c = exposure_counts(df, event).reindex(item_ids, fill_value=0)
        g = gini_from_counts(c.values)
        h = hhi_from_counts(c.values)
        head_ids, tail_ids = head_tail_split(c, head_q=head_q)
        total = c.sum()
        tail_share = (c.loc[list(tail_ids)].sum() / total) if total > 0 else float("nan")
        rows.append({"event": event, "gini": g, "hhi": h, f"tail_share@{int(head_q*100)}%": float(tail_share)})
    return pd.DataFrame(rows)


def compute_position_bias(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(["session", "ts"])
    df_sorted = df_sorted.assign(pos=df_sorted.groupby("session").cumcount() + 1)
    pos_stats = (
        df_sorted
        .groupby("pos")["type"]
        .value_counts(normalize=True)
        .rename("rate")
        .reset_index()
    )
    return pos_stats


def compute_decile_shares(df: pd.DataFrame, item_ids: np.ndarray, base_event: str, events: List[str], bins: int) -> pd.DataFrame:
    def labels(n: int) -> List[str]:
        return [f"D{d}" for d in range(1, n + 1)]

    all_counts = {e: exposure_counts(df, e).reindex(item_ids, fill_value=0) for e in events}
    base = all_counts[base_event].copy()
    rank = base.rank(method="first", ascending=True)
    quantiles = pd.qcut(rank, q=bins, labels=labels(bins))

    frames: List[pd.DataFrame] = []
    for e, counts in all_counts.items():
        tmp = pd.DataFrame({"aid": counts.index, "count": counts.values, "decile": quantiles.values})
        agg = tmp.groupby("decile")["count"].sum().reset_index()
        total = agg["count"].sum()
        agg["share"] = (agg["count"] / total) if total > 0 else np.nan
        agg["event"] = e
        frames.append(agg)
    return pd.concat(frames, ignore_index=True)


def compute_early_later(df: pd.DataFrame, item_ids: np.ndarray, early_k: int, events: List[str]) -> pd.DataFrame:
    df_sorted = df.sort_values(["session", "ts"]).assign(pos=lambda x: x.groupby("session").cumcount() + 1)
    early = df_sorted[df_sorted["pos"] <= early_k]
    early_counts = early.groupby("aid").size().reindex(item_ids, fill_value=0)
    rank_e = early_counts.rank(method="first", ascending=True)
    q_labels = [f"E{d}" for d in range(1, 11)]
    early_decile = pd.qcut(rank_e, q=10, labels=q_labels)

    frames: List[pd.DataFrame] = []
    for e in events:
        cur = df_sorted[df_sorted["type"] == e]
        tmp = cur.groupby("aid").size().reindex(item_ids, fill_value=0)
        tmp_df = pd.DataFrame({"aid": item_ids, "count": tmp.values, "edec": early_decile.values})
        agg = tmp_df.groupby("edec")["count"].sum().reset_index()
        total = agg["count"].sum()
        agg["share"] = (agg["count"] / total) if total > 0 else np.nan
        agg["event"] = e
        frames.append(agg)
    return pd.concat(frames, ignore_index=True)


def fit_slope_y_vs_x(y: np.ndarray) -> float:
    x = np.arange(1, len(y) + 1)
    if np.all(~np.isfinite(y)):
        return float("nan")
    coef = np.polyfit(x, y, 1)[0]
    return float(coef)


# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.title("Controls")
max_sessions = st.sidebar.slider("Sample sessions (cap)", min_value=100, max_value=10000, value=int(os.getenv("MAX_SESSIONS", "300")), step=100)
events = st.sidebar.multiselect("Events", options=["click", "cart", "order"], default=["click", "cart", "order"])
head_q = st.sidebar.slider("Head fraction (for tail-share)", min_value=0.05, max_value=0.5, value=0.10, step=0.05)
early_k = st.sidebar.slider("Early window (positions)", min_value=1, max_value=10, value=3, step=1)
decile_bins = st.sidebar.slider("Popularity decile bins", min_value=5, max_value=20, value=10, step=1)
scatter_item_cap = st.sidebar.slider("Scatter: max items", min_value=100, max_value=5000, value=1000, step=100)
time_bucket = st.sidebar.selectbox("Temporal drift bucket", options=["H (hourly)", "D (daily)"], index=1)

st.sidebar.caption("Tip: Reduce sessions for faster iteration; increase for smoother curves.")


# --------------------------
# Load data
# --------------------------
with st.spinner("Loading data..."):
    df, item_ids, source = load_dataset(max_sessions=max_sessions)

st.caption(f"Loaded rows: {len(df):,} | Source: {source}")

if len(events) == 0:
    st.warning("Select at least one event type to display.")
    st.stop()


# --------------------------
# KPI cards: Stage metrics
# --------------------------
st.markdown("## Stage-wise inequality and long-tail")
metrics_df = compute_stage_metrics(df, item_ids, events, head_q)

cols = st.columns(len(events))
for i, e in enumerate(events):
    sub = metrics_df[metrics_df["event"] == e]
    if sub.empty:
        continue
    gini_val = float(sub["gini"].iloc[0])
    hhi_val = float(sub["hhi"].iloc[0])
    tail_col = [c for c in sub.columns if c.startswith("tail_share@")][0]
    tail_val = float(sub[tail_col].iloc[0])
    with cols[i]:
        st.metric(label=f"{e} · Gini", value=f"{gini_val:.3f}")
        st.metric(label=f"{e} · HHI", value=f"{hhi_val:.3f}")
        st.metric(label=f"{e} · Tail-share ({tail_col.split('@')[1]})", value=f"{tail_val:.3f}")

st.download_button(
    "Download metrics (CSV)",
    data=metrics_df.to_csv(index=False).encode("utf-8"),
    file_name="stage_metrics.csv",
    mime="text/csv",
)


# --------------------------
# Exposure curves
# --------------------------
st.markdown("## Exposure curves by event")
tabs = st.tabs(events)
for i, e in enumerate(events):
    with tabs[i]:
        counts = exposure_counts(df, e).reindex(item_ids, fill_value=0).sort_values(ascending=False).reset_index(drop=True)
        fig = px.line(y=counts.values, labels={"x": "Items (sorted by exposure)", "y": "Exposures"}, title=f"Exposure curve: {e}")
        st.plotly_chart(fig, use_container_width=True)
        html_data = fig.to_html(include_plotlyjs="cdn", full_html=False).encode("utf-8")
        st.download_button(f"Download {e} curve (HTML)", data=html_data, file_name=f"exposure_curve_{e}.html", mime="text/html")


# --------------------------
# Lorenz curves (inequality lens)
# --------------------------
st.markdown("## Lorenz curves (cumulative exposure vs cumulative items)")
lorenz_tabs = st.tabs(events)
for i, e in enumerate(events):
    with lorenz_tabs[i]:
        counts = exposure_counts(df, e).reindex(item_ids, fill_value=0).values
        counts_sorted = np.sort(counts)  # ascending for Lorenz
        total = float(np.sum(counts_sorted))
        if total > 0 and counts_sorted.size > 0:
            cum_items = np.linspace(0, 1, counts_sorted.size + 1)
            cum_expo = np.concatenate([[0.0], np.cumsum(counts_sorted) / total])
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(x=cum_items, y=cum_expo, mode="lines", name=f"Lorenz {e}"))
            fig_l.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Equality", line=dict(dash="dash")))
            fig_l.update_layout(xaxis_title="Cumulative share of items", yaxis_title="Cumulative share of exposures", title=f"Lorenz curve: {e}")
            st.plotly_chart(fig_l, use_container_width=True)
            st.download_button(f"Download Lorenz ({e}) (HTML)", data=fig_l.to_html(include_plotlyjs="cdn", full_html=False).encode("utf-8"), file_name=f"lorenz_{e}.html", mime="text/html")
        else:
            st.info("No data for Lorenz curve.")


# --------------------------
# Position-bias proxy
# --------------------------
st.markdown("## Position-bias proxy (engagement rate by intra-session position)")
pos_stats = compute_position_bias(df)
fig_pos = go.Figure()
max_pos_to_show = int(min(25, pos_stats["pos"].max() if not pos_stats.empty else 0))
for e in events:
    sub = pos_stats[pos_stats["type"] == e].sort_values("pos")
    sub = sub[sub["pos"] <= max_pos_to_show]
    fig_pos.add_trace(go.Scatter(x=sub["pos"], y=sub["rate"], mode="lines+markers", name=e))
fig_pos.update_layout(xaxis_title="Position", yaxis_title="Engagement rate", legend_title="Event", title="Position-bias proxy")
st.plotly_chart(fig_pos, use_container_width=True)
st.download_button("Download position-bias plot (HTML)", data=fig_pos.to_html(include_plotlyjs="cdn", full_html=False).encode("utf-8"), file_name="position_bias.html", mime="text/html")


# --------------------------
# Decile shares (head vs tail)
# --------------------------
st.markdown("## Popularity deciles (base: click counts) · share by event")
decile_df = compute_decile_shares(df, item_ids, base_event="click", events=events, bins=decile_bins)

decile_tabs = st.tabs(events)
for i, e in enumerate(events):
    with decile_tabs[i]:
        sub = decile_df[decile_df["event"] == e].sort_values("decile")
        fig = px.bar(sub, x="decile", y="share", labels={"share": "Share of exposures"}, title=f"Decile exposure share: {e}")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(f"Download decile bars ({e}) (CSV)", data=sub.to_csv(index=False).encode("utf-8"), file_name=f"decile_shares_{e}.csv", mime="text/csv")


# --------------------------
# Decile conversion rates (click→cart, click→order, cart→order)
# --------------------------
st.markdown("## Conversion rates by popularity decile")
all_clicks = exposure_counts(df, "click").reindex(item_ids, fill_value=0)
all_carts = exposure_counts(df, "cart").reindex(item_ids, fill_value=0)
all_orders = exposure_counts(df, "order").reindex(item_ids, fill_value=0)

base_clicks = all_clicks.copy()
rank = base_clicks.rank(method="first", ascending=True)
labels = [f"D{d}" for d in range(1, decile_bins + 1)]
quantiles = pd.qcut(rank, q=decile_bins, labels=labels)

conv_df = pd.DataFrame({
    "aid": item_ids,
    "decile": quantiles.values,
    "clicks": all_clicks.values,
    "carts": all_carts.values,
    "orders": all_orders.values,
})
agg = conv_df.groupby("decile")[["clicks", "carts", "orders"]].sum().reset_index()
eps = 1e-9
agg["cart_per_click"] = agg["carts"] / np.maximum(agg["clicks"], eps)
agg["order_per_click"] = agg["orders"] / np.maximum(agg["clicks"], eps)
agg["order_per_cart"] = agg["orders"] / np.maximum(agg["carts"], eps)

conv_tabs = st.tabs(["click→cart", "click→order", "cart→order"])
metric_map = {
    "click→cart": "cart_per_click",
    "click→order": "order_per_click",
    "cart→order": "order_per_cart",
}
for i, name in enumerate(["click→cart", "click→order", "cart→order"]):
    with conv_tabs[i]:
        m = metric_map[name]
        fig_c = px.bar(agg.sort_values("decile"), x="decile", y=m, title=f"Conversion {name} by decile", labels={m: "Rate"})
        st.plotly_chart(fig_c, use_container_width=True)
        st.download_button(f"Download conversions ({name}) (CSV)", data=agg[["decile", m]].to_csv(index=False).encode("utf-8"), file_name=f"conversion_{m}.csv", mime="text/csv")


# --------------------------
# Early→later compounding
# --------------------------
st.markdown("## Early exposure → later-stage share")
rq3_df = compute_early_later(df, item_ids, early_k=early_k, events=events)

trend_cols = st.columns(len(events))
for i, e in enumerate(events):
    sub = rq3_df[rq3_df["event"] == e].sort_values("edec")
    y = sub["share"].values if not sub.empty else np.array([])
    slope = fit_slope_y_vs_x(y) if y.size > 0 else float("nan")
    e1 = float(sub[sub["edec"] == "E1"]["share"]) if not sub.empty else float("nan")
    e10 = float(sub[sub["edec"] == "E10"]["share"]) if not sub.empty else float("nan")
    with trend_cols[i]:
        st.metric(label=f"{e} · slope(E1→E10)", value=f"{slope:.3f}")
        if np.isfinite(e1) and np.isfinite(e10):
            st.metric(label=f"{e} · E10/E1", value=f"{(e10 / max(e1, 1e-9)):.2f}")

trend_tabs = st.tabs(events)
for i, e in enumerate(events):
    with trend_tabs[i]:
        sub = rq3_df[rq3_df["event"] == e].sort_values("edec")
        fig = px.line(sub, x="edec", y="share", markers=True, title=f"Later-stage share by early-exposure decile: {e}", labels={"edec": "Early exposure decile", "share": "Share"})
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(f"Download early→later plot ({e}) (HTML)", data=fig.to_html(include_plotlyjs="cdn", full_html=False).encode("utf-8"), file_name=f"early_later_{e}.html", mime="text/html")


# --------------------------
# Popularity vs efficiency scatter
# --------------------------
st.markdown("## Popularity vs efficiency (orders per click)")
scatter_df = pd.DataFrame({
    "aid": item_ids,
    "clicks": all_clicks.values,
    "orders": all_orders.values,
})
scatter_df["efficiency"] = scatter_df["orders"] / np.maximum(scatter_df["clicks"], 1e-9)
# Assign deciles (same as above)
scatter_df = scatter_df.assign(decile=quantiles.values)
# Limit by clicks to top-N for readability
scatter_df = scatter_df.sort_values("clicks", ascending=False).head(int(scatter_item_cap))
fig_s = px.scatter(scatter_df, x="clicks", y="efficiency", color="decile", size="clicks", hover_data=["aid"], title="Popularity (clicks) vs efficiency (orders/click)")
st.plotly_chart(fig_s, use_container_width=True)
st.download_button("Download scatter data (CSV)", data=scatter_df.to_csv(index=False).encode("utf-8"), file_name="pop_vs_efficiency.csv", mime="text/csv")


# --------------------------
# Temporal drift of Gini by event
# --------------------------
st.markdown("## Temporal drift of inequality (Gini over time)")
if not df.empty:
    dt = pd.to_datetime(df["ts"], unit="ms")
    freq = "D" if time_bucket.startswith("D") else "H"
    drift_rows: List[Dict] = []
    for e in events:
        sub = df[df["type"] == e].copy()
        if sub.empty:
            continue
        sub = sub.assign(dt=pd.to_datetime(sub["ts"], unit="ms"))
        # For each bucket, compute gini across item exposure counts in that bucket
        buckets = sub.set_index("dt").groupby(pd.Grouper(freq=freq))
        for bucket_time, g in buckets:
            counts = g.groupby("aid").size().values
            if counts.size == 0:
                continue
            drift_rows.append({"time": bucket_time, "event": e, "gini": gini_from_counts(counts)})
    drift_df = pd.DataFrame(drift_rows).sort_values("time")
    if not drift_df.empty:
        fig_d = px.line(drift_df, x="time", y="gini", color="event", title=f"Gini over time ({freq}-bucketed)")
        st.plotly_chart(fig_d, use_container_width=True)
        st.download_button("Download temporal drift (CSV)", data=drift_df.to_csv(index=False).encode("utf-8"), file_name="temporal_gini.csv", mime="text/csv")
    else:
        st.info("Not enough data to compute temporal drift.")
else:
    st.info("No data available for temporal drift.")


# --------------------------
# Privacy and methodology
# --------------------------
with st.expander("Privacy and methodology"):
    st.markdown(
        """
**Sensitive signals present**
- Session identifiers (stable or linkable), fine-grained timestamps
- Behavioral traces (click/cart/order sequences); rare paths can be quasi-identifiers
- Item IDs encode preferences; certain categories could be sensitive

**Risks**
- Re-identification via unique behavior + timestamps
- Preference inference even without PII

**Balancing transparency vs. confidentiality**
- Aggregate-only views with k-anonymity (e.g., k≥50) and coarse time windows
- Differential privacy noise for public metrics (Gini, tail-share, position slope, early-lift)
- On-device per-user views; avoid exporting raw traces
- Rotate pseudonymous IDs, minimize fields, bucket timestamps, shorten retention
- Role-based access, audit logs, consent/opt-out
        """
    )


# --------------------------
# Footer
# --------------------------
st.caption("CPSC 4640 · Exposure Bias Dashboard — rich-get-richer dynamics across clicks → carts → orders")


