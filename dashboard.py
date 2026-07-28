import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide", page_icon="🧠")

@st.cache_data(ttl=5)
def load_logs():
    rows = []
    try:
        with open("data/logs.jsonl", "r") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except FileNotFoundError:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df

df = load_logs()

st.title("🧠 LLM Cost Autopilot")
st.caption("Real-time cost, quality, and routing insight across a 3-tier model system")

if df.empty:
    st.warning("No logs found yet. Run some queries through main.py first.")
    st.stop()

# --- Sidebar filters ---
with st.sidebar:
    st.header("Filters")
    tiers_available = df["tier"].unique().tolist()
    selected_tiers = st.multiselect("Tier", tiers_available, default=tiers_available)
    df = df[df["tier"].isin(selected_tiers)]

    if "fell_back" in df.columns:
        show_fallback_only = st.checkbox("Show fallback events only", value=False)
        if show_fallback_only:
            df = df[df["fell_back"] == True]

# --- Top metrics ---
col1, col2, col3, col4, col5 = st.columns(5)

total_requests = len(df)
total_cost = df["cost_usd"].sum()
avg_latency = df["latency_sec"].mean()

MOST_EXPENSIVE_INPUT_PRICE = 0.59 / 1_000_000
MOST_EXPENSIVE_OUTPUT_PRICE = 0.79 / 1_000_000
hypothetical_cost = (df["input_tokens"] * MOST_EXPENSIVE_INPUT_PRICE +
                     df["output_tokens"] * MOST_EXPENSIVE_OUTPUT_PRICE).sum()
saved = hypothetical_cost - total_cost
saved_pct = (saved / hypothetical_cost * 100) if hypothetical_cost > 0 else 0

fallback_count = int(df["fell_back"].sum()) if "fell_back" in df.columns else 0

col1.metric("Total Requests", f"{total_requests:,}")
col2.metric("Total Cost", f"${total_cost:.5f}")
col3.metric("Savings vs Always-Premium", f"{saved_pct:.1f}%", f"${saved:.5f} saved")
col4.metric("Avg Latency", f"{avg_latency:.2f}s")
col5.metric("Fallback Events", fallback_count, help="Times a rate-limited model auto-downgraded")

st.divider()

# --- Tier distribution + cost side by side ---
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Request Volume by Tier")
    tier_counts = df["tier"].value_counts()
    st.bar_chart(tier_counts, color="#6366f1")

with col_b:
    st.subheader("💰 Cost Contribution by Tier")
    tier_cost = df.groupby("tier")["cost_usd"].sum()
    st.bar_chart(tier_cost, color="#f59e0b")

st.divider()

# --- Cost over time ---
st.subheader("📈 Cumulative Cost Over Time")
df_sorted = df.sort_values("datetime")
df_sorted["cumulative_cost"] = df_sorted["cost_usd"].cumsum()
st.line_chart(df_sorted.set_index("datetime")["cumulative_cost"])

st.divider()

# --- Latency by tier (proves cheap tiers are also faster) ---
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("⚡ Avg Latency by Tier")
    latency_by_tier = df.groupby("tier")["latency_sec"].mean().round(3)
    st.bar_chart(latency_by_tier, color="#10b981")

with col_d:
    st.subheader("🤖 Requests by Model")
    model_counts = df["model"].value_counts()
    st.bar_chart(model_counts, color="#ec4899")

st.divider()

# --- Recent requests table ---
st.subheader("🕓 Recent Requests")
search = st.text_input("Search prompts", "")

display_cols = ["datetime", "tier", "model", "prompt", "cost_usd", "latency_sec"]
available_cols = [c for c in display_cols if c in df.columns]
table_df = df[available_cols].sort_values("datetime", ascending=False)

if search:
    table_df = table_df[table_df["prompt"].str.contains(search, case=False, na=False)]

st.dataframe(table_df.head(50), use_container_width=True, hide_index=True)