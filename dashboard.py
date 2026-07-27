import streamlit as st
import pandas as pd
import json


st.set_page_config(page_title="LLM Cost Autopilot Dashboard", layout="wide")

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

st.title("🧠 LLM Cost Autopilot — Dashboard")

if df.empty:
    st.warning("No logs found yet. Run some queries through main.py first.")
    st.stop()

# --- Top metrics row ---
col1, col2, col3, col4 = st.columns(4)

total_requests = len(df)
total_cost = df["cost_usd"].sum()
avg_latency = df["latency_sec"].mean()

# estimate what it WOULD have cost if everything used the most expensive tier's pricing
MOST_EXPENSIVE_INPUT_PRICE = 0.59 / 1_000_000
MOST_EXPENSIVE_OUTPUT_PRICE = 0.79 / 1_000_000
hypothetical_cost = (df["input_tokens"] * MOST_EXPENSIVE_INPUT_PRICE +
                     df["output_tokens"] * MOST_EXPENSIVE_OUTPUT_PRICE).sum()
saved = hypothetical_cost - total_cost
saved_pct = (saved / hypothetical_cost * 100) if hypothetical_cost > 0 else 0

col1.metric("Total Requests", total_requests)
col2.metric("Total Cost", f"${total_cost:.6f}")
col3.metric("Estimated Savings", f"${saved:.6f}", f"{saved_pct:.1f}%")
col4.metric("Avg Latency", f"{avg_latency:.2f}s")

st.divider()

# --- Tier distribution ---
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Requests by Tier")
    tier_counts = df["tier"].value_counts()
    st.bar_chart(tier_counts)

with col_b:
    st.subheader("Cost by Tier")
    tier_cost = df.groupby("tier")["cost_usd"].sum()
    st.bar_chart(tier_cost)

st.divider()

# --- Cost over time ---
st.subheader("Cost Over Time (cumulative)")
df_sorted = df.sort_values("datetime")
df_sorted["cumulative_cost"] = df_sorted["cost_usd"].cumsum()
st.line_chart(df_sorted.set_index("datetime")["cumulative_cost"])

st.divider()

# --- Model usage breakdown ---
st.subheader("Requests by Model")
model_counts = df["model"].value_counts()
st.bar_chart(model_counts)

st.divider()

# --- Recent requests table ---
st.subheader("Recent Requests")
display_cols = ["datetime", "tier", "model", "prompt", "cost_usd", "latency_sec"]
available_cols = [c for c in display_cols if c in df.columns]
st.dataframe(
    df[available_cols].sort_values("datetime", ascending=False).head(50),
    use_container_width=True
)