import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
st.set_page_config(page_title="LLM Cost Autopilot", layout="wide", page_icon="🧠")

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


@st.cache_data(ttl=10)
def load_logs():
    try:
        result = (
            supabase.table("model_calls")
            .select(
                "*, requests(prompt, tier, routing_mode, max_cost_usd, "
                "actual_total_cost_usd, budget_accounted_cost_usd, "
                "remaining_budget_usd, budget_exceeded), quality_scores(score, passed)"
            )
            .order("created_at", desc=True)
            .limit(500)
            .execute()
        )
    except Exception as e:
        st.error(f"Could not reach Supabase: {e}")
        return pd.DataFrame()

    if not result.data:
        return pd.DataFrame()

    rows = []
    for row in result.data:
        req = row.get("requests") or {}
        scores = row.get("quality_scores") or []
        score = scores[0]["score"] if scores else None

        rows.append({
            "request_id": row["request_id"],
            "datetime": row["created_at"],
            "model": row["model"],
            "tier": req.get("tier"),
            "routing_mode": req.get("routing_mode"),
            "prompt": req.get("prompt"),
            "budget_limit_usd": req.get("max_cost_usd"),
            "request_actual_cost_usd": req.get("actual_total_cost_usd"),
            "budget_accounted_cost_usd": req.get("budget_accounted_cost_usd"),
            "remaining_budget_usd": req.get("remaining_budget_usd"),
            "budget_exceeded": req.get("budget_exceeded"),
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "cost_usd": row["cost_usd"],
            "actual_cost_usd": row.get("actual_cost_usd"),
            "latency_sec": row["latency_sec"],
            "fell_back": row["fell_back"],
            "is_escalation": row["is_escalation"],
            "quality_score": score,
        })

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


df = load_logs()

st.title("🧠 LLM Cost Autopilot")
st.caption("Real-time cost, quality, and routing insight — live from Supabase")

if df.empty:
    st.warning("No telemetry found yet. Run some queries through main.py or query_console.py first.")
    st.stop()

# --- everything below is your existing dashboard logic, unchanged ---
# (sidebar filters, metrics, charts, tables — all operate on df exactly as before,
#  since df has the same columns your JSONL version produced)

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

request_df = df.drop_duplicates(subset="request_id").copy()
legacy_costs = df.groupby("request_id")["cost_usd"].sum()
request_df["actual_spend_usd"] = request_df["request_actual_cost_usd"].fillna(
    request_df["request_id"].map(legacy_costs)
)
total_requests = len(request_df)
total_cost = request_df["actual_spend_usd"].sum()
avg_latency = df["latency_sec"].mean()

budgeted_requests = request_df[request_df["budget_limit_usd"].notna()]
total_budget = budgeted_requests["budget_limit_usd"].sum()
total_remaining = budgeted_requests["remaining_budget_usd"].sum()
budget_utilization = (
    budgeted_requests["budget_accounted_cost_usd"].sum() / total_budget * 100
    if total_budget > 0 else 0
)

fallback_count = int(df["fell_back"].sum()) if "fell_back" in df.columns else 0

col1.metric("Total Requests", f"{total_requests:,}")
col2.metric("Actual Spend", f"${total_cost:.6f}")
col3.metric("Budget Remaining", f"${total_remaining:.6f}")
col4.metric("Budget Utilization", f"{budget_utilization:.1f}%")
col5.metric("Avg Latency", f"{avg_latency:.2f}s")
st.caption(
    f"Fallback events: {fallback_count:,}. Remaining budget/utilization are shown "
    "for budgeted requests; unknown provider charges are conservatively reserved."
)

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

display_cols = [
    "datetime", "tier", "model", "prompt", "actual_cost_usd",
    "remaining_budget_usd", "latency_sec",
]
available_cols = [c for c in display_cols if c in df.columns]
table_df = df[available_cols].sort_values("datetime", ascending=False)

if search:
    table_df = table_df[table_df["prompt"].str.contains(search, case=False, na=False)]

st.dataframe(table_df.head(50), use_container_width=True, hide_index=True)