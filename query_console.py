import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LLM Cost Autopilot — Query Console", layout="wide", page_icon="🚀")

API_URL = "http://localhost:8000/query"
DEFAULT_API_KEY = os.getenv("LLM_AUTOPILOT_API_KEY", "")

st.title("🚀 Query Console")
st.caption("Ask a question, see which model was chosen, and what it would have cost with other options.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")

with st.form("query_form"):
    prompt = st.text_area("Your prompt", placeholder="e.g. Explain how vaccines work", height=100)

    col1, col2, col3 = st.columns(3)
    with col1:
        routing_mode = st.selectbox("Routing mode", ["balanced", "economy", "quality", "latency"])
    with col2:
        max_cost_usd = st.number_input("Max cost ($, optional)", min_value=0.0, value=0.0, format="%.6f")
    with col3:
        min_quality = st.slider("Min quality (optional)", 0.0, 1.0, 0.0, 0.05)

    submitted = st.form_submit_button("Run Query", use_container_width=True)

if submitted and prompt.strip():
    if not api_key:
        st.error("Please enter an API key in the sidebar.")
        st.stop()

    params = {"prompt": prompt, "routing_mode": routing_mode}
    if max_cost_usd > 0:
        params["max_cost_usd"] = max_cost_usd
    if min_quality > 0:
        params["min_quality"] = min_quality

    headers = {"X-API-Key": api_key}

    with st.spinner("Routing and generating..."):
        try:
            response = requests.post(API_URL, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()


    st.divider()

    # --- Cost comparison across ALL candidates ---
    st.subheader("💰 What Other Models Would Have Cost")

    candidates = data.get("candidates_considered", [])
    rejected = data.get("rejected_candidates", [])

    if candidates:
        chosen_model = data["model"]
        chosen_cost = next((c["estimated_cost"] for c in candidates if c["model"] == chosen_model), data["total_cost_usd"])

        most_expensive = max(candidates, key=lambda c: c["estimated_cost"])
        savings = most_expensive["estimated_cost"] - chosen_cost
        savings_pct = (savings / most_expensive["estimated_cost"] * 100) if most_expensive["estimated_cost"] > 0 else 0

        st.success(f"✅ You saved **${savings:.6f}** ({savings_pct:.1f}%) vs. the most expensive eligible option (`{most_expensive['model']}`)")

        for c in sorted(candidates, key=lambda c: c["estimated_cost"]):
            is_chosen = c["model"] == chosen_model
            label = f"{'⭐ ' if is_chosen else ''}{c['model']}"
            cols = st.columns([2, 1, 1, 1])
            cols[0].write(f"**{label}**" if is_chosen else label)
            cols[1].write(f"${c['estimated_cost']:.6f}")
            cols[2].write(f"Quality: {c['expected_quality']}")
            cols[3].write(f"Latency: {c['avg_latency_sec']}s")
    else:
        st.info("No alternative candidates were eligible for this request.")

    if rejected:
        with st.expander(f"❌ {len(rejected)} model(s) rejected — see why"):
            for r in rejected:
                st.write(f"**{r['model']}** — {r['reason']}")

else:
    st.info("Enter a prompt above and click 'Run Query' to see it in action.")