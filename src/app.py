import streamlit as st
import pandas as pd
import os
from model_c import predict_cashflow

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="DataSentinel Dashboard",
    layout="wide"
)

st.title("📊 DataSentinel — Retail Intelligence Dashboard")

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Behavior",
    "🛒 Recommendations",
    "💰 Cash Flow",
    "🤖 AI Insights"
])

# =========================
# TAB 3: CASH FLOW (YOUR MODEL)
# =========================
with tab3:
    st.header("💰 Cash Flow Forecast (Model C)")

    # Load forecast CSV if exists
    forecast_path = "./outputs/model_c_forecast.csv"

    if os.path.exists(forecast_path):
        df = pd.read_csv(forecast_path)

        st.subheader("📈 Forecast Chart")

        st.line_chart(
            df.set_index("ds")[["yhat"]]
        )

        st.subheader("📊 Forecast Summary")

        summary = predict_cashflow()

        col1, col2, col3 = st.columns(3)

        col1.metric("30 Days", summary["30_day_forecast"])
        col2.metric("60 Days", summary["60_day_forecast"])
        col3.metric("90 Days", summary["90_day_forecast"])

        st.write(f"**Trend:** {summary['trend']}")

    else:
        st.warning("⚠️ Please run model_c.py first to generate forecast")

# =========================
# OTHER TABS (PLACEHOLDER)
# =========================
with tab1:
    st.header("📉 Behavior")
    st.info("Coming soon: session risk & SHAP")

with tab2:
    st.header("🛒 Recommendations")
    st.info("Coming soon: product ranking")

with tab4:
    st.header("🤖 AI Insights")
    st.info("Coming soon: LLM summary")