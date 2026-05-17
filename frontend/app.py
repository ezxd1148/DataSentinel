import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime
import json

# ═════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DataSentinel Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("DataSentinel — Retail Intelligence Dashboard")
st.markdown("**Integrated Analytics:** Abandonment Risk → Recommendations → Cash Flow Forecasting")

# ═════════════════════════════════════════════════════════════════
# API CONFIG
# ═════════════════════════════════════════════════════════════════
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")

# Check API health
@st.cache_resource
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

api_healthy = check_api_health()
if not api_healthy:
    st.error(f" API not reachable at {API_BASE_URL}. Make sure FastAPI is running: `uvicorn src.api:app --reload`")

# ═════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Abandonment Risk (Model A)",
    "🛒 Recommendations (Model B)",
    "💰 Cash Flow Forecast (Model C)",
    "🤖 AI Insights"
])

# ═════════════════════════════════════════════════════════════════
# TAB 1: ABANDONMENT PREDICTION (MODEL A)
# ═════════════════════════════════════════════════════════════════
with tab1:
    st.header("📉 Abandonment Risk Predictor")
    st.markdown("Analyze session features to predict cart abandonment probability using XGBoost + SHAP")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Session Features")
        n_views = st.number_input("Number of Page Views", min_value=1, value=5)
        n_addtocart = st.number_input("Add-to-Cart Events", min_value=0, value=1)
        session_duration = st.number_input("Session Duration (minutes)", min_value=0.0, value=8.3)
        unique_items = st.number_input("Unique Items Viewed", min_value=1, value=4)
    
    with col2:
        st.subheader("Advanced Features")
        browse_to_cart = st.number_input("Browse-to-Cart Ratio", min_value=0.0, value=5.0)
        velocity = st.number_input("Session Velocity (actions/min)", min_value=0.0, value=0.6)
        last_gap = st.number_input("Last Gap (minutes)", min_value=0.0, value=3.2)
        hour_of_day = st.slider("Hour of Day", min_value=0, max_value=23, value=14)
        day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)
        has_cart_add = st.selectbox("Has Cart Addition", [0, 1], index=1)
    
    if st.button("🔮 Predict Abandonment Risk", key="model_a_predict", use_container_width=True):
        if api_healthy:
            try:
                payload = {
                    "n_views": int(n_views),
                    "n_addtocart": int(n_addtocart),
                    "session_duration_min": float(session_duration),
                    "unique_items_viewed": int(unique_items),
                    "browse_to_cart_ratio": float(browse_to_cart),
                    "session_velocity": float(velocity),
                    "last_gap_min": float(last_gap),
                    "hour_of_day": int(hour_of_day),
                    "day_of_week": int(day_of_week),
                    "has_cart_add": int(has_cart_add),
                }
                
                response = requests.post(
                    f"{API_BASE_URL}/models/a/predict",
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    col_prob, col_tier = st.columns(2)
                    
                    with col_prob:
                        st.metric(
                            "Abandonment Probability",
                            f"{result['abandonment_probability']:.1%}",
                            delta=None
                        )
                    
                    with col_tier:
                        risk_color = {
                            "low": "🟢",
                            "medium": "🟡",
                            "high": "🔴"
                        }
                        st.metric(
                            "Risk Tier",
                            f"{risk_color.get(result['risk_tier'], '❓')} {result['risk_tier'].upper()}"
                        )
                    
                    # SHAP explanations
                    st.subheader("📊 SHAP Feature Importance")
                    shap_df = pd.DataFrame([
                        {"Feature": r["feature"], "SHAP Value": r["shap_value"]}
                        for r in result["top_shap_reasons"]
                    ])
                    
                    col_chart, col_table = st.columns([2, 1])
                    with col_chart:
                        st.bar_chart(
                            shap_df.set_index("Feature")["SHAP Value"],
                            use_container_width=True
                        )
                    with col_table:
                        st.dataframe(shap_df, use_container_width=True, hide_index=True)
                    
                    # Store for use in insights tab
                    st.session_state.model_a_result = result
                    st.success("✓ Prediction complete!")
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error("API not available")


# ═════════════════════════════════════════════════════════════════
# TAB 2: PRODUCT RECOMMENDATIONS (MODEL B)
# ═════════════════════════════════════════════════════════════════
with tab2:
    st.header("🛒 Product Recommendations")
    st.markdown("Get risk-adjusted recommendations using collaborative filtering (SVD)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        user_id = st.number_input("User ID", min_value=1, value=12345)
    
    with col2:
        risk_tier = st.selectbox(
            "Abandonment Risk Tier",
            ["low", "medium", "high"],
            help="Higher risk → lower-friction recommendations"
        )
    
    with col3:
        top_n = st.slider("Number of Recommendations", min_value=3, max_value=20, value=5)
    
    if st.button("📦 Get Recommendations", key="model_b_recommend", use_container_width=True):
        if api_healthy:
            try:
                response = requests.post(
                    f"{API_BASE_URL}/models/b/recommend",
                    params={
                        "user_id": int(user_id),
                        "abandonment_risk": risk_tier,
                        "top_n": int(top_n),
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.metric("Strategy", result["strategy"])
                    
                    st.subheader("📋 Recommended Products")
                    
                    if result["recommendations"]:
                        recs_df = pd.DataFrame(result["recommendations"])
                        recs_df = recs_df.round(4)
                        
                        st.dataframe(
                            recs_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "item_id": st.column_config.NumberColumn("Item ID", format="%d"),
                                "predicted_rating": st.column_config.NumberColumn("Rating", format="%.2f ⭐"),
                                "low_friction_score": st.column_config.NumberColumn("Friction Score", format="%.4f"),
                                "final_score": st.column_config.NumberColumn("Final Score", format="%.4f"),
                            }
                        )
                        
                        # Visualization
                        col_chart, col_stats = st.columns(2)
                        
                        with col_chart:
                            st.subheader("Final Scores Distribution")
                            st.bar_chart(
                                recs_df.set_index("item_id")["final_score"],
                                use_container_width=True
                            )
                        
                        with col_stats:
                            st.metric("Avg Rating", f"{recs_df['predicted_rating'].mean():.2f}")
                            st.metric("Avg Final Score", f"{recs_df['final_score'].mean():.4f}")
                    else:
                        st.info("No recommendations available for this user")
                    
                    st.session_state.model_b_result = result
                    st.success("✓ Recommendations loaded!")
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error("API not available")


# ═════════════════════════════════════════════════════════════════
# TAB 3: CASH FLOW FORECAST (MODEL C)
# ═════════════════════════════════════════════════════════════════
with tab3:
    st.header("💰 Cash Flow Forecasting")
    st.markdown("Prophet-based forecasting of open balance (accounts receivable)")
    
    forecast_days = st.slider("Forecast Horizon (days)", min_value=30, max_value=365, value=90)
    
    if st.button("📈 Generate Cash Flow Forecast", key="model_c_forecast", use_container_width=True):
        if api_healthy:
            try:
                response = requests.get(
                    f"{API_BASE_URL}/models/c/forecast",
                    params={"days": int(forecast_days)},
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("status") == "success":
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "30-Day Forecast",
                                f"${result['forecast_30d']:,.0f}" if result['forecast_30d'] else "N/A"
                            )
                        
                        with col2:
                            st.metric(
                                "60-Day Forecast",
                                f"${result['forecast_60d']:,.0f}" if result['forecast_60d'] else "N/A"
                            )
                        
                        with col3:
                            st.metric(
                                "90-Day Forecast",
                                f"${result['forecast_90d']:,.0f}" if result['forecast_90d'] else "N/A"
                            )
                        
                        with col4:
                            trend_emoji = {
                                "increasing": "📈",
                                "decreasing": "📉",
                                "stable": "➡️"
                            }
                            st.metric(
                                "Trend",
                                f"{trend_emoji.get(result.get('trend', ''), '❓')} {result.get('trend', 'Unknown')}"
                            )
                        
                        # Forecast chart (if CSV exists)
                        forecast_path = "./outputs/model_c_forecast.csv"
                        if os.path.exists(forecast_path):
                            st.subheader("📊 Forecast Chart")
                            forecast_csv = pd.read_csv(forecast_path)
                            
                            # Create display dataframe
                            if "ds" in forecast_csv.columns and "yhat" in forecast_csv.columns:
                                display_df = forecast_csv[["ds", "yhat"]].copy()
                                display_df["ds"] = pd.to_datetime(display_df["ds"])
                                
                                st.line_chart(
                                    display_df.set_index("ds"),
                                    use_container_width=True
                                )
                        else:
                            st.info("📁 Forecast visualization available after running model_c.py")
                        
                        st.caption(f"Generated at: {result.get('generated_at', 'Unknown')}")
                        st.session_state.model_c_result = result
                        st.success("✓ Forecast generated!")
                    else:
                        st.error(f"Forecast Error: {result.get('error', 'Unknown error')}")
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error("API not available")


# ═════════════════════════════════════════════════════════════════
# TAB 4: COMBINED AI INSIGHTS
# ═════════════════════════════════════════════════════════════════
with tab4:
    st.header("🤖 AI Insights")
    st.markdown("Generate LLM insights combining all three models")

    # Defaults so the insights button always has values (rare tab/layout edge cases, linters)
    abandon_prob = 0.5
    shap_reasons = [{"feature": "placeholder", "shap_value": 0.0}]
    cash_delta = 0.0
    trend = "stable"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("From Model A")
        if "model_a_result" in st.session_state:
            abandon_prob = st.session_state.model_a_result["abandonment_probability"]
            shap_reasons = st.session_state.model_a_result["top_shap_reasons"]
            st.metric("Abandonment Probability", f"{abandon_prob:.1%}")
            st.write("**Top SHAP Reasons:**")
            for reason in shap_reasons[:3]:
                st.text(f"  • {reason['feature']}: {reason['shap_value']:.4f}")
        else:
            st.info("Run Model A prediction first")
            abandon_prob = st.number_input("Manual Abandonment Prob", 0.0, 1.0, 0.5)
            shap_reasons = [{"feature": "manual", "shap_value": 0.1}]
    
    with col2:
        st.subheader("From Model C")
        if "model_c_result" in st.session_state:
            cash_delta = st.session_state.model_c_result.get("forecast_90d", 0)
            trend = st.session_state.model_c_result.get("trend", "stable")
            st.metric("90-Day Forecast", f"${cash_delta:,.0f}" if cash_delta else "N/A")
            st.metric("Trend", trend)
        else:
            st.info("Run Model C forecast first")
            cash_delta = st.number_input("Manual Cash Flow Delta", -100000.0, 100000.0, 50000.0)
            trend = "stable"
    
    if st.button("✨ Generate AI Insights", key="generate_insights", use_container_width=True):
        if api_healthy:
            try:
                payload = {
                    "abandonment_score": abandon_prob,
                    "shap_reasons": shap_reasons,
                    "cash_flow_delta": cash_delta,
                    "cash_flow_horizon_days": 90,
                    "forecast_trend": trend,
                }
                
                response = requests.post(
                    f"{API_BASE_URL}/insights/combined",
                    json=payload,
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.subheader("📝 Generated Prompt")
                    st.text_area("", result["prompt"], height=150, disabled=True)
                    
                    st.subheader("🤖 LLM Response")
                    st.json(result["llm_output"])
                    
                    st.subheader("📄 Raw Response")
                    st.text(result["raw_response"])
                    
                    st.success("✓ Insights generated!")
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error("API not available")


# ═════════════════════════════════════════════════════════════════
# SIDEBAR: API STATUS & DOCUMENTATION
# ═════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ System Status")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("API", "🟢 Online" if api_healthy else "🔴 Offline")
    with col2:
        st.metric("URL", f"{API_BASE_URL.split('//')[-1]}")
    
    st.divider()
    
    st.subheader("📚 Model Documentation")
    
    with st.expander("Model A: Abandonment Predictor"):
        st.markdown("""
        **Algorithm:** XGBoost Classifier
        
        **Input:** Session features (views, duration, etc.)
        
        **Output:**
        - Abandonment probability (0-1)
        - Risk tier (low/medium/high)
        - SHAP feature explanations
        
        **Use Case:** Identify high-risk sessions for intervention
        """)
    
    with st.expander("Model B: Recommendations"):
        st.markdown("""
        **Algorithm:** SVD Collaborative Filtering
        
        **Input:** User ID + Abandonment risk tier
        
        **Output:**
        - Top N recommended products
        - Predicted ratings
        - Risk-adjusted scoring
        
        **Use Case:** Low-friction recommendations for at-risk users
        """)
    
    with st.expander("Model C: Cash Flow Forecast"):
        st.markdown("""
        **Algorithm:** Prophet Time Series
        
        **Input:** Historical open balance data
        
        **Output:**
        - 30/60/90-day forecasts
        - Trend direction
        - Confidence intervals
        
        **Use Case:** AR aging & cash flow planning
        """)
    
    st.divider()
    
    st.markdown("""
    **Run FastAPI Backend:**
    ```bash
    uvicorn src.api:app --reload --port 8000
    ```
    
    **Note:** Ensure all three models are trained first:
    - `python src/model_a.py`
    - `python src/model_b.py`
    - `python src/model_c.py`
    """)