from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

from .llm_insight import LLMInsightResult, generate_llm_insight, load_env_vars
from .model_a import predict_session
from .model_b import recommend_products
from .model_c import predict_cashflow

app = FastAPI(
    title="DataSentinel API",
    description="FastAPI backend integrating Models A (Abandonment), B (Recommendations), and C (CashFlow) with LLM insights.",
    version="1.0.0",
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL A: ABANDONMENT PREDICTION

class SessionFeatures(BaseModel):
    n_views: int = Field(..., description="Number of page views in session")
    n_addtocart: int = Field(..., description="Number of add-to-cart events")
    session_duration_min: float = Field(..., description="Session duration in minutes")
    unique_items_viewed: int = Field(..., description="Number of unique items viewed")
    browse_to_cart_ratio: float = Field(..., description="Ratio of views to cart additions")
    session_velocity: float = Field(..., description="Rate of actions per minute")
    last_gap_min: float = Field(..., description="Minutes since last action")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour when session started")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    has_cart_add: int = Field(..., ge=0, le=1, description="Whether user added to cart")


class AbandonmentResponse(BaseModel):
    abandonment_probability: float = Field(..., description="Predicted abandonment probability (0-1)")
    risk_tier: str = Field(..., description="Risk tier: low, medium, or high")
    top_shap_reasons: List[Dict[str, Any]] = Field(..., description="Top 3 SHAP feature explanations")


@app.post("/models/a/predict", response_model=AbandonmentResponse, tags=["Model A"])
def predict_abandonment(session: SessionFeatures) -> AbandonmentResponse:
    """
    Model A: Predict session abandonment risk
    
    Returns abandonment probability, risk tier, and SHAP explanations.
    """
    try:
        result = predict_session(session.dict())
        return AbandonmentResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model A prediction failed: {str(exc)}")

# MODEL B: PRODUCT RECOMMENDATIONS

class Recommendation(BaseModel):
    item_id: int = Field(..., description="Product item ID")
    predicted_rating: float = Field(..., description="Predicted user rating for item")
    low_friction_score: float = Field(..., description="Low-friction recommendation score")
    final_score: float = Field(..., description="Final ranking score")


class RecommendationResponse(BaseModel):
    user_id: int = Field(..., description="User ID")
    abandonment_risk: str = Field(..., description="Risk tier from Model A")
    recommendations: List[Recommendation] = Field(..., description="Recommended items")
    strategy: str = Field(..., description="Recommendation strategy applied")


@app.post("/models/b/recommend", response_model=RecommendationResponse, tags=["Model B"])
def get_recommendations(
    user_id: int = Query(..., description="User ID"),
    abandonment_risk: str = Query("low", description="Risk tier: low, medium, or high"),
    top_n: int = Query(5, ge=1, le=20, description="Number of recommendations"),
) -> RecommendationResponse:
    """
    Model B: Get risk-adjusted product recommendations
    
    Uses collaborative filtering with low-friction bias for high-risk users.
    """
    try:
        result = recommend_products(
            user_id=user_id,
            abandonment_risk=abandonment_risk,
            top_n=top_n,
        )
        # Convert to pydantic model with proper typing
        recommendations = [
            Recommendation(**rec) for rec in result["recommendations"]
        ]
        return RecommendationResponse(
            user_id=result["user_id"],
            abandonment_risk=result["abandonment_risk"],
            recommendations=recommendations,
            strategy=result["strategy"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model B recommendation failed: {str(exc)}")


# MODEL C: CASH FLOW FORECASTING

class CashFlowResponse(BaseModel):
    status: str = Field(..., description="Status: success or failure")
    forecast_30d: float = Field(None, description="30-day forecast")
    forecast_60d: float = Field(None, description="60-day forecast")
    forecast_90d: float = Field(None, description="90-day forecast")
    trend: str = Field(None, description="Trend direction")
    generated_at: str = Field(..., description="Timestamp when forecast was generated")
    error: Optional[str] = Field(None, description="Error message if status is failure")


@app.get("/models/c/forecast", response_model=CashFlowResponse, tags=["Model C"])
def get_cashflow_forecast(days: int = Query(90, ge=30, le=365, description="Forecast horizon in days")) -> CashFlowResponse:
    """
    Model C: Get cash flow forecast
    
    Returns forecast summary with 30/60/90-day predictions and trend.
    """
    try:
        result = predict_cashflow(days=days)
        
        if result.get("status") == "failure":
            raise HTTPException(status_code=500, detail=result.get("error", "Forecast failed"))
        
        return CashFlowResponse(
            status=result.get("status", "success"),
            forecast_30d=result.get("30_day_forecast"),
            forecast_60d=result.get("60_day_forecast"),
            forecast_90d=result.get("90_day_forecast"),
            trend=result.get("trend"),
            generated_at=result.get("generated_at", ""),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model C forecast failed: {str(exc)}")


# COMBINED INSIGHTS: MODELS A + B + C WITH LLM

class ShapReason(BaseModel):
    feature: str = Field(..., description="Feature name from SHAP explanation")
    shap_value: float = Field(..., description="SHAP impact value for the feature")


class InsightRequest(BaseModel):
    abandonment_score: float = Field(..., ge=0.0, le=1.0, description="Predicted abandonment probability from Model A")
    shap_reasons: List[ShapReason] = Field(..., min_items=1, description="Top SHAP feature reasons from Model A")
    cash_flow_delta: float = Field(..., description="Forecast cash flow delta (positive or negative) from Model C")
    cash_flow_horizon_days: int = Field(90, gt=0, description="Forecast horizon in days")
    forecast_trend: Optional[str] = Field(None, description="Optional cash flow trend text from Model C")


class InsightResponse(BaseModel):
    prompt: str = Field(..., description="The generated prompt sent to the LLM")
    llm_output: dict = Field(..., description="Parsed JSON response from the LLM")
    raw_response: str = Field(..., description="Raw LLM completion text")


@app.post("/insights/combined", response_model=InsightResponse, tags=["LLM Insights"])
def create_combined_insight(request: InsightRequest) -> InsightResponse:
    """
    Generate LLM insights combining Models A, B, and C
    
    Takes abandonment risk (A), recommendations strategy (B), and cash flow forecast (C)
    to generate actionable business insights via LLM.
    """
    try:
        result: LLMInsightResult = generate_llm_insight(
            abandonment_score=request.abandonment_score,
            shap_reasons=[reason.dict() for reason in request.shap_reasons],
            cash_flow_delta=request.cash_flow_delta,
            cash_flow_horizon_days=request.cash_flow_horizon_days,
            forecast_trend=request.forecast_trend,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return InsightResponse(
        prompt=result.prompt,
        llm_output=result.parsed,
        raw_response=result.raw,
    )


# ═════════════════════════════════════════════════════════════════
# HEALTH CHECK & INFO
# ═════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str
    models: Dict[str, str]


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Check API health and model availability"""
    return HealthResponse(
        status="healthy",
        models={
            "model_a": "Abandonment Prediction (XGBoost + SHAP)",
            "model_b": "Product Recommendations (SVD Collaborative Filtering)",
            "model_c": "Cash Flow Forecasting (Prophet)",
            "llm": "LLM Insights Generator",
        }
    )


@app.on_event("startup")
def startup_event() -> None:
    load_env_vars()
    print("✓ DataSentinel API started with all 3 models loaded")
