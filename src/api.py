from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from .llm_insight import LLMInsightResult, generate_llm_insight, load_env_vars

app = FastAPI(
    title="DataSentinel API",
    description="FastAPI backend for generating LLM insights from abandonment and cashflow signals.",
    version="0.1.0",
)


class ShapReason(BaseModel):
    feature: str = Field(..., description="Feature name from SHAP explanation")
    shap_value: float = Field(..., description="SHAP impact value for the feature")


class InsightRequest(BaseModel):
    abandonment_score: float = Field(..., ge=0.0, le=1.0, description="Predicted abandonment probability")
    shap_reasons: List[ShapReason] = Field(..., min_items=1, description="Top SHAP feature reasons")
    cash_flow_delta: float = Field(..., description="Forecast cash flow delta (positive or negative)")
    cash_flow_horizon_days: int = Field(90, gt=0, description="Forecast horizon in days")
    forecast_trend: Optional[str] = Field(None, description="Optional cash flow trend text")


class InsightResponse(BaseModel):
    prompt: str = Field(..., description="The generated prompt sent to the LLM")
    llm_output: dict = Field(..., description="Parsed JSON response from the LLM")
    raw_response: str = Field(..., description="Raw LLM completion text")


@app.on_event("startup")
def startup_event() -> None:
    load_env_vars()


@app.post("/insight", response_model=InsightResponse)
def create_insight(request: InsightRequest) -> InsightResponse:
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
