# DataSentinelAI

> For team members
>
> Checklist and Planning:
> [Checklist](CHECKLIST.md)
>
> For information gathering:
> [Team Note](TEAM_NOTE.md)

<table>
    <tr>
        <td style="text-align: center; vertical-align: middle;">
            <a href="https://github.com/ezxd1148">
                <sub>
                    <b> Afdhal Saufi </b>
                <sub>
            </a><br />
            Role: Not Available
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="https://github.com/ezxd1148">
                <sub>
                    <b> Julita </b>
                <sub>
            </a><br />
            Role: Not Available
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <a href="https://github.com/ezxd1148">
                <sub>
                    <b> Ili </b>
                <sub>
            </a><br />
            Role: Not Available
        </td>
    </tr>
</table>

## What 

> DataSentinel is an AI-powered retail intelligence platform.
> 
> It helps businesses:
> 
> •	Predict cart abandonment
> 
> •	Recommend products intelligently
> 
> •	Forecast cash flow (30/60/90 days)
> 
> •	Generate plain-English business insights using LLM
> 
> Built using:
> 
> •	Python ecosystem (FastAPI, Streamlit, XGBoost, Prophet, etc.)
> 
> Core components:
> 
> •	Behavior Analyzer
> 
> •	Abandonment Predictor (Model A)
> 
> •	Product Recommender (Model B)
> 
> •	Cash Flow Forecaster (Model C)
> 
> •	LLM Insight Layer


## Who

> Target Users:
> 
> •	Small and mid-sized Malaysian e-commerce retailers (Shopify, WooCommerce, Shopee, Lazada sellers)
> 
> •	Business owners with low technical expertise and no data team
> 


## How

> By combining three machine learning models + LLM layer:
> 
> 1.	XGBoost → predicts abandonment risk with SHAP explanations
>    
> 2.	SVD (collaborative filtering) → generates personalized recommendations
>    
> 3.	Prophet → forecasts revenue scenarios
>    
> System architecture:
> 
> •	Data processing: pandas, DuckDB
> 
> •	Backend API: FastAPI
> 
> •	Frontend dashboard: Streamlit
> 
> •	Automation: n8n workflows
> 
> •	Insight generation: LLM (Claude / GPT)
> 
> ## LLM Integration

1. Copy `.env.example` to `.env` and set `LLM_PROVIDER`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`.
2. Start the FastAPI backend with:
   `uvicorn src.api:app --reload --host 127.0.0.1 --port 8000`
3. Call `POST /insight` with JSON payload:
   - `abandonment_score`: float between 0.0 and 1.0
   - `shap_reasons`: list of `{feature, shap_value}`
   - `cash_flow_delta`: numeric cash flow change
   - `cash_flow_horizon_days`: optional forecast window
   - `forecast_trend`: optional trend label

Example request body:

```json
{
  "abandonment_score": 0.72,
  "shap_reasons": [
    {"feature": "n_addtocart", "shap_value": 0.18},
    {"feature": "session_velocity", "shap_value": 0.12},
    {"feature": "hour_of_day", "shap_value": -0.05}
  ],
  "cash_flow_delta": -12000.0,
  "cash_flow_horizon_days": 90,
  "forecast_trend": "declining"
}
```

The endpoint returns a JSON insight summary, parsed LLM output, and the exact prompt sent to the model.

Workflow:
> 
> •	User data → feature engineering → ML predictions → LLM explanation → dashboard insights
> 
> Outputs:
> 
> •	Risk scores
>
> •	Product recommendations
> 
> •	Revenue forecasts
> 
> •	Actionable business advice
> 


## When

> Designed for real-time and ongoing use by retailers for daily decision-making and future forecasting.

## Why

> To solve key problems faced by small e-commerce businesses:
> 
> •	High cart abandonment (~70%)
> 
> •	Lack of personalized recommendations
> 
> •	No visibility into future cash flow
> 
> To reduce hidden revenue loss (20–30%)
> 
> To provide data-driven insights without requiring technical expertise
>  
> To make AI actionable via plain-English explanations
> 


## Where

> Malaysia (initial), SEA expansion; deployed via Streamlit dashboard
