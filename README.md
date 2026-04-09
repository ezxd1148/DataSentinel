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

// DataSentinel is an AI-powered retail intelligence platform. 
// It helps businesses: 
//•	Predict cart abandonment 
//•	Recommend products intelligently 
//•	Forecast cash flow (30/60/90 days) 
//•	Generate plain-English business insights using LLM 
//Built using: 
//•	Python ecosystem (FastAPI, Streamlit, XGBoost, Prophet, etc.) 
//Core components: 
//•	Behavior Analyzer 
//•	Abandonment Predictor (Model A) 
//•	Product Recommender (Model B) 
//•	Cash Flow Forecaster (Model C) 
//•	LLM Insight Layer


## Who

// Target Users:
//•	Small and mid-sized Malaysian e-commerce retailers (Shopify, WooCommerce, Shopee, Lazada sellers) 
//•	Business owners with low technical expertise and no data team


## How

// By combining three machine learning models + LLM layer: 
//1.	XGBoost → predicts abandonment risk with SHAP explanations 
//2.	SVD (collaborative filtering) → generates personalized recommendations 
//3.	Prophet → forecasts revenue scenarios 
//System architecture: 
//•	Data processing: pandas, DuckDB 
//•	Backend API: FastAPI 
//•	Frontend dashboard: Streamlit 
//•	Automation: n8n workflows 
//•	Insight generation: LLM (Claude / GPT) 
//Workflow: 
//•	User data → feature engineering → ML predictions → LLM explanation → dashboard insights 
//Outputs: 
//•	Risk scores 
//•	Product recommendations 
//•	Revenue forecasts 
//•	Actionable business advice


## When

// Designed for real-time and ongoing use by retailers for daily decision-making and future forecasting.

## Why

// To solve key problems faced by small e-commerce businesses: 
//•	High cart abandonment (~70%) 
//•	Lack of personalized recommendations 
//•	No visibility into future cash flow 
//To reduce hidden revenue loss (20–30%) 
//To provide data-driven insights without requiring technical expertise 
//To make AI actionable via plain-English explanations


## Where

//Malaysia (initial), SEA expansion; deployed via Streamlit dashboard
