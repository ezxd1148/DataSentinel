<h1 style="text-align: center;">DataSentinel</h1>

![Header](assets/github-header-banner.png)

<p align="center">
    <img alt="GitHub contributors" src="https://img.shields.io/github/contributors-anon/ezxd1148/DataSentinel">
    <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/t/ezxd1148/DataSentinel">
    <img alt="GitHub Created At" src="https://img.shields.io/github/created-at/ezxd1148/DataSentinel">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ezxd1148/DataSentinel">
</p>

<div align="center">
<h1 align="center" style="color: blue, font-size: 28px, margin: 10px 0;">AI-powered retail intelligence platform</h1>
<p align="center" style="font-size: 10px; margin: 10px 0;">By Team Datasentinel</p>
</div>

# DataSentinel

> AI-powered retail intelligence for small and mid-sized e-commerce businesses — no data team required.

---

## Overview

DataSentinel helps Malaysian e-commerce retailers (Shopify, WooCommerce, Shopee, Lazada) make smarter decisions through predictive AI and plain-English insights. It tackles three core problems:

- **~70% cart abandonment** with no automated recovery
- **No cash flow visibility** beyond the current week
- **No personalized recommendations** for customers

The result: 20–30% of potential revenue quietly walks out the door. DataSentinel closes that gap.

---

## Features

| Component | Model | What it does |
|---|---|---|
| **Behavior Analyzer** | — | Tracks and processes user interaction signals |
| **Abandonment Predictor** | Model A | Flags at-risk carts before they drop |
| **Product Recommender** | Model B | Personalized product suggestions per customer |
| **Cash Flow Forecaster** | Model C | 30/60/90-day revenue projections |
| **LLM Insight Layer** | — | Translates model output into plain-English business insights |

---

## Tech Stack

- **Backend** — FastAPI
- **Frontend** — Streamlit
- **ML Models** — XGBoost (classification), Prophet (time-series forecasting)
- **Insight Layer** — LLM (plain-English report generation)
- **Language** — Python

---

## Target Users

Small and mid-sized Malaysian e-commerce retailers selling on:

- Shopify / WooCommerce
- Shopee / Lazada

Designed for business owners with **no technical background and no dedicated data team**.

---

## Use Case

DataSentinel is built for **real-time and ongoing** use, daily decision-making and forward-looking forecasting. Retailers get actionable answers without touching a spreadsheet or writing a single query.

---

## Getting Started

```bash
git clone https://github.com/ezxd1148/DataSentinel
cd DataSentinel
uv init
uv venv
source .venv/Bin/Activate
uv add -r requirements.txt
uvicorn api.main:app --reload
```

For the dashboard:

```bash
streamlit run dashboard/app.py
```

---

## License

MIT
