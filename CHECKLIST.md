# DataSentinel — Project Checklist

> Legend: 👨‍💻 Requires coding skill · 📋 Requires documentation/research skill · ⚙️ Requires automation/no-code skill

---

## Phase 1 — Planning
- [ ] Complete 5W1H in `README.md` · 📋 **Kak Ju** · _07/04_
- [ ] Finish Input & Output section in `TEAM_NOTE.md` · 📋 **Kak Ju** · _07/04_
- [ ] Define and document data schema + data dictionary for both datasets · 📋 **Kak Ju** · _07/04_
- [ ] Set up GitHub repo structure (folders: `/data`, `/models`, `/api`, `/dashboard`, `/n8n`, `/docs`) · 👨‍💻 **Afdhal** · _07/04_
- [ ] Create `requirements.txt` with all dependencies · 👨‍💻 **Afdhal** · _07/04_
- [ ] Write system architecture diagram and commit to `/docs` · 📋 **Kak Ju** · _08/04_

---

## Phase 2 — Data & Feature Engineering
- [X] Download and verify RetailRocket dataset from Kaggle · 👨‍💻 **Afdhal** · _08/04_
- [ ] Load both datasets into DuckDB · 👨‍💻 **Afdhal** · _08/04_
- [ ] Clean and preprocess RetailRocket (handle nulls, parse timestamps, filter noise) · 👨‍💻 **Afdhal** · _08/04_
- [ ] Engineer session-level features (browse-to-cart ratio, session velocity, recency, frequency, time since last click) · 👨‍💻 **Afdhal** · _08/04_
- [ ] Engineer bridge feature: map `abandonment_rate` → `estimated_revenue_loss` · 👨‍💻 **Afdhal** · _09/04_
- [ ] Preprocess teammate's transaction data (date parsing, cost/revenue columns, monthly aggregation) · 👨‍💻 **Afdhal** · _09/04_
- [ ] Commit data pipeline script to `/data` · 👨‍💻 **Afdhal** · _09/04_

---

## Phase 3 — ML Models
### Model A — Abandonment Predictor
- [ ] Split RetailRocket data into train/test sets · 👨‍💻 **Afdhal** · _09/04_
- [ ] Train XGBoost classifier on session features · 👨‍💻 **Afdhal** · _09/04_
- [ ] Evaluate with AUC-ROC and precision-recall curve · 👨‍💻 **Afdhal** · _09/04_
- [ ] Integrate SHAP TreeExplainer (top 3 reasons per prediction) · 👨‍💻 **Afdhal** · _10/04_
- [ ] Save trained model as `.pkl` or `.ubj` to `/models` · 👨‍💻 **Afdhal** · _10/04_
- [ ] Document model parameters and validation metrics in `TEAM_NOTE.md` · 📋 **Kak Ju** · _10/04_

### Model B — Product Recommender
- [ ] Build user-item interaction matrix from RetailRocket purchase events · 👨‍💻 **Afdhal** · _10/04_
- [ ] Train SVD model using scikit-surprise · 👨‍💻 **Afdhal** · _10/04_
- [ ] Evaluate with RMSE on held-out test split · 👨‍💻 **Afdhal** · _10/04_
- [ ] Implement risk-adjusted recommendation logic (high-risk users → lower-friction products) · 👨‍💻 **Afdhal** · _10/04_
- [ ] Save model to `/models` · 👨‍💻 **Afdhal** · _10/04_

### Model C — Cash Flow Forecaster
- [ ] Prepare Prophet-compatible dataframe (`ds`, `y` columns) from transaction data · 👨‍💻 **Afdhal** · _11/04_
- [ ] Train Prophet model, tune seasonality settings · 👨‍💻 **Afdhal** · _11/04_
- [ ] Generate 30/60/90-day projections (baseline vs. intervention scenario) · 👨‍💻 **Afdhal** · _11/04_
- [ ] Evaluate with MAE and MAPE on historical holdout · 👨‍💻 **Afdhal** · _11/04_
- [ ] Save forecast output as CSV to `/data` · 👨‍💻 **Afdhal** · _11/04_

---

## Phase 4 — Backend API
- [ ] Initialise FastAPI project structure in `/api` · 👨‍💻 **Afdhal** · _11/04_
- [ ] Implement `POST /predict/abandonment` endpoint (returns score + SHAP values) · 👨‍💻 **Afdhal** · _11/04_
- [ ] Implement `GET /recommend/{user_id}` endpoint · 👨‍💻 **Afdhal** · _12/04_
- [ ] Implement `GET /cashflow/forecast` endpoint · 👨‍💻 **Afdhal** · _12/04_
- [ ] Implement `POST /insight` endpoint (sends context to LLM, returns plain-English output) · 👨‍💻 **Afdhal / Kak Ili** · _12/04_
- [ ] Add error handling and logging to all endpoints · 👨‍💻 **Afdhal** · _12/04_
- [ ] Test all endpoints locally with sample payloads · 👨‍💻 **Afdhal** · _12/04_
- [ ] Write API endpoint documentation in `README.md` · 📋 **Kak Ju** · _12/04_

---

## Phase 5 — LLM Integration
- [ ] Set up `.env` file with API key (Anthropic or OpenAI) · 👨‍💻 **Kak Ili** · _12/04_
- [ ] Write prompt template that injects abandonment score, SHAP reasons, and cash flow delta · 👨‍💻 **Kak Ili** · _12/04_
- [ ] Implement LLM API call in Python and parse response · 👨‍💻 **Kak Ili** · _12/04_
- [ ] Test prompt outputs and refine for clarity and accuracy · 👨‍💻 **Kak Ili** · _13/04_
- [ ] Connect LLM module to `POST /insight` endpoint · 👨‍💻 **Afdhal / Kak Ili** · _13/04_

---

## Phase 6 — Automation (n8n)
- [ ] Set up n8n locally or via n8n Cloud · ⚙️ **Kak Ju** · _12/04_
- [ ] Build Workflow 1: Scheduled trigger → `GET /cashflow/forecast` → email alert if revenue drops below threshold · ⚙️ **Kak Ju** · _13/04_
- [ ] Build Workflow 2: Webhook trigger → simulates cart abandonment event → `POST /predict/abandonment` → logs result · ⚙️ **Kak Ju** · _13/04_
- [ ] Export both workflows as `.json` and commit to `/n8n` · ⚙️ **Kak Ju** · _13/04_
- [ ] Document both workflows in `TEAM_NOTE.md` (what triggers what, expected output) · 📋 **Kak Ju** · _13/04_

---

## Phase 7 — Frontend Dashboard
- [ ] Initialise Streamlit project in `/dashboard` · 👨‍💻 **Kak Ili** · _11/04_
- [ ] Build Tab 1: Behavior — session risk scores, SHAP explanation per user · 👨‍💻 **Kak Ili** · _12/04_
- [ ] Build Tab 2: Recommendations — product cards ranked by conversion likelihood · 👨‍💻 **Kak Ili** · _12/04_
- [ ] Build Tab 3: Cash Flow — 30/60/90-day forecast chart, baseline vs. intervention · 👨‍💻 **Kak Ili** · _13/04_
- [ ] Build Tab 4: AI Insights — LLM-generated plain-English summary and actions · 👨‍💻 **Kak Ili** · _13/04_
- [ ] Connect all tabs to live FastAPI endpoints · 👨‍💻 **Kak Ili** · _13/04_
- [ ] Test dashboard responsiveness and response times · 👨‍💻 **Kak Ili** · _13/04_

---

## Phase 8 — Integration & Testing
- [ ] Full end-to-end test: data in → model → API → dashboard · 👨‍💻 **Afdhal** · _14/04_
- [ ] Verify n8n workflows trigger correctly against live API · ⚙️ **Kak Ju** · _14/04_
- [ ] Fix bugs from integration test · 👨‍💻 **Afdhal / Kak Ili** · _14/04_
- [ ] Add logging to track model performance in production · 👨‍💻 **Afdhal** · _14/04_
- [ ] Confirm all endpoints return correct responses under edge cases · 👨‍💻 **Afdhal** · _14/04_

---

## Phase 9 — Documentation & Submission Prep
- [ ] Write final `README.md` (setup guide, how to run, architecture overview) · 📋 **Kak Ju** · _15/04_
- [ ] Write `docs/market_analysis.md` (target audience, cost-benefit, scalability) · 📋 **Kak Ju** · _15/04_
- [ ] Write `docs/ethics.md` (data privacy, bias considerations, fair use) · 📋 **Kak Ju** · _15/04_
- [ ] Write `docs/team_division.md` (who did what, commit breakdown) · 📋 **Kak Ju** · _15/04_
- [ ] Verify Git commit history shows parallel work across all members · 👨‍💻 **Afdhal** · _15/04_
- [ ] Review codebase for license compliance (no proprietary dependencies) · 👨‍💻 **Afdhal** · _15/04_
- [ ] Record demo video (scripted: user browses → risk climbs → recommender shifts → cash flow updates → LLM insight fires) · 👨‍💻 **All** · _15/04_
- [ ] Final submission package review · 👨‍💻 **Afdhal** · _16/04_ _(buffer)_

---

## Member Skill Summary

| Member | Skill Profile | Owns |
|---|---|---|
| Afdhal | Python, ML, backend, architecture | Phases 2, 3, 4, 8 core tasks |
| Kak Ili | Simple Python, some automation | Phases 5, 6 LLM, 7 frontend |
| Kak Ju | No-code, n8n, documentation | Phase 6 automation, all docs |