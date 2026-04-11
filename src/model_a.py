"""
DataSentinel — Model A: Abandonment Predictor
==============================================
Pipeline:
  1. Load RetailRocket events CSV
  2. Engineer session-level features
  3. Train XGBoost classifier
  4. Evaluate (AUC-ROC, classification report)
  5. SHAP explanations
  6. Save model + feature list

RetailRocket dataset:
  Download from: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
  Required file: events.csv
  Place it in: ./data/events.csv
"""

import os
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    RocCurveDisplay,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG (paths relative to this package so API cwd does not matter)
# ─────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
DATA_PATH = SRC_DIR / "data" / "events.csv"
MODEL_PATH = SRC_DIR / "models" / "model_a.pkl"
FEAT_PATH = SRC_DIR / "models" / "model_a_features.pkl"
OUTPUT_DIR = SRC_DIR / "outputs"
SAMPLE_SIZE = 500_000   # rows to load (None = full dataset, reduce if RAM limited)
TEST_SIZE   = 0.2
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_PATH.parent, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────
def load_data(path: str, sample: int | None = None) -> pd.DataFrame:
    print("[ 1/6 ] Loading events data...")
    df = pd.read_csv(
        path,
        nrows=sample,
        dtype={
            "visitorid": "int32",
            "itemid":    "int32",
            "event":     "category",
        },
    )
    # RetailRocket timestamp is in milliseconds
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)
    print(f"         Loaded {len(df):,} events | {df['visitorid'].nunique():,} unique visitors")
    return df


# ─────────────────────────────────────────────
# STEP 2 — BUILD SESSIONS
# ─────────────────────────────────────────────
def build_sessions(df: pd.DataFrame, session_gap_minutes: int = 30) -> pd.DataFrame:
    """
    Split each visitor's events into sessions using a 30-minute inactivity gap.
    A session that ends with a 'transaction' event = NOT abandoned (label 0).
    A session that ends with 'view' or 'addtocart' = abandoned (label 1).
    """
    print("[ 2/6 ] Building sessions...")

    df = df.copy()
    df["prev_time"] = df.groupby("visitorid")["timestamp"].shift(1)
    df["gap_minutes"] = (
        (df["timestamp"] - df["prev_time"]).dt.total_seconds() / 60
    ).fillna(0)

    # New session flag: first event of visitor OR gap > threshold
    df["new_session"] = (df["gap_minutes"] > session_gap_minutes) | (df["gap_minutes"] == 0)
    df["session_id"] = df.groupby("visitorid")["new_session"].cumsum()
    df["session_id"] = df["visitorid"].astype(str) + "_" + df["session_id"].astype(str)

    return df


# ─────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw events into one row per session with predictive features.
    """
    print("[ 3/6 ] Engineering features...")

    grp = df.groupby("session_id")

    # Event counts
    event_counts = (
        df.groupby(["session_id", "event"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"view": "n_views", "addtocart": "n_addtocart", "transaction": "n_transactions"})
    )
    for col in ["n_views", "n_addtocart", "n_transactions"]:
        if col not in event_counts.columns:
            event_counts[col] = 0

    features = event_counts.copy()

    # Session duration (minutes)
    session_times = grp["timestamp"].agg(["min", "max"])
    features["session_duration_min"] = (
        (session_times["max"] - session_times["min"]).dt.total_seconds() / 60
    )

    # Unique items viewed
    features["unique_items_viewed"] = grp["itemid"].nunique()

    # Browse-to-cart ratio (views per cart add; 0 if no cart adds)
    features["browse_to_cart_ratio"] = (
        features["n_views"] / (features["n_addtocart"] + 1)
    )

    # Session velocity: events per minute (avoid div by zero)
    total_events = features["n_views"] + features["n_addtocart"] + features["n_transactions"]
    features["session_velocity"] = total_events / (features["session_duration_min"] + 1)

    # Time gap between last two events (recency signal)
    def last_gap(g):
        times = g.sort_values().values
        if len(times) < 2:
            return 0.0
        return (times[-1] - times[-2]).astype("timedelta64[s]").astype(float) / 60

    features["last_gap_min"] = grp["timestamp"].apply(last_gap)

    # Hour of day when session started (time-of-day signal)
    features["hour_of_day"] = grp["timestamp"].min().dt.hour

    # Day of week (0=Mon … 6=Sun)
    features["day_of_week"] = grp["timestamp"].min().dt.dayofweek

    # Has cart add flag
    features["has_cart_add"] = (features["n_addtocart"] > 0).astype(int)

    # ── LABEL ──────────────────────────────────
    # 0 = purchased (not abandoned), 1 = abandoned
    features["abandoned"] = (features["n_transactions"] == 0).astype(int)

    # Drop sessions with zero views (noise)
    features = features[features["n_views"] > 0].copy()

    # Drop the raw transaction count from features (it would leak the label)
    feature_cols = [
        "n_views",
        "n_addtocart",
        "session_duration_min",
        "unique_items_viewed",
        "browse_to_cart_ratio",
        "session_velocity",
        "last_gap_min",
        "hour_of_day",
        "day_of_week",
        "has_cart_add",
    ]

    print(f"         Sessions: {len(features):,} | "
          f"Abandoned: {features['abandoned'].sum():,} "
          f"({features['abandoned'].mean()*100:.1f}%)")

    return features, feature_cols


# ─────────────────────────────────────────────
# STEP 4 — TRAIN MODEL
# ─────────────────────────────────────────────
def train_model(features: pd.DataFrame, feature_cols: list) -> tuple:
    print("[ 4/6 ] Training XGBoost...")

    X = features[feature_cols].fillna(0)
    y = features["abandoned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Class weight to handle imbalance (purchases are rare)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    print(f"         Trees trained: {model.n_estimators}")
    return model, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# STEP 5 — EVALUATE
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, feature_cols):
    print("[ 5/6 ] Evaluating model...")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)

    print(f"\n  ── Metrics ────────────────────────")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  Avg Precision:     {ap:.4f}")
    print(f"\n  ── Classification Report ──────────")
    print(classification_report(y_test, y_pred, target_names=["Purchased", "Abandoned"]))

    # Save ROC curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0], name="XGBoost")
    axes[0].set_title("ROC Curve — Abandonment Predictor")

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec, prec, color="steelblue")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve (AP={ap:.3f})")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_a_evaluation.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/model_a_evaluation.png")

    return auc, ap


# ─────────────────────────────────────────────
# STEP 6 — SHAP EXPLANATIONS
# ─────────────────────────────────────────────
def compute_shap(model, X_train, X_test, feature_cols):
    print("[ 6/6 ] Computing SHAP values...")

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test)

    # Global feature importance plot
    plt.figure()
    shap.summary_plot(
        shap_vals, X_test,
        feature_names=feature_cols,
        show=False,
        plot_size=(10, 5),
    )
    plt.title("SHAP Feature Importance — Model A")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_a_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/model_a_shap_summary.png")

    return explainer


# ─────────────────────────────────────────────
# INFERENCE HELPER  (used by FastAPI)
# ─────────────────────────────────────────────
def predict_session(session_dict: dict) -> dict:
    """
    Given a dict of session features, return:
      - abandonment_probability (float 0–1)
      - risk_tier ("low" | "medium" | "high")
      - top_shap_reasons (list of dicts)

    Usage:
        from model_a import predict_session
        result = predict_session({
            "n_views": 5,
            "n_addtocart": 1,
            "session_duration_min": 8.3,
            "unique_items_viewed": 4,
            "browse_to_cart_ratio": 5.0,
            "session_velocity": 0.6,
            "last_gap_min": 3.2,
            "hour_of_day": 14,
            "day_of_week": 2,
            "has_cart_add": 1,
        })
    """
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEAT_PATH, "rb") as f:
        feature_cols = pickle.load(f)

    X = pd.DataFrame([session_dict])[feature_cols].fillna(0)
    prob = float(model.predict_proba(X)[0, 1])

    if prob < 0.4:
        tier = "low"
    elif prob < 0.7:
        tier = "medium"
    else:
        tier = "high"

    # SHAP for this single prediction
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)[0]
    shap_pairs = sorted(
        zip(feature_cols, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    top_reasons = [
        {"feature": feat, "shap_value": round(float(val), 4)}
        for feat, val in shap_pairs[:3]
    ]

    return {
        "abandonment_probability": round(prob, 4),
        "risk_tier": tier,
        "top_shap_reasons": top_reasons,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Check data file ──
    if not os.path.exists(DATA_PATH):
        print(f"\n  ERROR: {DATA_PATH} not found.")
        print("  Download events.csv from:")
        print("  https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset")
        print(f"  Then place it in {DATA_PATH}\n")
        exit(1)

    # ── Run pipeline ──
    df           = load_data(DATA_PATH, sample=SAMPLE_SIZE)
    df           = build_sessions(df)
    features, feature_cols = engineer_features(df)

    model, X_train, X_test, y_train, y_test = train_model(features, feature_cols)
    auc, ap      = evaluate_model(model, X_test, y_test, feature_cols)
    explainer    = compute_shap(model, X_train, X_test, feature_cols)

    # ── Save model ──
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(FEAT_PATH, "wb") as f:
        pickle.dump(feature_cols, f)

    print(f"\n  Model saved  → {MODEL_PATH}")
    print(f"  Features saved → {FEAT_PATH}")

    # ── Quick inference test ──
    print("\n  ── Sample Prediction ──────────────")
    sample = {
        "n_views": 7,
        "n_addtocart": 1,
        "session_duration_min": 12.5,
        "unique_items_viewed": 5,
        "browse_to_cart_ratio": 7.0,
        "session_velocity": 0.56,
        "last_gap_min": 4.1,
        "hour_of_day": 15,
        "day_of_week": 3,
        "has_cart_add": 1,
    }
    result = predict_session(sample)
    print(f"  Abandonment probability : {result['abandonment_probability']}")
    print(f"  Risk tier               : {result['risk_tier']}")
    print(f"  Top SHAP reasons        :")
    for r in result["top_shap_reasons"]:
        direction = "↑ raises risk" if r["shap_value"] > 0 else "↓ lowers risk"
        print(f"    {r['feature']:<30} {r['shap_value']:+.4f}  {direction}")

    print(f"\n  Final AUC-ROC: {auc:.4f}  |  Avg Precision: {ap:.4f}")
    print("\n  Done. DataSentinel Model A ready.\n")