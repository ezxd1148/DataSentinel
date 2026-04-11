"""
DataSentinel — Model B: Product Recommender
============================================
Pipeline:
  1. Load RetailRocket events CSV (same file as Model A)
  2. Build user-item interaction matrix from purchase + addtocart events
  3. Train SVD collaborative filtering model (scikit-surprise)
  4. Evaluate with RMSE and cross-validation
  5. Risk-adjusted recommendation logic (plugs into Model A output)
  6. Save model for FastAPI inference

Usage:
  python model_b.py

Inference (called by FastAPI):
  from model_b import recommend_products
  result = recommend_products(user_id=12345, abandonment_risk="high", top_n=5)
"""

import os
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG (paths relative to this package so API cwd does not matter)
# ─────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
DATA_PATH = SRC_DIR / "data" / "events.csv"
MODEL_PATH = SRC_DIR / "models" / "model_b.pkl"
ITEM_META_PATH = SRC_DIR / "models" / "model_b_item_meta.pkl"
OUTPUT_DIR = SRC_DIR / "outputs"
SAMPLE_SIZE    = 500_000   # match Model A sample size
RANDOM_SEED    = 42
TOP_N          = 10        # default recommendations per user

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_PATH.parent, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1 — LOAD & FILTER DATA
# ─────────────────────────────────────────────
def load_data(path: str, sample: int | None = None) -> pd.DataFrame:
    print("[ 1/5 ] Loading events data...")
    df = pd.read_csv(
        path,
        nrows=sample,
        dtype={
            "visitorid": "int32",
            "itemid":    "int32",
            "event":     "category",
        },
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    print(f"         Loaded {len(df):,} events | {df['visitorid'].nunique():,} visitors | {df['itemid'].nunique():,} items")
    return df


# ─────────────────────────────────────────────
# STEP 2 — BUILD INTERACTION MATRIX
# ─────────────────────────────────────────────
def build_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw events into implicit ratings per user-item pair.

    Rating scale (implicit feedback):
      view        → 1  (mild interest)
      addtocart   → 3  (strong interest)
      transaction → 5  (purchased)

    Multiple interactions are summed and capped at 5.
    """
    print("[ 2/5 ] Building user-item interaction matrix...")

    event_weights = {"view": 1, "addtocart": 3, "transaction": 5}
    df = df.copy()
    df["rating"] = df["event"].astype(str).map(event_weights).fillna(0)

    # Aggregate: sum weights per user-item, cap at 5
    interactions = (
        df.groupby(["visitorid", "itemid"])["rating"]
        .sum()
        .clip(upper=5)
        .reset_index()
    )
    interactions.columns = ["user_id", "item_id", "rating"]

    # Keep only users with at least 2 interactions (cold start filter)
    user_counts = interactions.groupby("user_id")["item_id"].count()
    active_users = user_counts[user_counts >= 2].index
    interactions = interactions[interactions["user_id"].isin(active_users)]

    # Keep only items with at least 2 interactions
    item_counts = interactions.groupby("item_id")["user_id"].count()
    active_items = item_counts[item_counts >= 2].index
    interactions = interactions[interactions["item_id"].isin(active_items)]

    print(f"         Interactions: {len(interactions):,} | "
          f"Users: {interactions['user_id'].nunique():,} | "
          f"Items: {interactions['item_id'].nunique():,}")

    return interactions


# ─────────────────────────────────────────────
# STEP 3 — BUILD ITEM METADATA
# ─────────────────────────────────────────────
def build_item_metadata(df: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple item profile used for risk-adjusted filtering.
    Since RetailRocket has no price/rating columns, we derive proxies:
      - popularity_score : how many users interacted with this item
      - purchase_rate    : fraction of interactions that were purchases
      - avg_rating       : mean implicit rating across all users
    """
    item_ids = interactions["item_id"].unique()
    item_df  = df[df["itemid"].isin(item_ids)].copy()

    # Popularity: unique visitors who interacted
    popularity = (
        item_df.groupby("itemid")["visitorid"]
        .nunique()
        .rename("popularity_score")
    )

    # Purchase rate: transactions / total events per item
    total_events = item_df.groupby("itemid").size().rename("total_events")
    purchases    = item_df[item_df["event"] == "transaction"].groupby("itemid").size().rename("purchases")

    meta = pd.concat([popularity, total_events, purchases], axis=1).fillna(0)
    meta["purchase_rate"] = meta["purchases"] / meta["total_events"]
    meta["avg_rating"]    = interactions.groupby("item_id")["rating"].mean()

    # Low-friction score: higher = easier to convert
    # Combines high purchase rate and high popularity
    meta["low_friction_score"] = (
        0.6 * meta["purchase_rate"] + 0.4 * (meta["popularity_score"] / meta["popularity_score"].max())
    )

    meta = meta.reset_index().rename(columns={"itemid": "item_id"})
    return meta


# ─────────────────────────────────────────────
# STEP 4 — TRAIN SVD MODEL
# ─────────────────────────────────────────────
def train_model(interactions: pd.DataFrame) -> tuple:
    print("[ 3/5 ] Training SVD collaborative filtering model...")

    reader  = Reader(rating_scale=(1, 5))
    data    = Dataset.load_from_df(interactions[["user_id", "item_id", "rating"]], reader)

    # Cross-validate to get honest RMSE estimate
    svd = SVD(
        n_factors=50,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=RANDOM_SEED,
    )

    print("         Running 3-fold cross-validation...")
    cv_results = cross_validate(svd, data, measures=["RMSE", "MAE"], cv=3, verbose=False)

    rmse_mean = cv_results["test_rmse"].mean()
    mae_mean  = cv_results["test_mae"].mean()
    print(f"         CV RMSE: {rmse_mean:.4f} | CV MAE: {mae_mean:.4f}")

    # Train final model on full dataset
    trainset = data.build_full_trainset()
    svd.fit(trainset)

    # Save CV results plot
    fig, ax = plt.subplots(figsize=(7, 4))
    folds = range(1, len(cv_results["test_rmse"]) + 1)
    ax.plot(folds, cv_results["test_rmse"], marker="o", label="RMSE", color="steelblue")
    ax.plot(folds, cv_results["test_mae"],  marker="s", label="MAE",  color="coral")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Error")
    ax.set_title("Model B — SVD Cross-Validation (3-Fold)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_b_cv_scores.png", dpi=150)
    plt.close()
    print(f"         Saved: {OUTPUT_DIR}/model_b_cv_scores.png")

    return svd, trainset, rmse_mean, mae_mean


# ─────────────────────────────────────────────
# STEP 5 — EVALUATE ON HELD-OUT TEST SET
# ─────────────────────────────────────────────
def evaluate_model(interactions: pd.DataFrame, rmse_mean: float, mae_mean: float):
    print("[ 4/5 ] Evaluating on held-out test split...")

    reader   = Reader(rating_scale=(1, 5))
    data     = Dataset.load_from_df(interactions[["user_id", "item_id", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)

    svd_eval = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=RANDOM_SEED)
    svd_eval.fit(trainset)
    predictions = svd_eval.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae  = accuracy.mae(predictions,  verbose=False)

    print(f"\n  ── Metrics ────────────────────────")
    print(f"  Test RMSE      : {rmse:.4f}")
    print(f"  Test MAE       : {mae:.4f}")
    print(f"  CV RMSE (mean) : {rmse_mean:.4f}")
    print(f"  CV MAE  (mean) : {mae_mean:.4f}")

    # Distribution of predicted vs actual ratings
    actuals    = [p.r_ui for p in predictions]
    predicted  = [p.est  for p in predictions]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(actuals,   bins=20, alpha=0.6, label="Actual",    color="steelblue")
    ax.hist(predicted, bins=20, alpha=0.6, label="Predicted", color="coral")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Model B — Actual vs Predicted Rating Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_b_rating_dist.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/model_b_rating_dist.png")

    return rmse, mae


# ─────────────────────────────────────────────
# RISK-ADJUSTED RECOMMENDATION LOGIC
# ─────────────────────────────────────────────
def get_top_n_recommendations(
    model,
    trainset,
    item_meta: pd.DataFrame,
    user_id: int,
    n: int = TOP_N,
    abandonment_risk: str = "low",
) -> list[dict]:
    """
    Get top-N product recommendations for a user, adjusted by abandonment risk.

    Risk adjustment logic:
      low    → standard SVD score ranking (best predicted match)
      medium → blend SVD score 70% + low_friction_score 30%
      high   → blend SVD score 40% + low_friction_score 60%
                (push lower-friction, higher-converting products)

    Returns list of dicts: [{ item_id, predicted_rating, low_friction_score, final_score }]
    """
    risk_weights = {
        "low":    (1.0, 0.0),
        "medium": (0.7, 0.3),
        "high":   (0.4, 0.6),
    }
    svd_w, friction_w = risk_weights.get(abandonment_risk, (1.0, 0.0))

    # Items the user has already interacted with
    try:
        inner_uid     = trainset.to_inner_uid(user_id)
        seen_items    = set(j for (j, _) in trainset.ur[inner_uid])
        seen_item_ids = set(trainset.to_raw_iid(j) for j in seen_items)
    except ValueError:
        # Unknown user — return popular items as fallback
        seen_item_ids = set()

    # All items in training set
    all_item_ids = set(trainset.to_raw_iid(i) for i in trainset.all_items())
    unseen_items = all_item_ids - seen_item_ids

    # Predict ratings for unseen items
    predictions = [
        (iid, model.predict(user_id, iid).est)
        for iid in unseen_items
    ]

    # Normalise SVD scores to 0-1
    scores    = np.array([p[1] for p in predictions])
    min_s, max_s = scores.min(), scores.max()
    norm_scores  = (scores - min_s) / (max_s - min_s + 1e-9)

    # Merge with item metadata for low-friction score
    meta_lookup = item_meta.set_index("item_id")["low_friction_score"].to_dict()

    results = []
    for (iid, raw_score), norm_score in zip(predictions, norm_scores):
        friction = meta_lookup.get(iid, 0.5)
        final    = svd_w * norm_score + friction_w * friction
        results.append({
            "item_id":          int(iid),
            "predicted_rating": round(float(raw_score), 4),
            "low_friction_score": round(float(friction), 4),
            "final_score":      round(float(final), 4),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:n]


# ─────────────────────────────────────────────
# INFERENCE HELPER  (used by FastAPI)
# ─────────────────────────────────────────────
def recommend_products(
    user_id: int,
    abandonment_risk: str = "low",
    top_n: int = 5,
) -> dict:
    """
    Load saved model and return risk-adjusted recommendations.

    Usage:
        from model_b import recommend_products
        result = recommend_products(
            user_id=12345,
            abandonment_risk="high",
            top_n=5
        )

    Returns:
        {
          "user_id": 12345,
          "abandonment_risk": "high",
          "recommendations": [
            { "item_id": 456, "predicted_rating": 4.2,
              "low_friction_score": 0.78, "final_score": 0.85 },
            ...
          ],
          "strategy": "low-friction bias active"
        }
    """
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    model    = payload["model"]
    trainset = payload["trainset"]
    item_meta = payload["item_meta"]

    strategy_labels = {
        "low":    "standard recommendation",
        "medium": "slight low-friction bias",
        "high":   "low-friction bias active",
    }

    recs = get_top_n_recommendations(
        model, trainset, item_meta,
        user_id=user_id,
        n=top_n,
        abandonment_risk=abandonment_risk,
    )

    return {
        "user_id":          user_id,
        "abandonment_risk": abandonment_risk,
        "recommendations":  recs,
        "strategy":         strategy_labels.get(abandonment_risk, "standard recommendation"),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"\n  ERROR: {DATA_PATH} not found.")
        print("  Download events.csv from:")
        print("  https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset")
        print(f"  Then place it in {DATA_PATH}\n")
        exit(1)

    # ── Pipeline ──
    df           = load_data(DATA_PATH, sample=SAMPLE_SIZE)
    interactions = build_interactions(df)
    item_meta    = build_item_metadata(df, interactions)

    model, trainset, rmse_cv, mae_cv = train_model(interactions)
    rmse, mae = evaluate_model(interactions, rmse_cv, mae_cv)

    # ── Save ──
    print("\n[ 5/5 ] Saving model...")
    payload = {
        "model":    model,
        "trainset": trainset,
        "item_meta": item_meta,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"  Model saved → {MODEL_PATH}")

    # ── Quick inference test ──
    sample_user = interactions["user_id"].iloc[0]

    print(f"\n  ── Sample Predictions (user {sample_user}) ────────────")

    for risk in ["low", "medium", "high"]:
        result = recommend_products(sample_user, abandonment_risk=risk, top_n=3)
        print(f"\n  Risk tier : {risk.upper()} | Strategy: {result['strategy']}")
        for r in result["recommendations"]:
            print(f"    Item {r['item_id']:<8} "
                  f"SVD rating: {r['predicted_rating']:.2f}  "
                  f"Friction: {r['low_friction_score']:.2f}  "
                  f"Final: {r['final_score']:.2f}")

    print(f"\n  Final Test RMSE: {rmse:.4f}  |  MAE: {mae:.4f}")
    print("\n  Done. DataSentinel Model B ready.\n")