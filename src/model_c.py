import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import pickle

# ==============================
# Paths
# ==============================
DATA_PATH = "./data/ar_aging.xlsx"
OUTPUT_DIR = "./outputs"
MODEL_DIR = "./models"

FORECAST_CSV_PATH = os.path.join(OUTPUT_DIR, "model_c_forecast.csv")
FORECAST_PLOT_PATH = os.path.join(OUTPUT_DIR, "model_c_forecast.png")
MODEL_PATH = os.path.join(MODEL_DIR, "model_c.pkl")


# ==============================
# Ensure directories exist
# ==============================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================
# Load and clean data
# ==============================
def load_and_clean_data(file_path):
    """
    Load Excel file and clean dataset
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found at {file_path}")
        print("👉 Please place your file as: ./data/ar_aging.xlsx")
        return None

    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return None

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Due Date"] = pd.to_datetime(df["Due Date"], errors="coerce")

    # Filter only Invoice & Payment
    df = df[df["Transaction Type"].isin(["Invoice", "Payment"])]

    # Remove null or zero balances
    df = df[df["Open Balance"].notna()]
    df = df[df["Open Balance"] != 0]

    # Drop rows with invalid dates
    df = df.dropna(subset=["Date"])

    return df


# ==============================
# Aggregate to daily totals
# ==============================
def aggregate_daily(df):
    """
    Aggregate Open Balance by Date
    """
    daily = (
        df.groupby("Date")["Open Balance"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )

    # Rename for Prophet
    daily = daily.rename(columns={"Date": "ds", "Open Balance": "y"})

    return daily


# ==============================
# Train Prophet model
# ==============================
def train_model(daily_df):
    """
    Train Prophet model with seasonality
    """
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    model.fit(daily_df)
    return model


# ==============================
# Forecast future
# ==============================
def generate_forecast(model, periods=90):
    """
    Generate forecast for future periods
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast


# ==============================
# Plot forecast
# ==============================
def plot_forecast(daily_df, forecast):
    """
    Plot actual vs forecast
    """
    plt.figure(figsize=(12, 6))

    # Actual
    plt.plot(daily_df["ds"], daily_df["y"], label="Actual")

    # Forecast
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast")

    plt.title("Cash Flow Forecast (Open Balance)")
    plt.xlabel("Date")
    plt.ylabel("Open Balance")
    plt.legend()

    plt.savefig(FORECAST_PLOT_PATH)
    plt.close()


# ==============================
# Extract summary metrics
# ==============================
def get_forecast_summary(forecast):
    """
    Extract 30/60/90 day forecasts
    """
    last_date = forecast["ds"].max()

    def get_value(days):
        target_date = last_date - pd.Timedelta(days=(90 - days))
        row = forecast[forecast["ds"] >= target_date].iloc[0]
        return float(row["yhat"])

    f30 = get_value(30)
    f60 = get_value(60)
    f90 = get_value(90)

    # Determine trend
    if f90 > f30:
        trend = "growing"
    elif f90 < f30:
        trend = "declining"
    else:
        trend = "stable"

    return {
        "30_day_forecast": round(f30, 2),
        "60_day_forecast": round(f60, 2),
        "90_day_forecast": round(f90, 2),
        "trend": trend
    }


# ==============================
# Main pipeline
# ==============================
def main():
    print("🚀 Running Model C: Cash Flow Forecaster...")

    df = load_and_clean_data(DATA_PATH)
    if df is None:
        return

    daily_df = aggregate_daily(df)

    model = train_model(daily_df)

    forecast = generate_forecast(model, periods=90)

    # Save forecast CSV
    forecast.to_csv(FORECAST_CSV_PATH, index=False)

    # Plot forecast
    plot_forecast(daily_df, forecast)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Summary
    summary = get_forecast_summary(forecast)

    print("\n📊 Forecast Summary:")
    print(f"30-day: {summary['30_day_forecast']}")
    print(f"60-day: {summary['60_day_forecast']}")
    print(f"90-day: {summary['90_day_forecast']}")
    print(f"Trend: {summary['trend']}")

    print("\n✅ Outputs saved:")
    print(f"- CSV: {FORECAST_CSV_PATH}")
    print(f"- Plot: {FORECAST_PLOT_PATH}")
    print(f"- Model: {MODEL_PATH}")


# ==============================
# API-ready function
# ==============================
def predict_cashflow():
    """
    Load saved model and return forecast summary
    (Used by FastAPI backend)
    """
    if not os.path.exists(MODEL_PATH):
        return {
            "error": "Model not found. Please run model_c.py first."
        }

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Generate fresh forecast
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    summary = get_forecast_summary(forecast)

    return summary


# ==============================
# Entry point
# ==============================
if __name__ == "__main__":
    main()