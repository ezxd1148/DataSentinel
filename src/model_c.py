import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import pickle
import warnings
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# ==============================
# Paths (with absolute path handling)
# ==============================
# Get the src directory (where this script is located)
SRC_DIR = Path(__file__).parent
DATA_PATH = SRC_DIR / "data" / "ar_aging.xlsx"
OUTPUT_DIR = SRC_DIR / "outputs"
MODEL_DIR = SRC_DIR / "models"

FORECAST_CSV_PATH = OUTPUT_DIR / "model_c_forecast.csv"
FORECAST_PLOT_PATH = OUTPUT_DIR / "model_c_forecast.png"
MODEL_PATH = MODEL_DIR / "model_c.pkl"


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
    Load Excel file and clean dataset.
    Handles Excel files with metadata headers (skips first N rows).
    
    Args:
        file_path (str or Path): Path to the Excel file
        
    Returns:
        pd.DataFrame: Cleaned dataframe or None if error
    """
    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
    
    if not file_path.exists():
        logger.error(f"File not found at {file_path}")
        logger.info("Please place your file as: ./data/ar_aging.xlsx")
        return None

    try:
        # Try reading with standard header
        df = pd.read_excel(file_path, engine="openpyxl")
        
        # If columns look like "Unnamed: X", skip metadata rows and find real header
        if any("Unnamed" in str(col) for col in df.columns):
            logger.info("Detected metadata in Excel file, searching for actual headers...")
            # Find row with actual headers (contains "Customer", "Open Balance", etc.)
            for skip_rows in range(min(10, len(df))):
                df = pd.read_excel(file_path, engine="openpyxl", skiprows=skip_rows)
                if not any("Unnamed" in str(col) for col in df.columns) and len(df.columns) > 1:
                    logger.info(f"Found actual headers at row {skip_rows}")
                    break
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        return None

    # Normalize column names: strip whitespace, lowercase for comparison
    df.columns = df.columns.str.strip()
    col_mapping = {col: col.lower().strip() for col in df.columns}
    
    # Map to standard column names (case-insensitive, flexible matching)
    standard_mapping = {}
    col_lower_map = {col.lower().strip(): col for col in df.columns}
    
    # Try to find Date column
    date_col = None
    for key in ["date", "transaction date", "posting date"]:
        if key in col_lower_map:
            date_col = col_lower_map[key]
            break
    
    # Try to find Transaction Type column
    trans_col = None
    for key in ["transaction type", "trans type", "type"]:
        if key in col_lower_map:
            trans_col = col_lower_map[key]
            break
    
    # Try to find Balance column
    balance_col = None
    for key in ["open balance", "balance", "amount", "outstanding"]:
        if key in col_lower_map:
            balance_col = col_lower_map[key]
            break
    
    # Check if required columns were found
    required_cols = {
        "Date": date_col,
        "Transaction Type": trans_col,
        "Open Balance": balance_col
    }
    
    missing_cols = [name for name, col in required_cols.items() if col is None]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        return None
    
    # Rename columns to standard names
    df = df.rename(columns={
        date_col: "Date",
        trans_col: "Transaction Type",
        balance_col: "Open Balance"
    })

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # Convert Open Balance to numeric, handling strings with currency symbols
    df["Open Balance"] = pd.to_numeric(df["Open Balance"], errors="coerce")

    # Clean Transaction Type: remove extra whitespace and normalize case
    df["Transaction Type"] = df["Transaction Type"].str.strip().str.title()

    # Drop rows with invalid dates FIRST
    initial_len = len(df)
    df = df.dropna(subset=["Date"])
    logger.info(f"Removed {initial_len - len(df)} invalid date records")

    # Remove null or zero balances
    initial_len = len(df)
    df = df[df["Open Balance"].notna()]
    df = df[df["Open Balance"] != 0]
    logger.info(f"Removed {initial_len - len(df)} null/zero balance records")

    # Filter only Invoice & Payment (case-insensitive)
    valid_types = df["Transaction Type"].unique()
    invoice_types = [t for t in valid_types if "invoice" in t.lower()]
    payment_types = [t for t in valid_types if "payment" in t.lower()]
    all_valid = invoice_types + payment_types
    
    if all_valid:
        df = df[df["Transaction Type"].isin(all_valid)]
        logger.info(f"Filtered to {len(df)} transaction records (types: {all_valid})")
    else:
        logger.warning(f"No Invoice/Payment types found. Available types: {valid_types.tolist()}")
        logger.info(f"Using all {len(df)} transaction records")

    if len(df) == 0:
        logger.warning("No data remaining after cleaning")
        return None

    return df


# ==============================
# Aggregate to daily totals
# ==============================
def aggregate_daily(df):
    """
    Aggregate Open Balance by Date.
    
    Args:
        df (pd.DataFrame): Input dataframe with Date and Open Balance
        
    Returns:
        pd.DataFrame: Daily aggregated data with 'ds' and 'y' columns for Prophet
    """
    daily = (
        df.groupby("Date")["Open Balance"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )

    # Rename for Prophet
    daily = daily.rename(columns={"Date": "ds", "Open Balance": "y"})
    
    # Handle missing dates and fill gaps
    date_range = pd.date_range(start=daily["ds"].min(), end=daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(date_range, fill_value=0).reset_index()
    daily.columns = ["ds", "y"]
    
    logger.info(f"Aggregated to {len(daily)} daily records (date range: {daily['ds'].min()} to {daily['ds'].max()})")

    return daily


# ==============================
# Train Prophet model
# ==============================
def train_model(daily_df):
    """
    Train Prophet model with seasonality and error handling.
    
    Args:
        daily_df (pd.DataFrame): Daily aggregated data
        
    Returns:
        Prophet: Trained Prophet model
    """
    logger.info("Training Prophet model...")
    
    # Disable verbose output from Prophet
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95  # 95% confidence interval
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(daily_df)
    
    logger.info("Model training completed")
    return model


# ==============================
# Forecast future
# ==============================
def generate_forecast(model, periods=90):
    """
    Generate forecast for future periods.
    
    Args:
        model (Prophet): Trained Prophet model
        periods (int): Number of days to forecast (default: 90)
        
    Returns:
        pd.DataFrame: Forecast dataframe with predictions and intervals
    """
    logger.info(f"Generating {periods}-day forecast...")
    
    future = model.make_future_dataframe(periods=periods)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = model.predict(future)
    
    logger.info(f"Forecast generated with {len(forecast)} records")
    return forecast


# ==============================
# Plot forecast
# ==============================
def plot_forecast(daily_df, forecast, output_path=None):
    """
    Plot actual vs forecast with confidence intervals.
    
    Args:
        daily_df (pd.DataFrame): Actual data
        forecast (pd.DataFrame): Forecast data
        output_path (str): Path to save plot (uses default if None)
    """
    if output_path is None:
        output_path = FORECAST_PLOT_PATH
    
    plt.figure(figsize=(14, 7))

    # Actual data
    plt.plot(daily_df["ds"], daily_df["y"], label="Actual", linewidth=2, color="blue")

    # Forecast with confidence interval
    forecast_future = forecast[forecast["ds"] > daily_df["ds"].max()]
    plt.plot(forecast_future["ds"], forecast_future["yhat"], label="Forecast", 
             linewidth=2, color="orange", linestyle="--")
    plt.fill_between(
        forecast_future["ds"],
        forecast_future["yhat_lower"],
        forecast_future["yhat_upper"],
        alpha=0.3,
        color="orange",
        label="95% Confidence Interval"
    )

    plt.title("Cash Flow Forecast (Open Balance)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Open Balance ($)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved to {output_path}")


# ==============================
# Extract summary metrics
# ==============================
def get_forecast_summary(forecast, daily_df=None):
    """
    Extract 30/60/90 day forecasts with proper error handling.
    
    Args:
        forecast (pd.DataFrame): Forecast dataframe
        daily_df (pd.DataFrame): Optional actual data for baseline comparison
        
    Returns:
        dict: Summary metrics including forecasts, trend, and statistics
    """
    # Get the last actual date if provided, otherwise use minimum forecast date
    if daily_df is not None and len(daily_df) > 0:
        last_actual_date = daily_df["ds"].max()
        forecast_only = forecast[forecast["ds"] > last_actual_date].reset_index(drop=True)
    else:
        forecast_only = forecast.reset_index(drop=True)

    # Ensure we have enough forecast data
    if len(forecast_only) < 30:
        logger.warning("Insufficient forecast data for 90-day summary")
        return {
            "error": "Insufficient forecast data",
            "available_days": len(forecast_only)
        }

    def get_value(days):
        """Get forecast value at specific day offset"""
        idx = min(days - 1, len(forecast_only) - 1)
        return float(forecast_only.iloc[idx]["yhat"])

    f30 = get_value(30) if len(forecast_only) >= 30 else forecast_only["yhat"].iloc[-1]
    f60 = get_value(60) if len(forecast_only) >= 60 else forecast_only["yhat"].iloc[-1]
    f90 = get_value(90) if len(forecast_only) >= 90 else forecast_only["yhat"].iloc[-1]

    # Determine trend using percentage change
    if f90 > f30 * 1.05:  # 5% threshold to avoid classification as "stable"
        trend = "growing"
        trend_pct = ((f90 - f30) / f30 * 100) if f30 != 0 else 0
    elif f90 < f30 * 0.95:
        trend = "declining"
        trend_pct = ((f90 - f30) / f30 * 100) if f30 != 0 else 0
    else:
        trend = "stable"
        trend_pct = 0

    # Calculate additional statistics
    avg_forecast = forecast_only["yhat"].mean()
    max_forecast = forecast_only["yhat"].max()
    min_forecast = forecast_only["yhat"].min()

    return {
        "30_day_forecast": round(f30, 2),
        "60_day_forecast": round(f60, 2),
        "90_day_forecast": round(f90, 2),
        "trend": trend,
        "trend_percentage": round(trend_pct, 2),
        "forecast_average": round(avg_forecast, 2),
        "forecast_max": round(max_forecast, 2),
        "forecast_min": round(min_forecast, 2),
        "forecast_volatility": round(forecast_only["yhat"].std(), 2)
    }


# ==============================
# Main pipeline
# ==============================
def main():
    """
    Main pipeline for cash flow forecasting.
    Executes all steps: load, clean, aggregate, train, forecast, and save.
    """
    logger.info("=" * 60)
    logger.info("🚀 Running Model C: Cash Flow Forecaster...")
    logger.info("=" * 60)

    # Step 1: Load and clean data
    df = load_and_clean_data(DATA_PATH)
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return False

    # Step 2: Aggregate to daily totals
    daily_df = aggregate_daily(df)
    if daily_df is None or len(daily_df) < 20:
        logger.error("Insufficient data for modeling (need at least 20 days)")
        return False

    # Step 3: Train model
    try:
        model = train_model(daily_df)
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        return False

    # Step 4: Generate forecast
    try:
        forecast = generate_forecast(model, periods=90)
    except Exception as e:
        logger.error(f"Failed to generate forecast: {e}")
        return False

    # Step 5: Save outputs
    try:
        # Save forecast CSV
        forecast.to_csv(str(FORECAST_CSV_PATH), index=False)
        logger.info(f"Forecast CSV saved to {FORECAST_CSV_PATH}")

        # Plot forecast
        plot_forecast(daily_df, forecast)

        # Save model
        with open(str(MODEL_PATH), "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to save outputs: {e}")
        return False

    # Step 6: Display summary
    summary = get_forecast_summary(forecast, daily_df)
    
    if "error" in summary:
        logger.error(f"Summary error: {summary['error']}")
        return False

    logger.info("\n" + "=" * 60)
    logger.info("📊 FORECAST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"30-day forecast:  ${summary['30_day_forecast']:,.2f}")
    logger.info(f"60-day forecast:  ${summary['60_day_forecast']:,.2f}")
    logger.info(f"90-day forecast:  ${summary['90_day_forecast']:,.2f}")
    logger.info(f"Trend:            {summary['trend'].upper()} ({summary['trend_percentage']:+.2f}%)")
    logger.info(f"Avg forecast:     ${summary['forecast_average']:,.2f}")
    logger.info(f"Volatility (σ):   ${summary['forecast_volatility']:,.2f}")
    logger.info("=" * 60)

    logger.info("\n✅ All outputs saved:")
    logger.info(f"   📄 CSV:   {FORECAST_CSV_PATH.absolute()}")
    logger.info(f"   📈 Plot:  {FORECAST_PLOT_PATH.absolute()}")
    logger.info(f"   💾 Model: {MODEL_PATH.absolute()}")
    logger.info("=" * 60 + "\n")

    return True


# ==============================
# API-ready function
# ==============================
def predict_cashflow(days=90):
    """
    Load saved model and return forecast summary.
    (Used by FastAPI backend)
    
    Args:
        days (int): Number of days to forecast (default: 90)
        
    Returns:
        dict: Forecast summary with predictions and statistics
    """
    if not MODEL_PATH.exists():
        logger.error("Model not found")
        return {
            "error": "Model not found. Please run model_c.py first.",
            "status": "failure"
        }

    try:
        with open(str(MODEL_PATH), "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {
            "error": f"Failed to load model: {e}",
            "status": "failure"
        }

    # Generate fresh forecast
    try:
        future = model.make_future_dataframe(periods=days)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = model.predict(future)
        
        summary = get_forecast_summary(forecast)
        summary["status"] = "success"
        summary["generated_at"] = datetime.now().isoformat()
        
        return summary
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}")
        return {
            "error": f"Forecast generation failed: {e}",
            "status": "failure"
        }


# ==============================
# Entry point
# ==============================
if __name__ == "__main__":
    main()