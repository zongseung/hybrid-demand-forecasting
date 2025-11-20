"""
Inference Demo Script
Demonstrates how to use the trained model for 24-hour forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.inference_pipeline import PowerDemandForecaster, create_exog_features
from src.utils.grafana_client import GrafanaClient


def load_historical_data(csv_path: str, window_size: int = 168) -> pd.DataFrame:
    """
    Load historical data for inference
    
    Args:
        csv_path: Path to CSV file
        window_size: Minimum required historical window
        
    Returns:
        Historical data with all components (trend, seasonality, residual)
    """
    print(f"\n📂 Loading historical data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df["일시"] = pd.to_datetime(df["일시"])
    df.set_index("일시", inplace=True)
    
    # For inference, we need the decomposed components
    # These should be computed using the trained models
    # For now, we'll load pre-computed values
    
    print(f"✓ Loaded {len(df)} historical records")
    print(f"  - Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def make_forecast(forecaster: PowerDemandForecaster,
                 historical_data: pd.DataFrame,
                 forecast_date: str = None) -> pd.DataFrame:
    """
    Make 24-hour forecast
    
    Args:
        forecaster: Trained PowerDemandForecaster
        historical_data: Historical data with components
        forecast_date: Target forecast date (default: next day)
        
    Returns:
        Forecast DataFrame
    """
    if forecast_date is None:
        forecast_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"\n🔮 Making 24-hour forecast for: {forecast_date}")
    
    # Get last window_size hours of historical data
    historical_window = historical_data.iloc[-forecaster.window_size:]
    
    # Create future exogenous features
    forecast_start = pd.Timestamp(forecast_date)
    future_timestamps = pd.date_range(
        start=forecast_start,
        periods=forecaster.horizon,
        freq='H'
    )
    exog_future = create_exog_features(future_timestamps)
    
    # Make forecast
    forecast_df = forecaster.forecast_with_timestamps(
        historical_data=historical_window,
        exog_features_future=exog_future,
        device='cuda'
    )
    
    return forecast_df


def plot_forecast(historical_data: pd.DataFrame,
                 forecast_df: pd.DataFrame,
                 actual_data: pd.DataFrame = None,
                 save_path: str = None):
    """
    Plot forecast with components
    
    Args:
        historical_data: Historical data (last 7 days)
        forecast_df: Forecast DataFrame
        actual_data: Actual future data (for comparison)
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # Historical range (last 7 days)
    hist_hours = 24 * 7
    hist_data = historical_data.iloc[-hist_hours:]
    
    # 1. Final forecast
    ax = axes[0]
    ax.plot(hist_data.index, hist_data["power demand(MW)"], 
            label="Historical", linewidth=2, color='blue')
    ax.plot(forecast_df.index, forecast_df['forecast'], 
            label="Forecast", linewidth=2, color='red', linestyle='--')
    
    if actual_data is not None:
        ax.plot(actual_data.index, actual_data["power demand(MW)"], 
                label="Actual", linewidth=2, color='green', alpha=0.7)
    
    ax.set_title("24-Hour Power Demand Forecast", fontsize=14, fontweight='bold')
    ax.set_ylabel("Power Demand (MW)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Trend component
    ax = axes[1]
    if 'trend' in hist_data.columns:
        ax.plot(hist_data.index, hist_data['trend'], 
                label="Historical Trend", color='blue', alpha=0.5)
    ax.plot(forecast_df.index, forecast_df['trend'], 
            label="Forecasted Trend", linewidth=2, color='orange', linestyle='--')
    ax.set_title("Trend Component", fontsize=12)
    ax.set_ylabel("Trend (MW)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Seasonality component
    ax = axes[2]
    if 'seasonality' in hist_data.columns:
        ax.plot(hist_data.index, hist_data['seasonality'], 
                label="Historical Seasonality", color='blue', alpha=0.5)
    ax.plot(forecast_df.index, forecast_df['seasonality'], 
            label="Forecasted Seasonality", linewidth=2, color='purple', linestyle='--')
    ax.set_title("Fourier Seasonality Component", fontsize=12)
    ax.set_ylabel("Seasonality (MW)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Residual component
    ax = axes[3]
    if 'residual' in hist_data.columns:
        ax.plot(hist_data.index, hist_data['residual'], 
                label="Historical Residual", color='blue', alpha=0.5)
    ax.plot(forecast_df.index, forecast_df['residual'], 
            label="Forecasted Residual (LSTM)", linewidth=2, color='green', linestyle='--')
    ax.set_title("LSTM Residual Component", fontsize=12)
    ax.set_ylabel("Residual (MW)")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    
    plt.show()


def evaluate_forecast(forecast_df: pd.DataFrame,
                     actual_data: pd.DataFrame) -> dict:
    """
    Evaluate forecast against actual data
    
    Args:
        forecast_df: Forecast DataFrame
        actual_data: Actual data DataFrame
        
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
    
    # Align timestamps
    common_idx = forecast_df.index.intersection(actual_data.index)
    
    if len(common_idx) == 0:
        print("⚠ No overlapping timestamps for evaluation")
        return {}
    
    y_true = actual_data.loc[common_idx, "power demand(MW)"].values
    y_pred = forecast_df.loc[common_idx, 'forecast'].values
    
    def smape(y_true, y_pred):
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        diff = np.abs(y_true - y_pred)
        mask = denominator == 0
        denominator[mask] = 1
        diff[mask] = 0
        return np.mean(diff / denominator) * 100
    
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'SMAPE': smape(y_true, y_pred)
    }
    
    print("\n📊 Evaluation Metrics:")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name:10s}: {metric_value:10.4f}")
    print("="*50)
    
    return metrics


def main():
    """Main inference demo"""
    
    print("\n" + "="*80)
    print("POWER DEMAND FORECASTING - INFERENCE DEMO")
    print("="*80)
    
    # Configuration
    model_dir = "models/production"
    csv_path = "/mnt/nvme/tilting/power_demand_final.csv"
    window_size = 168  # 7 days
    horizon = 24       # 24 hours
    
    # 1. Initialize forecaster
    print("\n1️⃣ Initializing forecaster...")
    forecaster = PowerDemandForecaster(
        model_dir=model_dir,
        window_size=window_size,
        horizon=horizon
    )
    
    # 2. Load trained models
    print("\n2️⃣ Loading trained models...")
    try:
        forecaster.load_models()
    except FileNotFoundError:
        print("❌ Models not found. Please train models first using weekly_retrain_hybrid.py")
        print("   Run: python flows/weekly_retrain_hybrid.py")
        return
    
    # 3. Load historical data
    print("\n3️⃣ Loading historical data...")
    historical_data = load_historical_data(csv_path, window_size)
    
    # 4. Make forecast
    print("\n4️⃣ Making 24-hour forecast...")
    forecast_df = make_forecast(forecaster, historical_data)
    
    print("\n✅ Forecast complete!")
    print(forecast_df)
    
    # 5. Plot forecast
    print("\n5️⃣ Plotting forecast...")
    plot_forecast(
        historical_data=historical_data,
        forecast_df=forecast_df,
        save_path="forecast_demo.png"
    )
    
    # 6. (Optional) Evaluate if actual data is available
    # Get actual data for the forecast period
    forecast_start = forecast_df.index[0]
    forecast_end = forecast_df.index[-1]
    
    actual_data = historical_data[
        (historical_data.index >= forecast_start) & 
        (historical_data.index <= forecast_end)
    ]
    
    if len(actual_data) > 0:
        print("\n6️⃣ Evaluating forecast...")
        metrics = evaluate_forecast(forecast_df, actual_data)
        
        # 7. Send to Grafana
        print("\n7️⃣ Sending results to Grafana...")
        try:
            grafana = GrafanaClient()
            grafana.send_forecast(forecast_df)
            grafana.send_metrics(metrics, horizon=horizon)
            grafana.close()
            print("✅ Results sent to Grafana")
        except Exception as e:
            print(f"⚠ Failed to send to Grafana: {e}")
    
    # Save forecast to CSV
    forecast_csv_path = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    forecast_df.to_csv(forecast_csv_path)
    print(f"\n✅ Forecast saved to: {forecast_csv_path}")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

