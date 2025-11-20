"""
Daily Forecast Flow for Hybrid Power Demand Forecasting
Generates 24-hour ahead forecasts using trained models
"""

from prefect import flow, task
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.inference_pipeline import PowerDemandForecaster, create_exog_features
from src.db.connection import get_db_connection


@task(name="Fetch Historical Data", retries=2)
def fetch_historical_window(
    window_size: int = 168,
    db_host: str = "postgres",
    db_port: int = 5432,
    db_name: str = "demand_forecasting",
    db_user: str = "postgres",
    db_password: str = "postgres"
) -> pd.DataFrame:
    """
    Fetch the most recent window_size hours of data from raw_demand table
    
    Args:
        window_size: Number of hours to fetch (default: 168 = 7 days)
        
    Returns:
        DataFrame with historical data
    """
    print(f"\n📥 Fetching last {window_size} hours from database...")
    
    conn = get_db_connection(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )
    
    try:
        query = f"""
            SELECT timestamp, demand_value, hm, ta, holiday_name, weekday, weekend, 
                   spring, summer, autoum, winter, is_holiday_dummies
            FROM raw_demand
            ORDER BY timestamp DESC
            LIMIT {window_size}
        """
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) < window_size:
            print(f"⚠️  Warning: Only {len(df)} samples available (need {window_size})")
        
        # Sort by timestamp ascending
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        
        # Rename demand_value to match model training
        df.rename(columns={'demand_value': 'power demand(MW)'}, inplace=True)
        
        print(f"✓ Fetched {len(df)} samples from {df.index[0]} to {df.index[-1]}")
        
        return df
        
    finally:
        conn.close()


@task(name="Generate Daily Forecast", retries=1)
def run_daily_forecast_task(
    history_df: pd.DataFrame,
    model_dir: str,
    horizon: int,
    model_version: str
) -> pd.DataFrame:
    """
    Generate forecast using trained models
    
    Args:
        history_df: Historical data (168 hours)
        model_dir: Directory with trained models
        horizon: Forecast horizon (24 hours)
        model_version: Model version string
        
    Returns:
        DataFrame with forecasts
    """
    print("\n🔮 Generating forecast...")
    
    # Initialize forecaster
    forecaster = PowerDemandForecaster(
        model_dir=model_dir,
        window_size=len(history_df),
        horizon=horizon
    )
    
    # Load trained models
    forecaster.load_models()
    
    # Prepare historical data with trend, seasonality, residual
    prepared_history = _prepare_historical_context(history_df, forecaster)
    
    # Generate future timestamps
    last_timestamp = prepared_history.index[-1]
    future_index = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=horizon,
        freq='H'
    )
    
    # Create future exogenous features
    exog_future = create_exog_features(
        future_index,
        historical_data=prepared_history
    )
    
    # Make forecast
    forecast_dict = forecaster.forecast(
        historical_data=prepared_history,
        exog_features_future=exog_future,
        device='cuda'
    )
    
    # Create result DataFrame
    forecast_df = pd.DataFrame({
        'timestamp': future_index,
        'forecast': forecast_dict['final_forecast'],
        'trend_component': forecast_dict['trend'],
        'seasonal_component': forecast_dict['seasonality'],
        'lstm_component': forecast_dict['lstm_residual'],
        'model_version': model_version
    })
    
    print(f"✓ Generated forecast for {len(forecast_df)} hours")
    print(f"  - Period: {forecast_df['timestamp'].iloc[0]} to {forecast_df['timestamp'].iloc[-1]}")
    print(f"  - Mean forecast: {forecast_df['forecast'].mean():.2f} MW")
    print(f"  - Min/Max: {forecast_df['forecast'].min():.2f} / {forecast_df['forecast'].max():.2f} MW")
    
    return forecast_df


def _prepare_historical_context(
    df: pd.DataFrame,
    forecaster: PowerDemandForecaster
) -> pd.DataFrame:
    """
    Prepare historical data with trend, seasonality, and residual components
    
    Args:
        df: Historical DataFrame
        forecaster: Loaded PowerDemandForecaster
        
    Returns:
        DataFrame with added components
    """
    import statsmodels.api as sm
    
    prepared = df.copy()
    
    # Get train_data_length from config for correct index calculation
    if forecaster.config and 'train_data_length' in forecaster.config:
        train_data_length = forecaster.config['train_data_length']
        # Global index starts after all training data
        global_start_idx = train_data_length
    else:
        # Fallback: assume we start from 0
        global_start_idx = 0
    
    t_idx = np.arange(global_start_idx, global_start_idx + len(df))
    
    # 1. Trend
    X_hist = sm.add_constant(t_idx)
    trend_log = forecaster.trend_model.predict(X_hist)
    prepared['trend'] = np.exp(trend_log)
    prepared['detrend'] = prepared['power demand(MW)'] - prepared['trend']
    
    # 2. Fourier + Exog
    fourier_terms = np.hstack([
        forecaster.generate_fourier_terms(t_idx, 24, forecaster.Kd),
        forecaster.generate_fourier_terms(t_idx, 24 * 7, forecaster.Kw),
        forecaster.generate_fourier_terms(t_idx, int(24 * 365.25), forecaster.Ky),
    ])
    
    exog_hist = create_exog_features(prepared.index, historical_data=prepared)
    
    if forecaster.exog_scaler is not None and not exog_hist.empty:
        scaled_exog = forecaster.exog_scaler.transform(exog_hist.values)
        X_full = np.hstack([fourier_terms, scaled_exog])
    else:
        X_full = fourier_terms
    
    prepared['seasonality'] = forecaster.fourier_model.predict(X_full)
    prepared['residual'] = prepared['detrend'] - prepared['seasonality']
    
    return prepared


@task(name="Store Predictions", retries=2)
def store_predictions(
    forecast_df: pd.DataFrame,
    db_host: str = "postgres",
    db_port: int = 5432,
    db_name: str = "demand_forecasting",
    db_user: str = "postgres",
    db_password: str = "postgres"
):
    """
    Store forecast results to predictions table
    
    Args:
        forecast_df: DataFrame with forecast results
    """
    print("\n💾 Storing predictions to database...")
    
    conn = get_db_connection(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )
    
    try:
        # Insert into predictions table
        forecast_df.to_sql(
            'predictions',
            conn,
            if_exists='append',
            index=False,
            method='multi'
        )
        
        print(f"✓ Stored {len(forecast_df)} predictions")
        
    finally:
        conn.close()


@flow(
    name="Daily Hybrid Forecast",
    description="Generate 24-hour forecast using trained hybrid model"
)
def daily_forecast_flow(
    model_dir: str = "/app/models/production",
    window_size: int = 168,
    horizon: int = 24,
    model_version: str = "hybrid_v1.0"
):
    """
    Complete daily forecast flow
    
    Args:
        model_dir: Directory with trained models
        window_size: Historical window size (hours)
        horizon: Forecast horizon (hours)
        model_version: Model version identifier
    """
    print("\n" + "="*80)
    print("DAILY HYBRID FORECAST FLOW")
    print("="*80)
    print(f"Execution time: {datetime.now()}")
    print(f"Model directory: {model_dir}")
    print(f"Window size: {window_size} hours")
    print(f"Horizon: {horizon} hours")
    print("="*80)
    
    # 1. Fetch historical data
    history_df = fetch_historical_window(window_size=window_size)
    
    # 2. Generate forecast
    forecast_df = run_daily_forecast_task(
        history_df=history_df,
        model_dir=model_dir,
        horizon=horizon,
        model_version=model_version
    )
    
    # 3. Store predictions
    store_predictions(forecast_df=forecast_df)
    
    print("\n" + "="*80)
    print("✅ DAILY FORECAST FLOW COMPLETED")
    print("="*80)
    
    return {
        'forecast_count': len(forecast_df),
        'mean_forecast': float(forecast_df['forecast'].mean()),
        'forecast_period': f"{forecast_df['timestamp'].iloc[0]} to {forecast_df['timestamp'].iloc[-1]}"
    }


if __name__ == "__main__":
    # Run the flow locally
    daily_forecast_flow()

