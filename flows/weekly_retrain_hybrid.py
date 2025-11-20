"""
Weekly Retraining Flow for Hybrid Power Demand Forecasting
Trains trend + fourier + LSTM models every Sunday with MLflow tracking

Modified to follow test_data.ipynb style:
- Train/predict trend on ALL data (train+val+test)
- Generate Fourier terms for ALL data
- Then split for actual training
"""

from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
import os
import sys
from pathlib import Path
import statsmodels.api as sm
import logging

# Suppress SQLAlchemy connection pool warnings on exit
logging.getLogger("sqlalchemy.pool").setLevel(logging.ERROR)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.train_pipeline import PowerDemandTrainer
from src.models.inference_pipeline import PowerDemandForecaster, create_exog_features


def _prepare_context(df: pd.DataFrame,
                     forecaster: PowerDemandForecaster,
                     csv_path: str,
                     global_start_idx: int) -> pd.DataFrame:
    """
    Ensure dataframe contains trend, seasonality, residual columns for LSTM.
    
    Like test_data.ipynb: Use global index (from 0 to all_data_length)
    
    Args:
        df: Historical window data
        forecaster: Loaded forecaster with trained models
        csv_path: Path to CSV for exogenous features
        global_start_idx: Global starting index for this window (0-based, from start of ALL data)
    """
    prepared = df.copy().sort_index()
    
    # Use global index (like test_data.ipynb: t_all = np.arange(len(df)))
    t_idx = np.arange(global_start_idx, global_start_idx + len(df))
    
    # Trend
    X_hist = sm.add_constant(t_idx)
    trend_log = forecaster.trend_model.predict(X_hist)
    prepared["trend"] = np.exp(trend_log)
    prepared["detrend"] = prepared["power demand(MW)"] - prepared["trend"]
    
    # Seasonality + exog
    fourier_terms = np.hstack([
        forecaster.generate_fourier_terms(t_idx, 24, forecaster.Kd),
        forecaster.generate_fourier_terms(t_idx, 24 * 7, forecaster.Kw),
        forecaster.generate_fourier_terms(t_idx, int(24 * 365.25), forecaster.Ky),
    ])
    
    exog_hist = create_exog_features(prepared.index, historical_data=prepared, csv_path=csv_path)
    if forecaster.exog_scaler is not None and not exog_hist.empty:
        scaled_exog = forecaster.exog_scaler.transform(exog_hist.values)
        X_full = np.hstack([fourier_terms, scaled_exog])
    else:
        X_full = fourier_terms
    
    prepared["seasonality"] = forecaster.fourier_model.predict(X_full)
    prepared["residual"] = prepared["detrend"] - prepared["seasonality"]
    return prepared


from src.utils.grafana_client import GrafanaClient


@task(name="Load Training Data", retries=2)
def load_data(csv_path: str, train_end: str = None, val_end: str = None, 
              train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Load and split data into train/val/test sets
    
    Default: Use only 2019-2021 data for training/validation/testing (80:10:10)
    2022+ data is reserved for real inference (not included in training).
    
    Args:
        csv_path: Path to CSV file
        train_end: End date for training data (YYYY-MM-DD), optional
        val_end: End date for validation data (YYYY-MM-DD), optional
        train_ratio: Ratio of data for training (default: 0.8)
        val_ratio: Ratio of data for validation (default: 0.1)
        
    Returns:
        (train_data, val_data, test_data): DataFrames
    """
    print(f"\n📂 Loading data from: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    df["일시"] = pd.to_datetime(df["일시"])
    df["holiday_name"].fillna("non-event", inplace=True)
    df.set_index("일시", inplace=True)
    df = df.sort_index()
    
    print(f"✓ Full data loaded: {len(df)} samples from {df.index[0]} to {df.index[-1]}")
    
    # Filter to 2019-2021 data for training only
    if train_end is None or val_end is None:
        # Use only 2019-2021 data
        df_train_period = df[(df.index >= "2019-01-01") & (df.index < "2022-01-01")].copy()
        
        print(f"\n📅 Using 2019-2021 data for training/validation/test:")
        print(f"  - Total samples: {len(df_train_period):,} ({df_train_period.index[0]} to {df_train_period.index[-1]})")
        print(f"  - Split ratio: Train {train_ratio:.0%}, Val {val_ratio:.0%}, Test {1-train_ratio-val_ratio:.0%}")
        
        # Automatic split based on ratios
        n_total = len(df_train_period)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_end = df_train_period.index[n_train]
        val_end = df_train_period.index[n_train + n_val]
        
        print(f"  - Train end: {train_end}")
        print(f"  - Val end: {val_end}")
        
        # Use filtered data
        df = df_train_period
    
    # Split data
    train_data = df[df.index < train_end].copy()
    val_data = df[(df.index >= train_end) & (df.index < val_end)].copy()
    test_data = df[df.index >= val_end].copy()
    
    # Drop holiday_name (not used in modeling)
    for data in [train_data, val_data, test_data]:
        if 'holiday_name' in data.columns:
            data.drop(columns=['holiday_name'], inplace=True)
    
    print(f"\n✓ Data split complete:")
    print(f"  - Train: {len(train_data):,} samples ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"  - Val:   {len(val_data):,} samples ({val_data.index[0]} to {val_data.index[-1]})")
    print(f"  - Test:  {len(test_data):,} samples ({test_data.index[0]} to {test_data.index[-1]})")
    print(f"\n⚠️  2022+ data is NOT used for training - reserved for real inference!")
    
    return train_data, val_data, test_data


@task(name="Train Complete Pipeline", retries=1, timeout_seconds=7200)
def train_pipeline(train_data: pd.DataFrame, 
                   val_data: pd.DataFrame,
                   test_data: pd.DataFrame,
                   model_dir: str,
                   window_size: int = 168,
                   horizon: int = 24,
                   n_lstm_iter: int = 50,
                   lstm_epochs: int = 100):
    """
    Train complete pipeline: trend + fourier + LSTM
    
    Like test_data.ipynb: Train/predict trend and fourier on ALL data (train+val+test),
    then split for actual training.
    
    Returns:
        Dictionary with training metrics
    """
    print("\n" + "="*80)
    print("STARTING COMPLETE TRAINING PIPELINE (test_data.ipynb style)")
    print("="*80)
    
    # Combine all data for global trend/fourier calculation (like test_data.ipynb)
    all_data = pd.concat([train_data, val_data, test_data])
    print(f"\n📊 Data summary:")
    print(f"  - Train: {len(train_data):,} samples ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"  - Val:   {len(val_data):,} samples ({val_data.index[0]} to {val_data.index[-1]})")
    print(f"  - Test:  {len(test_data):,} samples ({test_data.index[0]} to {test_data.index[-1]})")
    print(f"  - Total: {len(all_data):,} samples")
    
    # Initialize trainer
    trainer = PowerDemandTrainer(
        model_dir=model_dir,
        window_size=window_size,
        horizon=horizon
    )
    
    # 1. Train trend model on TRAIN data only (but predict on ALL data)
    print("\n📈 Training Trend Model (test_data.ipynb style)...")
    print(f"  - Training on: {len(train_data):,} samples")
    trend_pred_train = trainer.train_trend(train_data, target_col="power demand(MW)")
    
    # Predict trend for ALL data (like test_data.ipynb: t_all = np.arange(len(df)))
    t_all = np.arange(len(all_data))
    X_all = sm.add_constant(t_all)
    trend_pred_all = np.exp(trainer.trend_model.predict(X_all))
    
    print(f"  - Predicting for ALL data: t=0 to t={len(all_data)-1}")
    
    # Split trend predictions back to train/val/test
    all_data['trend'] = trend_pred_all
    all_data['detrend'] = all_data["power demand(MW)"] - all_data['trend']
    
    train_data = all_data.iloc[:len(train_data)].copy()
    val_data = all_data.iloc[len(train_data):len(train_data)+len(val_data)].copy()
    test_data = all_data.iloc[len(train_data)+len(val_data):].copy()
    
    mlflow.log_metric("trend_r2", trainer.trend_model.rsquared)
    
    # 2. Train Fourier seasonality model with Grid Search (on ALL data, like test_data.ipynb)
    print("\n🌊 Training Fourier Seasonality Model (test_data.ipynb style)...")
    exog_features = ["hm", "ta", "weekday", "weekend", 
                     "spring", "summer", "autoum", "winter", "is_holiday_dummies"]
    
    # Check which features exist
    exog_features = [f for f in exog_features if f in all_data.columns]
    
    # Generate Fourier terms for ALL data (like test_data.ipynb)
    # Then split into train/val for grid search
    print(f"  - Generating Fourier terms for ALL data: t=0 to t={len(all_data)-1}")
    
    # Split detrended data
    train_detrend = train_data['detrend'].copy()
    val_detrend = val_data['detrend'].copy()
    
    # Grid Search over Kd, Kw, Ky (each from 1 to 14)
    # Pass all_data length so trainer knows to generate Fourier terms for entire range
    train_seasonality, val_seasonality = trainer.train_fourier(
        train_data=train_data,
        val_data=val_data,
        detrended_col="detrend",
        exog_features=exog_features if exog_features else None,
        grid_search=True,
        Kd_range=np.arange(1, 15),  # Daily harmonics: 1 to 14
        Kw_range=np.arange(1, 15),  # Weekly harmonics: 1 to 14
        Ky_range=np.arange(1, 15),   # Yearly harmonics: 1 to 14
        all_data_length=len(all_data)  # NEW: tells trainer to use global index
    )
    
    train_data['seasonality'] = train_seasonality
    val_data['seasonality'] = val_seasonality
    
    mlflow.log_param("fourier_Kd", trainer.best_fourier_params['Kd'])
    mlflow.log_param("fourier_Kw", trainer.best_fourier_params['Kw'])
    mlflow.log_param("fourier_Ky", trainer.best_fourier_params['Ky'])
    
    # 3. Calculate residuals
    train_data['residual'] = train_data['detrend'] - train_data['seasonality']
    val_data['residual'] = val_data['detrend'] - val_data['seasonality']
    
    # 4. Train LSTM on residuals
    print("\n🤖 Training LSTM Residual Model...")
    mlflow.log_param("lstm_window_size", window_size)
    mlflow.log_param("lstm_horizon", horizon)
    
    train_residuals = train_data['residual'].values
    val_residuals = val_data['residual'].values
    
    best_val_loss = trainer.train_lstm(
        train_residuals=train_residuals,
        val_residuals=val_residuals,
        n_iter=n_lstm_iter,
        epochs=lstm_epochs
    )
    
    mlflow.log_metric("lstm_best_val_loss", best_val_loss)
    
    # Log best LSTM hyperparameters
    for param_name, param_value in trainer.best_lstm_params.items():
        mlflow.log_param(f"lstm_{param_name}", param_value)
    
    # 5. Save all models (include train_data_length for correct index calculation)
    trainer.save_models(train_data_length=len(train_data))
    
    # Log model artifacts to MLflow
    mlflow.log_artifacts(model_dir, artifact_path="models")
    
    print("\n✅ Training pipeline complete!")
    
    return {
        'trend_r2': trainer.trend_model.rsquared,
        'fourier_Kd': trainer.best_fourier_params['Kd'],
        'fourier_Kw': trainer.best_fourier_params['Kw'],
        'fourier_Ky': trainer.best_fourier_params['Ky'],
        'lstm_val_loss': best_val_loss
    }


@task(name="Evaluate on Test Set", retries=1)
def evaluate_model(test_data: pd.DataFrame,
                   model_dir: str,
                   csv_path: str,
                   window_size: int = 168,
                   horizon: int = 24):
    """
    Evaluate trained model on test set
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*80)
    print("EVALUATING MODEL ON TEST SET")
    print("="*80)
    
    # Initialize forecaster
    forecaster = PowerDemandForecaster(
        model_dir=model_dir,
        window_size=window_size,
        horizon=horizon
    )
    
    # Load trained models
    forecaster.load_models()
    
    predictions = []
    actuals = []
    
    # Sliding window evaluation (stride = horizon, non-overlapping like test_data.ipynb)
    stride = horizon
    test_values = test_data["power demand(MW)"].values
    test_index = test_data.index
    
    # Calculate how many windows we can create (non-overlapping)
    n_windows = (len(test_data) - window_size - horizon + 1) // stride
    
    print(f"\n📊 Test data length: {len(test_data)} samples")
    print(f"📊 Window size: {window_size}, Horizon: {horizon}, Stride: {stride}")
    print(f"📊 Calculated n_windows (non-overlapping): {n_windows}")
    
    if n_windows <= 0:
        print("⚠️  Warning: Not enough test data for evaluation. Skipping...")
        return {
            'metrics': {
                'mae': 0.0,
                'mse': 0.0,
                'rmse': 0.0,
                'r2': 0.0,
                'mape': 0.0,
                'smape': 0.0
            },
            'predictions': np.array([]),
            'actuals': np.array([])
        }
    
    print(f"\nMaking {n_windows} sliding window forecasts on test data (stride={stride})...")
    
    # Get train_data_length from config to calculate global index
    if forecaster.config and 'train_data_length' in forecaster.config:
        train_data_length = forecaster.config['train_data_length']
        print(f"  ℹ️  Using train_data_length={train_data_length} from model config")
    else:
        print("  ⚠️  Warning: train_data_length not found in config, using 0")
        train_data_length = 0
    
    for i in range(n_windows):
        # Get historical window (non-overlapping: stride = horizon)
        start_idx = i * stride
        end_idx = start_idx + window_size
        future_end_idx = end_idx + horizon
        
        if future_end_idx > len(test_data):
            break
        
        # Extract window data
        historical_window = test_data.iloc[start_idx:end_idx].copy()
        actual_future_window = test_data.iloc[end_idx:future_end_idx].copy()
        
        # Calculate global index (like test_data.ipynb)
        # Global index = train_data_length (e.g., 21043) + start_idx (e.g., 0, 24, 48, ...)
        # This ensures we use the same index as during training
        global_start_idx = train_data_length + start_idx
        
        # Prepare context for this specific window (each window needs independent context)
        prepared_historical = _prepare_context(historical_window, forecaster, csv_path, global_start_idx=global_start_idx)
        
        # Create future exog features from CSV (treats CSV as forecast data)
        future_timestamps = actual_future_window.index
        exog_future = create_exog_features(future_timestamps, historical_data=prepared_historical, csv_path=csv_path)
        
        # Make forecast
        try:
            forecast_dict = forecaster.forecast(
                historical_data=prepared_historical,
                exog_features_future=exog_future,
                device='cuda'
            )
            
            predictions.append(forecast_dict['final_forecast'])
            actuals.append(actual_future_window["power demand(MW)"].values)
        except Exception as e:
            print(f"  ⚠ Skipping window {i}: {e}")
            continue
    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    if predictions.size == 0 or actuals.size == 0:
        print("⚠️  No successful evaluation windows. Skipping metric computation.")
        return {
            'metrics': {
                'mae': 0.0,
                'mse': 0.0,
                'rmse': 0.0,
                'r2': 0.0,
                'mape': 0.0,
                'smape': 0.0
            },
            'predictions': predictions,
            'actuals': actuals
        }
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
    
    def smape(y_true, y_pred):
        """Symmetric Mean Absolute Percentage Error"""
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        diff = np.abs(y_true - y_pred)
        mask = denominator == 0
        denominator[mask] = 1
        diff[mask] = 0
        return np.mean(diff / denominator) * 100
    
    metrics = {
        'mae': float(mean_absolute_error(actuals, predictions)),
        'mse': float(mean_squared_error(actuals, predictions)),
        'rmse': float(np.sqrt(mean_squared_error(actuals, predictions))),
        'r2': float(r2_score(actuals.flatten(), predictions.flatten())),
        'mape': float(mean_absolute_percentage_error(actuals.flatten(), predictions.flatten()) * 100),
        'smape': float(smape(actuals, predictions))
    }
    
    # Log metrics to MLflow
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(f"test_{metric_name}", metric_value)
    
    print("\n📊 Test Set Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  - {metric_name.upper()}: {metric_value:.4f}")
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'actuals': actuals
    }


@task(name="Send to Grafana", retries=2)
def send_to_grafana(
    evaluation_results: dict,
    test_data: pd.DataFrame,
    horizon: int
):
    """
    Send evaluation results to Grafana (InfluxDB)
    
    Args:
        evaluation_results: Results from evaluate_model
        test_data: Test dataset
        horizon: Forecast horizon
    """
    print("\n" + "="*80)
    print("SENDING RESULTS TO GRAFANA")
    print("="*80)
    
    # Initialize Grafana client
    grafana = GrafanaClient()
    print("✓ Grafana client initialized")
    print(f"  - URL: {grafana.url}")
    print(f"  - Org: {grafana.org}")
    print(f"  - Bucket: {grafana.bucket}")
    
    # Send metrics
    metrics = evaluation_results['metrics']
    print("\n📤 Sending metrics to Grafana...")
    grafana.send_metrics(metrics)
    print("✅ Sent metrics to Grafana")
    
    print(f"\n📊 Metrics:")
    print(f"  - MAE: {metrics['mae']:.4f}")
    print(f"  - MSE: {metrics['mse']:.4f}")
    print(f"  - RMSE: {metrics['rmse']:.4f}")
    print(f"  - R2: {metrics['r2']:.4f}")
    print(f"  - MAPE: {metrics['mape']:.4f}")
    print(f"  - SMAPE: {metrics['smape']:.4f}")


@flow(
    name="Weekly Hybrid Model Retraining",
    description="Train trend + fourier + LSTM models with MLflow tracking",
    task_runner=ConcurrentTaskRunner()
)
def weekly_retrain_flow(
    csv_path: str = "/app/models/power_demand_final.csv",
    model_dir: str = "models/production",
    train_end: Optional[str] = None,
    val_end: Optional[str] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    window_size: int = 168,
    horizon: int = 24,
    n_lstm_iter: int = 50,
    lstm_epochs: int = 100
):
    """
    Complete weekly retraining flow
    
    Default behavior: Use 2019-2021 data only (80:10:10 split)
    2022+ data is reserved for real inference.
    
    Args:
        csv_path: Path to CSV data file
        model_dir: Directory to save models
        train_end: End date for training (optional, default: auto from 2019-2021)
        val_end: End date for validation (optional, default: auto from 2019-2021)
        train_ratio: Training data ratio (default: 0.8)
        val_ratio: Validation data ratio (default: 0.1)
        window_size: LSTM window size (hours)
        horizon: Forecast horizon (hours)
        n_lstm_iter: Number of random search iterations for LSTM
        lstm_epochs: Maximum epochs per LSTM trial
    """
    print("\n" + "="*80)
    print("WEEKLY HYBRID MODEL RETRAINING FLOW")
    print("="*80)
    print(f"Execution time: {datetime.now()}")
    print(f"CSV path: {csv_path}")
    print(f"Model directory: {model_dir}")
    print(f"Train end: {train_end if train_end else 'Auto (2019-2021 data)'}")
    print(f"Val end: {val_end if val_end else 'Auto (2019-2021 data)'}")
    print(f"Data split ratio: Train {train_ratio:.0%}, Val {val_ratio:.0%}, Test {1-train_ratio-val_ratio:.0%}")
    print(f"Window size: {window_size} hours")
    print(f"Horizon: {horizon} hours")
    print("="*80)
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("power-demand-hybrid-weekly")
    
    try:
        with mlflow.start_run(run_name=f"weekly_retrain_{datetime.now().strftime('%Y%m%d')}"):
            # Log run parameters
            mlflow.log_param("csv_path", csv_path)
            mlflow.log_param("train_end", train_end)
            mlflow.log_param("val_end", val_end)
            mlflow.log_param("train_ratio", train_ratio)
            mlflow.log_param("val_ratio", val_ratio)
            mlflow.log_param("window_size", window_size)
            mlflow.log_param("horizon", horizon)
            mlflow.log_param("n_lstm_iter", n_lstm_iter)
            mlflow.log_param("lstm_epochs", lstm_epochs)
            
            # 1. Load data (2019-2021 only if train_end/val_end are None)
            train_data, val_data, test_data = load_data(csv_path, train_end, val_end, train_ratio, val_ratio)
            
            # 2. Train pipeline (pass test_data for global trend/fourier calculation)
            training_metrics = train_pipeline(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                model_dir=model_dir,
                window_size=window_size,
                horizon=horizon,
                n_lstm_iter=n_lstm_iter,
                lstm_epochs=lstm_epochs
            )
            
            # 3. Evaluate on test set
            evaluation_results = evaluate_model(
                test_data=test_data,
                model_dir=model_dir,
                csv_path=csv_path,
                window_size=window_size,
                horizon=horizon
            )
            
            # 4. Send to Grafana
            send_to_grafana(
                evaluation_results=evaluation_results,
                test_data=test_data,
                horizon=horizon
            )
            
            print("\n" + "="*80)
            print("✅ WEEKLY RETRAINING FLOW COMPLETED")
            print("="*80)
            
            result = {
                'training_metrics': training_metrics,
                'evaluation_metrics': evaluation_results['metrics']
            }
    finally:
        # Cleanup: Close any open connections
        # This helps prevent SQLAlchemy connection pool errors on exit
        import gc
        gc.collect()
    
    return result


if __name__ == "__main__":
    # Run the flow
    weekly_retrain_flow(
        window_size=168,
        horizon=24,
        n_lstm_iter=5,  # Reduced for testing
        lstm_epochs=30   # Reduced for testing
    )
