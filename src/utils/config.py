"""
Configuration management for the demand forecasting system
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "demand_forecasting")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")
    
    # API Keys
    DEMAND_API_URL: str = os.getenv("DEMAND_API_URL", "")
    DEMAND_API_KEY: Optional[str] = os.getenv("DEMAND_API_KEY")
    WEATHER_API_URL: str = os.getenv("WEATHER_API_URL", "")
    WEATHER_API_KEY: Optional[str] = os.getenv("WEATHER_API_KEY")
    
    # Model paths
    MODEL_BASE_PATH: str = os.getenv("MODEL_BASE_PATH", "/app/models")
    PRODUCTION_MODEL_PATH: str = f"{MODEL_BASE_PATH}/production"
    TEMP_MODEL_PATH: str = f"{MODEL_BASE_PATH}/temp"
    
    # Prefect
    PREFECT_API_URL: str = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")
    
    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "demand_forecasting")
    
    # FastAPI
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Model hyperparameters
    LSTM_EPOCHS: int = int(os.getenv("LSTM_EPOCHS", "50"))
    LSTM_BATCH_SIZE: int = int(os.getenv("LSTM_BATCH_SIZE", "32"))
    LSTM_HIDDEN_UNITS: int = int(os.getenv("LSTM_HIDDEN_UNITS", "64"))
    AR_LAGS: int = int(os.getenv("AR_LAGS", "24"))
    FOURIER_ORDER: int = int(os.getenv("FOURIER_ORDER", "10"))
    
    # Forecast horizon
    FORECAST_HORIZON: int = int(os.getenv("FORECAST_HORIZON", "168"))  # 7 days in hours
    
    # Alerts
    SLACK_WEBHOOK_URL: Optional[str] = os.getenv("SLACK_WEBHOOK_URL")
    ALERT_EMAIL: Optional[str] = os.getenv("ALERT_EMAIL")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()



