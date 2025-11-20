"""
FastAPI Server for Demand Forecasting System
Provides REST API endpoints for predictions, metrics, and system status
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict
import pandas as pd

from src.models.inference_pipeline import PowerDemandForecaster
from src.db.connection import read_sql_to_dataframe
from src.utils.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Demand Forecasting API",
    description="MLOps system for demand forecasting using hybrid time series model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class PredictionResponse(BaseModel):
    timestamp: datetime
    forecast: float
    trend: Optional[float] = None
    seasonality: Optional[float] = None
    residual: Optional[float] = None
    model_version: Optional[str] = None


class MetricsResponse(BaseModel):
    model_version: str
    mape: float
    rmse: float
    mae: float
    r2: float
    last_updated: datetime


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    database_connected: bool
    model_loaded: bool
    last_prediction: Optional[datetime] = None


class PredictRequest(BaseModel):
    timestamp: Optional[datetime] = None
    features: Optional[Dict] = None
    n_steps: int = 24


# Global model cache
_forecaster_cache: Optional[PowerDemandForecaster] = None


def get_forecaster() -> PowerDemandForecaster:
    """Load and cache the production hybrid forecaster"""
    global _forecaster_cache

    if _forecaster_cache is None:
        try:
            _forecaster_cache = PowerDemandForecaster(
                model_dir=settings.PRODUCTION_MODEL_PATH
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model not available: {exc}")
    return _forecaster_cache


def _fetch_historical_window(hours: int) -> pd.DataFrame:
    """
    Fetch the latest historical window from raw_demand for inference.

    Returns dataframe indexed by timestamp with column 'power demand(MW)'.
    """
    query = """
        SELECT timestamp,
               demand_value,
               hm,
               ta,
               holiday_name,
               weekday,
               weekend,
               spring,
               summer,
               autoum,
               winter,
               is_holiday_dummies
        FROM raw_demand
        ORDER BY timestamp DESC
        LIMIT %s
    """
    df = read_sql_to_dataframe(query, (hours,))
    if df.empty:
        raise HTTPException(
            status_code=503, detail="Insufficient historical data for inference"
        )
    df = df.rename(columns={"demand_value": "power demand(MW)"})
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    return df


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Demand Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "predictions": "/api/predictions/latest",
            "components": "/api/predictions/components",
            "metrics": "/api/metrics/performance",
            "health": "/api/health"
        }
    }


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """System health check"""
    # Check database connection
    db_connected = False
    try:
        query = "SELECT 1"
        result = read_sql_to_dataframe(query)
        db_connected = not result.empty
    except:
        db_connected = False
    
    # Check model availability
    model_loaded = False
    try:
        get_forecaster()
        model_loaded = True
    except Exception:
        model_loaded = False
    
    # Get last prediction timestamp
    last_prediction = None
    try:
        query = "SELECT MAX(timestamp) as last_pred FROM predictions"
        result = read_sql_to_dataframe(query)
        if not result.empty and result.iloc[0]['last_pred'] is not None:
            last_prediction = result.iloc[0]['last_pred']
    except:
        pass
    
    status = "healthy" if (db_connected and model_loaded) else "degraded"
    
    return HealthResponse(
        status=status,
        timestamp=datetime.now(),
        database_connected=db_connected,
        model_loaded=model_loaded,
        last_prediction=last_prediction
    )


@app.get("/api/predictions/latest", response_model=List[PredictionResponse])
def get_latest_predictions(
    hours: int = Query(default=168, ge=1, le=720, description="Number of hours to retrieve")
):
    """
    Get latest predictions
    
    Args:
        hours: Number of hours of predictions to retrieve (default: 168 = 7 days)
    
    Returns:
        List of predictions
    """
    query = """
    SELECT 
        timestamp,
        forecast,
        trend_component as trend,
        seasonal_component as seasonality,
        lstm_component as residual,
        model_version
    FROM predictions
    WHERE timestamp >= NOW() AND timestamp <= NOW() + INTERVAL '%s hours'
    ORDER BY timestamp
    LIMIT %s
    """
    
    try:
        df = read_sql_to_dataframe(query, (hours, hours))
        
        if df.empty:
            # If no future predictions, get most recent ones
            query_fallback = """
        SELECT 
            timestamp,
            forecast,
            trend_component as trend,
            seasonal_component as seasonality,
            lstm_component as residual,
            model_version
        FROM predictions
        ORDER BY created_at DESC
        LIMIT %s
            """
            df = read_sql_to_dataframe(query_fallback, (hours,))
        
        predictions = df.to_dict('records')
        return [PredictionResponse(**pred) for pred in predictions]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching predictions: {str(e)}")


@app.get("/api/predictions/components")
def get_prediction_components(
    date: str = Query(..., description="Date in YYYY-MM-DD format")
):
    """
    Get component decomposition for a specific date
    
    Args:
        date: Date in YYYY-MM-DD format
    
    Returns:
        Dictionary with component values
    """
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    query = """
    SELECT 
        timestamp,
        forecast,
        trend_component,
        seasonal_component,
        lstm_component
    FROM predictions
    WHERE DATE(timestamp) = %s
    ORDER BY timestamp
    """
    
    try:
        df = read_sql_to_dataframe(query, (target_date.date(),))
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No predictions found for {date}")
        
        # Calculate average components for the day
        components = {
            "date": date,
            "num_predictions": len(df),
            "avg_forecast": float(df['forecast'].mean()),
            "avg_trend": float(df['trend_component'].mean()) if df['trend_component'].notna().any() else None,
            "avg_seasonality": float(df['seasonal_component'].mean()) if df['seasonal_component'].notna().any() else None,
            "avg_residual": float(df['lstm_component'].mean()) if df['lstm_component'].notna().any() else None,
            "hourly_data": df.to_dict('records')
        }
        
        return components
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching components: {str(e)}")


@app.get("/api/metrics/performance", response_model=List[MetricsResponse])
def get_performance_metrics(
    limit: int = Query(default=10, ge=1, le=100, description="Number of recent metrics to retrieve")
):
    """
    Get model performance metrics
    
    Args:
        limit: Number of recent metric records to retrieve
    
    Returns:
        List of performance metrics
    """
    query = """
    SELECT 
        model_version,
        mape,
        rmse,
        mae,
        r2,
        run_timestamp as last_updated
    FROM model_metrics
    ORDER BY run_timestamp DESC
    LIMIT %s
    """
    
    try:
        df = read_sql_to_dataframe(query, (limit,))
        
        if df.empty:
            return []
        
        metrics = df.to_dict('records')
        return [MetricsResponse(**m) for m in metrics]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {str(e)}")


@app.post("/api/predict")
def predict_on_demand(request: PredictRequest):
    """
    Generate on-demand predictions
    
    Args:
        request: Prediction request with optional timestamp and features
    
    Returns:
        Predictions for requested time steps
    """
    try:
        forecaster = get_forecaster()
        horizon = max(1, request.n_steps)

        history_hours = forecaster.window_size + horizon
        history_df = _fetch_historical_window(history_hours)

        forecast_df = forecaster.forecast(
            historical_data=history_df, horizon=horizon
        )

        model_version = (
            forecaster.config.get("model_version")
            if forecaster.config
            else "production"
        )

        predictions = []
        for ts, row in forecast_df.iterrows():
            predictions.append(
                {
                    "timestamp": ts.to_pydatetime(),
                    "forecast": float(row["forecast"]),
                    "trend": float(row["trend"]),
                    "seasonality": float(row["seasonality"]),
                    "residual": float(row["residual"]),
                    "model_version": model_version,
                }
            )

        return {
            "status": "success",
            "n_predictions": len(predictions),
            "predictions": predictions,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/actual-vs-forecast")
def get_actual_vs_forecast(
    hours: int = Query(default=168, ge=1, le=720, description="Number of hours to compare")
):
    """
    Compare actual demand with forecasts
    
    Args:
        hours: Number of hours to compare
    
    Returns:
        Comparison data
    """
    query = """
    SELECT 
        p.timestamp,
        p.forecast,
        rd.demand_value as actual,
        ABS(p.forecast - rd.demand_value) as error,
        ABS((p.forecast - rd.demand_value) / rd.demand_value * 100) as error_pct
    FROM predictions p
    INNER JOIN raw_demand rd ON p.timestamp = rd.timestamp
    WHERE p.timestamp >= NOW() - INTERVAL '%s hours'
    ORDER BY p.timestamp DESC
    LIMIT %s
    """
    
    try:
        df = read_sql_to_dataframe(query, (hours, hours))
        
        if df.empty:
            return {"message": "No comparison data available", "data": []}
        
        # Calculate summary statistics
        summary = {
            "mean_error": float(df['error'].mean()),
            "mean_error_pct": float(df['error_pct'].mean()),
            "max_error": float(df['error'].max()),
            "min_error": float(df['error'].min())
        }
        
        return {
            "summary": summary,
            "data": df.to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching comparison: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)



