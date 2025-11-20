"""
MLflow tracking utilities for experiment tracking and model registry
"""
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from datetime import datetime
from typing import Dict, Optional, Any
import os
from src.utils.config import settings


def init_mlflow():
    """Initialize MLflow tracking"""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)


def start_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """Start a new MLflow run"""
    init_mlflow()
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return mlflow.start_run(run_name=run_name, tags=tags or {})


def log_params(params: Dict[str, Any]):
    """Log parameters to MLflow"""
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """Log metrics to MLflow"""
    mlflow.log_metrics(metrics, step=step)


def log_model(model, artifact_path: str, model_type: str = "pytorch"):
    """Log model to MLflow"""
    if model_type == "pytorch":
        mlflow.pytorch.log_model(model, artifact_path)
    elif model_type == "sklearn":
        mlflow.sklearn.log_model(model, artifact_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def log_artifacts(local_dir: str, artifact_path: Optional[str] = None):
    """Log artifacts to MLflow"""
    mlflow.log_artifacts(local_dir, artifact_path)


def log_artifact(local_path: str, artifact_path: Optional[str] = None):
    """Log a single artifact to MLflow"""
    mlflow.log_artifact(local_path, artifact_path)


def end_run(status: str = "FINISHED"):
    """End the current MLflow run"""
    mlflow.end_run(status=status)


class MLflowTracker:
    """Context manager for MLflow tracking"""
    
    def __init__(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        self.run_name = run_name
        self.tags = tags or {}
        self.run = None
    
    def __enter__(self):
        init_mlflow()
        if self.run_name is None:
            self.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            mlflow.end_run(status="FAILED")
        else:
            mlflow.end_run(status="FINISHED")
        return False
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model, artifact_path: str, model_type: str = "pytorch"):
        """Log model"""
        if model_type == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path)
        elif model_type == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log artifacts"""
        mlflow.log_artifacts(local_dir, artifact_path)


def log_forecast_metrics(metrics: Dict[str, float], model_version: str, run_name: Optional[str] = None):
    """Log forecast metrics to MLflow"""
    with MLflowTracker(run_name=run_name or f"forecast_{model_version}", tags={"type": "forecast"}):
        mlflow.log_params({"model_version": model_version})
        mlflow.log_metrics(metrics)


def log_training_metrics(
    metrics: Dict[str, float],
    hyperparameters: Dict[str, Any],
    model_version: str,
    run_name: Optional[str] = None
):
    """Log training metrics and hyperparameters to MLflow"""
    with MLflowTracker(run_name=run_name or f"training_{model_version}", tags={"type": "training"}):
        mlflow.log_params({**hyperparameters, "model_version": model_version})
        mlflow.log_metrics(metrics)

