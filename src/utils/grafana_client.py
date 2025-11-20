"""
Grafana Integration for Forecast Visualization and Metrics
Sends predictions and evaluation metrics to Grafana via InfluxDB
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import os


class GrafanaClient:
    """
    Client for sending forecast data and metrics to Grafana via InfluxDB
    """
    
    def __init__(self,
                 url: str = None,
                 token: str = None,
                 org: str = None,
                 bucket: str = "power_demand"):
        """
        Initialize Grafana client
        
        Args:
            url: InfluxDB URL (default: from env)
            token: InfluxDB token (default: from env)
            org: InfluxDB organization (default: from env)
            bucket: InfluxDB bucket name
        """
        self.url = url or os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.token = token or os.getenv("INFLUXDB_TOKEN", "")
        self.org = org or os.getenv("INFLUXDB_ORG", "open-stef")
        self.bucket = bucket
        
        # Initialize client
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        
        print(f"✓ Grafana client initialized")
        print(f"  - URL: {self.url}")
        print(f"  - Org: {self.org}")
        print(f"  - Bucket: {self.bucket}")
    
    def send_forecast(self,
                     forecast_df: pd.DataFrame,
                     measurement: str = "power_demand_forecast"):
        """
        Send forecast data to InfluxDB
        
        Args:
            forecast_df: DataFrame with columns [timestamp, trend, seasonality, residual, forecast]
            measurement: InfluxDB measurement name
        """
        print(f"\n📤 Sending forecast to Grafana...")
        
        points = []
        for idx, row in forecast_df.iterrows():
            timestamp = idx if isinstance(idx, pd.Timestamp) else row['timestamp']
            
            point = Point(measurement) \
                .time(timestamp) \
                .field("trend", float(row['trend'])) \
                .field("seasonality", float(row['seasonality'])) \
                .field("residual", float(row['residual'])) \
                .field("forecast", float(row['forecast'])) \
                .tag("model_type", "hybrid") \
                .tag("horizon", len(forecast_df))
            
            points.append(point)
        
        # Write to InfluxDB
        try:
            self.write_api.write(bucket=self.bucket, record=points)
            print(f"✅ Sent {len(points)} forecast points to Grafana")
        except Exception as e:
            print(f"❌ Error sending forecast: {e}")
    
    def send_actual(self,
                   actual_df: pd.DataFrame,
                   value_col: str = "power demand(MW)",
                   measurement: str = "power_demand_actual",
                   include_columns: Optional[List[str]] = None,
                   tag_columns: Optional[List[str]] = None):
        """
        Send actual power demand data to InfluxDB
        
        Args:
            actual_df: DataFrame with actual values
            value_col: Column name for actual values
            measurement: InfluxDB measurement name
            include_columns: Additional numeric columns to store as fields (defaults to all other columns)
            tag_columns: Columns that should be stored as tags (e.g., holiday_name)
        """
        print(f"\n📤 Sending actual data to Grafana...")
        
        points = []
        feature_columns = include_columns or [col for col in actual_df.columns if col != value_col]
        tag_columns = set(tag_columns or [])

        for idx, row in actual_df.iterrows():
            timestamp = idx if isinstance(idx, pd.Timestamp) else row['timestamp']
            
            point = Point(measurement) \
                .time(timestamp) \
                .field("actual", float(row[value_col])) \
                .tag("data_type", "historical")
            
            for col in feature_columns:
                if col not in actual_df.columns:
                    continue
                value = row[col]
                if pd.isna(value):
                    continue
                if col in tag_columns:
                    point = point.tag(col, str(value))
                else:
                    point = point.field(col, float(value))
            
            points.append(point)
        
        # Write to InfluxDB
        try:
            self.write_api.write(bucket=self.bucket, record=points)
            print(f"✅ Sent {len(points)} actual points to Grafana")
        except Exception as e:
            print(f"❌ Error sending actual data: {e}")
    
    def calculate_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        def smape(y_true, y_pred):
            """Symmetric Mean Absolute Percentage Error"""
            y_true = np.array(y_true).reshape(-1)
            y_pred = np.array(y_pred).reshape(-1)
            
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            diff = np.abs(y_true - y_pred)
            
            mask = denominator == 0
            denominator[mask] = 1
            diff[mask] = 0
            
            return np.mean(diff / denominator) * 100
        
        metrics = {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred)),
            'mape': float(mean_absolute_percentage_error(y_true, y_pred) * 100),
            'smape': float(smape(y_true, y_pred))
        }
        
        return metrics
    
    def send_metrics(self,
                    metrics: Dict[str, float],
                    horizon: int = None,
                    measurement: str = "forecast_metrics"):
        """
        Send evaluation metrics to InfluxDB
        
        Args:
            metrics: Dictionary with metric names and values
            horizon: Forecast horizon (for tagging)
            measurement: InfluxDB measurement name
        """
        print(f"\n📤 Sending metrics to Grafana...")
        
        timestamp = datetime.now()
        
        point = Point(measurement) \
            .time(timestamp)
        
        # Add metric fields
        for metric_name, metric_value in metrics.items():
            point = point.field(metric_name, metric_value)
        
        # Add tags
        point = point.tag("model_type", "hybrid")
        if horizon is not None:
            point = point.tag("horizon", str(horizon))
        
        # Write to InfluxDB
        try:
            self.write_api.write(bucket=self.bucket, record=point)
            print(f"✅ Sent metrics to Grafana")
            print("\n📊 Metrics:")
            for name, value in metrics.items():
                print(f"  - {name.upper()}: {value:.4f}")
        except Exception as e:
            print(f"❌ Error sending metrics: {e}")
    
    def send_horizon_metrics(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           horizon: int = 24,
                           measurement: str = "horizon_metrics"):
        """
        Calculate and send metrics for each forecast horizon step
        
        Args:
            y_true: True values (N, horizon)
            y_pred: Predicted values (N, horizon)
            horizon: Forecast horizon
            measurement: InfluxDB measurement name
        """
        print(f"\n📤 Sending horizon-specific metrics to Grafana...")
        
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, horizon)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, horizon)
        
        timestamp = datetime.now()
        points = []
        
        for h in range(horizon):
            y_true_h = y_true[:, h]
            y_pred_h = y_pred[:, h]
            
            metrics = self.calculate_metrics(y_true_h, y_pred_h)
            
            point = Point(measurement) \
                .time(timestamp) \
                .tag("model_type", "hybrid") \
                .tag("horizon_step", str(h+1))
            
            for metric_name, metric_value in metrics.items():
                point = point.field(metric_name, metric_value)
            
            points.append(point)
        
        # Write to InfluxDB
        try:
            self.write_api.write(bucket=self.bucket, record=points)
            print(f"✅ Sent horizon metrics for {horizon} steps to Grafana")
        except Exception as e:
            print(f"❌ Error sending horizon metrics: {e}")
    
    def send_component_metrics(self,
                              actual: np.ndarray,
                              trend: np.ndarray,
                              seasonality: np.ndarray,
                              residual: np.ndarray,
                              measurement: str = "component_metrics"):
        """
        Send metrics for individual forecast components
        
        Args:
            actual: Actual values
            trend: Trend predictions
            seasonality: Seasonality predictions
            residual: Residual predictions
            measurement: InfluxDB measurement name
        """
        print(f"\n📤 Sending component metrics to Grafana...")
        
        timestamp = datetime.now()
        
        # Calculate metrics for each component
        components = {
            'trend': trend,
            'seasonality': seasonality,
            'residual': residual
        }
        
        for component_name, component_pred in components.items():
            # For components, we measure their contribution
            metrics = {
                'mean': float(np.mean(component_pred)),
                'std': float(np.std(component_pred)),
                'min': float(np.min(component_pred)),
                'max': float(np.max(component_pred)),
                'range': float(np.max(component_pred) - np.min(component_pred))
            }
            
            point = Point(measurement) \
                .time(timestamp) \
                .tag("component", component_name)
            
            for metric_name, metric_value in metrics.items():
                point = point.field(metric_name, metric_value)
            
            try:
                self.write_api.write(bucket=self.bucket, record=point)
                print(f"  ✓ Sent {component_name} component metrics")
            except Exception as e:
                print(f"  ❌ Error sending {component_name} metrics: {e}")
    
    def close(self):
        """Close InfluxDB client"""
        self.client.close()
        print("✓ Grafana client closed")


def create_grafana_dashboard_json(bucket: str = "power_demand") -> dict:
    """
    Generate Grafana dashboard JSON configuration
    
    Returns:
        Dashboard JSON configuration
    """
    dashboard = {
        "dashboard": {
            "title": "Power Demand Forecasting",
            "tags": ["power", "forecast", "ml"],
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "24-Hour Forecast vs Actual",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 0},
                    "targets": [
                        {
                            "query": f'from(bucket: "{bucket}") |> range(start: -7d) |> filter(fn: (r) => r._measurement == "power_demand_forecast" and r._field == "forecast")',
                            "refId": "A"
                        },
                        {
                            "query": f'from(bucket: "{bucket}") |> range(start: -7d) |> filter(fn: (r) => r._measurement == "power_demand_actual" and r._field == "actual")',
                            "refId": "B"
                        }
                    ]
                },
                {
                    "id": 2,
                    "title": "Forecast Components",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
                    "targets": [
                        {
                            "query": f'from(bucket: "{bucket}") |> range(start: -7d) |> filter(fn: (r) => r._measurement == "power_demand_forecast")',
                            "refId": "A"
                        }
                    ]
                },
                {
                    "id": 3,
                    "title": "Overall Metrics",
                    "type": "stat",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                    "targets": [
                        {
                            "query": f'from(bucket: "{bucket}") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "forecast_metrics")',
                            "refId": "A"
                        }
                    ]
                },
                {
                    "id": 4,
                    "title": "Horizon-Specific MAE",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                    "targets": [
                        {
                            "query": f'from(bucket: "{bucket}") |> range(start: -7d) |> filter(fn: (r) => r._measurement == "horizon_metrics" and r._field == "mae")',
                            "refId": "A"
                        }
                    ]
                },
                {
                    "id": 5,
                    "title": "Component Statistics",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 24},
                    "targets": [
                        {
                            "query": f'from(bucket: "{bucket}") |> range(start: -7d) |> filter(fn: (r) => r._measurement == "component_metrics")',
                            "refId": "A"
                        }
                    ]
                }
            ]
        },
        "overwrite": True
    }
    
    return dashboard


if __name__ == "__main__":
    # Example usage
    client = GrafanaClient()
    
    # Example: Send forecast
    # forecast_df = pd.DataFrame({
    #     'timestamp': pd.date_range('2024-01-01', periods=24, freq='H'),
    #     'trend': np.random.randn(24) * 100 + 50000,
    #     'seasonality': np.random.randn(24) * 500,
    #     'residual': np.random.randn(24) * 100,
    #     'forecast': np.random.randn(24) * 100 + 50000
    # })
    # client.send_forecast(forecast_df)
    
    # Example: Calculate and send metrics
    # y_true = np.random.randn(100) * 100 + 50000
    # y_pred = y_true + np.random.randn(100) * 50
    # metrics = client.calculate_metrics(y_true, y_pred)
    # client.send_metrics(metrics, horizon=24)
    
    client.close()

