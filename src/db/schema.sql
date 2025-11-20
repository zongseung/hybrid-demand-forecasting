-- Database schema for demand forecasting system
-- PostgreSQL with TimescaleDB extension

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Raw demand data table
CREATE TABLE IF NOT EXISTS raw_demand (
    timestamp TIMESTAMPTZ NOT NULL,
    demand_value FLOAT NOT NULL,
    hm FLOAT,
    ta FLOAT,
    holiday_name VARCHAR(200),
    weekday SMALLINT,
    weekend SMALLINT,
    spring SMALLINT,
    summer SMALLINT,
    autoum SMALLINT,
    winter SMALLINT,
    is_holiday_dummies SMALLINT,
    location VARCHAR(100),
    source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, location)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('raw_demand', 'timestamp', if_not_exists => TRUE);

-- Weather history table (optional, not used in simplified model)
-- CREATE TABLE IF NOT EXISTS weather_history (
--     timestamp TIMESTAMPTZ NOT NULL,
--     location VARCHAR(100),
--     temperature FLOAT,
--     humidity FLOAT,
--     wind_speed FLOAT,
--     precipitation FLOAT,
--     created_at TIMESTAMPTZ DEFAULT NOW(),
--     PRIMARY KEY (timestamp, location)
-- );
-- SELECT create_hypertable('weather_history', 'timestamp', if_not_exists => TRUE);

-- Calendar information table
CREATE TABLE IF NOT EXISTS calendar_info (
    date DATE NOT NULL PRIMARY KEY,
    is_holiday BOOLEAN DEFAULT FALSE,
    day_of_week INTEGER,
    week_of_year INTEGER,
    month INTEGER,
    year INTEGER,
    holiday_name VARCHAR(200)
);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    forecast FLOAT NOT NULL,
    trend_component FLOAT,
    seasonal_component FLOAT,
    lstm_component FLOAT,
    model_version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable('predictions', 'timestamp', if_not_exists => TRUE);

-- Model metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    run_timestamp TIMESTAMPTZ NOT NULL,
    model_version VARCHAR(50),
    mape FLOAT,
    rmse FLOAT,
    mae FLOAT,
    r2 FLOAT,
    validation_start TIMESTAMPTZ,
    validation_end TIMESTAMPTZ,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model deployment history
CREATE TABLE IF NOT EXISTS model_deployments (
    id SERIAL PRIMARY KEY,
    deployed_at TIMESTAMPTZ DEFAULT NOW(),
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50),
    model_path VARCHAR(500),
    deployed_by VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active',
    notes TEXT
);

-- Prefect flow runs tracking (optional, for custom monitoring)
CREATE TABLE IF NOT EXISTS flow_runs (
    id SERIAL PRIMARY KEY,
    flow_name VARCHAR(200) NOT NULL,
    run_id VARCHAR(100) UNIQUE,
    status VARCHAR(50),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- CSV data load state tracking
CREATE TABLE IF NOT EXISTS data_load_state (
    load_date DATE PRIMARY KEY,
    loaded_at TIMESTAMPTZ DEFAULT NOW(),
    row_count INTEGER,
    source_file VARCHAR(500),
    status VARCHAR(50) DEFAULT 'completed',
    notes TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_raw_demand_timestamp ON raw_demand (timestamp DESC);
-- CREATE INDEX IF NOT EXISTS idx_weather_timestamp ON weather_history (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_version ON predictions (model_version);
CREATE INDEX IF NOT EXISTS idx_model_metrics_version ON model_metrics (model_version);
CREATE INDEX IF NOT EXISTS idx_flow_runs_status ON flow_runs (status);

-- Views for common queries
CREATE OR REPLACE VIEW latest_predictions AS
SELECT 
    timestamp,
    forecast,
    trend_component,
    seasonal_component,
    lstm_component,
    model_version,
    created_at
FROM predictions
WHERE created_at >= NOW() - INTERVAL '7 days'
ORDER BY timestamp DESC;

CREATE OR REPLACE VIEW model_performance_summary AS
SELECT 
    model_version,
    AVG(mape) as avg_mape,
    AVG(rmse) as avg_rmse,
    AVG(mae) as avg_mae,
    AVG(r2) as avg_r2,
    COUNT(*) as num_runs,
    MAX(run_timestamp) as last_run
FROM model_metrics
GROUP BY model_version
ORDER BY last_run DESC;



