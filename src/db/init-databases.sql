-- Initialize databases for Prefect and MLflow
-- This script is run automatically when PostgreSQL container starts

-- Create Prefect database
CREATE DATABASE prefect;

-- Create MLflow database
CREATE DATABASE mlflow;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE prefect TO postgres;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO postgres;

