"""
Daily Data Ingestion Flow
Backfills one day of data from CSV to PostgreSQL
"""

from prefect import flow, task
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.connection import get_db_connection


# CSV column mapping
CSV_COLUMN_MAP = {
    "일시": "timestamp",
    "power demand(MW)": "demand_value",
    "hm": "hm",
    "ta": "ta",
    "holiday_name": "holiday_name",
    "weekday": "weekday",
    "weekend": "weekend",
    "spring": "spring",
    "summer": "summer",
    "autoum": "autoum",
    "winter": "winter",
    "is_holiday_dummies": "is_holiday_dummies",
}

# Data type definitions
FLOAT_COLUMNS = ['demand_value', 'hm', 'ta']
BOOL_COLUMNS = ['weekday', 'weekend', 'spring', 'summer', 'autoum', 'winter', 'is_holiday_dummies']


@task(name="Resolve Target Date", retries=1)
def resolve_target_date(target_date: Optional[str] = None) -> str:
    """
    Resolve the target date for backfill
    
    Args:
        target_date: Specific date (YYYY-MM-DD), or None for yesterday
        
    Returns:
        Target date string (YYYY-MM-DD)
    """
    if target_date is None:
        # Default: yesterday
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"\n📅 Target date: {target_date}")
    return target_date


@task(name="Extract Daily Slice", retries=2)
def extract_daily_slice(
    csv_path: str,
    target_date: str
) -> pd.DataFrame:
    """
    Extract one day of data from CSV
    
    Args:
        csv_path: Path to CSV file
        target_date: Date to extract (YYYY-MM-DD)
        
    Returns:
        DataFrame with one day of data
    """
    print(f"\n📂 Loading data from CSV: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Rename columns
    df.rename(columns=CSV_COLUMN_MAP, inplace=True)
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter for target date
    target_date_dt = pd.to_datetime(target_date)
    daily_df = df[df['timestamp'].dt.date == target_date_dt.date()].copy()
    
    if len(daily_df) == 0:
        raise ValueError(f"No data found for date {target_date}")
    
    # Convert data types
    for col in FLOAT_COLUMNS:
        if col in daily_df.columns:
            daily_df[col] = daily_df[col].astype(float)
    
    for col in BOOL_COLUMNS:
        if col in daily_df.columns:
            daily_df[col] = daily_df[col].astype(int)
    
    # Fill missing holiday names
    if 'holiday_name' in daily_df.columns:
        daily_df['holiday_name'].fillna('non-event', inplace=True)
    
    # Add metadata
    daily_df['location'] = 'South Korea'
    daily_df['source'] = 'CSV'
    
    print(f"✓ Extracted {len(daily_df)} samples for {target_date}")
    
    return daily_df


@task(name="Load to Database", retries=2)
def load_raw_demand(
    daily_df: pd.DataFrame,
    db_host: str = "postgres",
    db_port: int = 5432,
    db_name: str = "demand_forecasting",
    db_user: str = "postgres",
    db_password: str = "postgres"
):
    """
    Load daily data to raw_demand table
    
    Args:
        daily_df: DataFrame with daily data
    """
    print(f"\n💾 Loading {len(daily_df)} samples to database...")
    
    conn = get_db_connection(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )
    
    try:
        # Insert data (replace if exists)
        daily_df.to_sql(
            'raw_demand',
            conn,
            if_exists='append',
            index=False,
            method='multi'
        )
        
        print(f"✓ Successfully loaded {len(daily_df)} samples")
        
    finally:
        conn.close()


@flow(
    name="Daily CSV Backfill",
    description="Backfill one day of data from CSV to PostgreSQL"
)
def daily_data_ingestion_flow(
    csv_path: str = "/app/models/power_demand_final.csv",
    target_date: Optional[str] = None
):
    """
    Complete daily data ingestion flow
    
    Args:
        csv_path: Path to CSV file
        target_date: Date to backfill (YYYY-MM-DD), None for yesterday
    """
    print("\n" + "="*80)
    print("DAILY DATA INGESTION FLOW")
    print("="*80)
    print(f"Execution time: {datetime.now()}")
    print(f"CSV path: {csv_path}")
    print("="*80)
    
    # 1. Resolve target date
    date_str = resolve_target_date(target_date=target_date)
    
    # 2. Extract daily slice from CSV
    daily_df = extract_daily_slice(
        csv_path=csv_path,
        target_date=date_str
    )
    
    # 3. Load to database
    load_raw_demand(daily_df=daily_df)
    
    print("\n" + "="*80)
    print("✅ DAILY INGESTION FLOW COMPLETED")
    print("="*80)
    
    return {
        'target_date': date_str,
        'samples_loaded': len(daily_df)
    }


if __name__ == "__main__":
    # Run the flow locally
    daily_data_ingestion_flow()

