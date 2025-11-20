"""
Backfill historical data (2019-2020) from CSV to database
"""
import pandas as pd
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.connection import insert_dataframe, execute_query, read_sql_to_dataframe
from src.utils.config import settings


def backfill_from_csv(
    csv_path: str = "/mnt/nvme/tilting/power_demand_final.csv",
    start_year: int = 2019,
    end_year: int = 2020,
    location: str = "default"
):
    """
    Backfill historical data from CSV
    
    Args:
        csv_path: Path to CSV file
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        location: Location identifier
    """
    print("=" * 60)
    print("BACKFILLING HISTORICAL DATA FROM CSV")
    print("=" * 60)
    print(f"CSV Path: {csv_path}")
    print(f"Period: {start_year}-{end_year}")
    print(f"Location: {location}")
    print()
    
    # Read CSV
    print("Reading CSV file...")
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df):,}")
    
    # Rename columns to match database schema
    column_map = {
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
        "is_holiday_dummies": "is_holiday_dummies"
    }
    missing = [col for col in column_map if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df.rename(columns=column_map, inplace=True)
    
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    
    # Filter for target years
    print(f"\nFiltering data for {start_year}-{end_year}...")
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year + 1}-01-01"  # Exclusive
    
    filtered_df = df[
        (df["timestamp"] >= start_date) &
        (df["timestamp"] < end_date)
    ].copy()
    
    print(f"Filtered rows: {len(filtered_df):,}")
    
    if filtered_df.empty:
        print("❌ No data found for the specified period!")
        return
    
    # Prepare data for insertion
    print("\nPreparing data for insertion...")
    demand_df = filtered_df[
        [
            "timestamp",
            "demand_value",
            "hm",
            "ta",
            "holiday_name",
            "weekday",
            "weekend",
            "spring",
            "summer",
            "autoum",
            "winter",
            "is_holiday_dummies",
        ]
    ].copy()
    demand_df["location"] = location
    demand_df["source"] = "csv_backfill"
    
    # Check for existing data
    print("\nChecking for existing data...")
    check_query = """
    SELECT COUNT(*) as count, MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
    FROM raw_demand
    WHERE timestamp >= %s AND timestamp < %s AND location = %s
    """
    existing = read_sql_to_dataframe(
        check_query, 
        (start_date, end_date, location)
    )
    
    if not existing.empty and existing.iloc[0]["count"] > 0:
        print(f"⚠️  Found {existing.iloc[0]['count']} existing rows")
        print(f"   Range: {existing.iloc[0]['min_ts']} to {existing.iloc[0]['max_ts']}")
        response = input("\nDo you want to proceed? This will skip duplicates. (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Insert data in chunks
    print("\nInserting data into database...")
    chunk_size = 1000
    total_inserted = 0
    
    for i in range(0, len(demand_df), chunk_size):
        chunk = demand_df.iloc[i:i+chunk_size]
        try:
            insert_dataframe(chunk, "raw_demand")
            total_inserted += len(chunk)
            if (i + chunk_size) % 10000 == 0:
                print(f"   Inserted {total_inserted:,} / {len(demand_df):,} rows...")
        except Exception as e:
            print(f"   Error inserting chunk at index {i}: {e}")
            continue
    
    print(f"\n✅ Successfully inserted {total_inserted:,} rows")
    
    # Update data_load_state for each date
    print("\nUpdating data load state...")
    unique_dates = pd.to_datetime(filtered_df["timestamp"].dt.date).unique()
    
    for date in unique_dates:
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        rows_for_date = len(filtered_df[filtered_df["timestamp"].dt.date == pd.to_datetime(date).date()])
        
        execute_query(
            """
            INSERT INTO data_load_state (load_date, row_count, source_file, status)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (load_date) DO UPDATE SET
                row_count = EXCLUDED.row_count,
                loaded_at = NOW(),
                status = EXCLUDED.status
            """,
            (date_str, rows_for_date, csv_path, 'backfilled')
        )
    
    print(f"✅ Updated load state for {len(unique_dates)} dates")
    
    # Verify insertion
    print("\nVerifying insertion...")
    verify_query = """
    SELECT 
        COUNT(*) as total_rows,
        MIN(timestamp) as earliest,
        MAX(timestamp) as latest,
        AVG(demand_value) as avg_demand
    FROM raw_demand
    WHERE timestamp >= %s AND timestamp < %s AND location = %s
    """
    
    result = read_sql_to_dataframe(verify_query, (start_date, end_date, location))
    
    if not result.empty:
        print(f"Total rows in DB: {result.iloc[0]['total_rows']:,}")
        print(f"Date range: {result.iloc[0]['earliest']} to {result.iloc[0]['latest']}")
        print(f"Average demand: {result.iloc[0]['avg_demand']:.2f} MW")
    
    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    backfill_from_csv()

