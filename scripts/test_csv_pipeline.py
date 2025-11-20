"""
Test the CSV-based data pipeline
CSV 기반 데이터 파이프라인을 테스트하는 스크립트
"""
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.fetch_csv_data import (
    fetch_demand_from_csv,
    fetch_weather_from_csv,
    get_csv_date_range
)


def test_csv_data_reading():
    """Test reading data from CSV"""
    print("=" * 60)
    print("TESTING CSV DATA PIPELINE")
    print("=" * 60)
    
    # Check CSV date range
    print("\n1. Checking CSV date range...")
    date_range = get_csv_date_range()
    if date_range:
        print(f"   Min date: {date_range['min_date']}")
        print(f"   Max date: {date_range['max_date']}")
        print(f"   Total rows: {date_range['total_rows']:,}")
    else:
        print("   ❌ Failed to read CSV")
        return
    
    # Test demand data fetch
    print("\n2. Testing demand data fetch...")
    start_time = datetime(2019, 1, 1, 0, 0, 0)
    end_time = datetime(2019, 1, 2, 0, 0, 0)
    
    demand_df = fetch_demand_from_csv(
        start_time=start_time,
        end_time=end_time
    )
    
    print(f"   Fetched {len(demand_df)} rows")
    if not demand_df.empty:
        print(f"   Date range: {demand_df['timestamp'].min()} to {demand_df['timestamp'].max()}")
        print(f"   Avg demand: {demand_df['demand_value'].mean():.2f} MW")
        print("\n   Sample data:")
        print(demand_df.head())
    
    # Test weather data fetch
    print("\n3. Testing weather data fetch...")
    weather_df = fetch_weather_from_csv(
        start_time=start_time,
        end_time=end_time
    )
    
    print(f"   Fetched {len(weather_df)} rows")
    if not weather_df.empty:
        print(f"   Avg temperature: {weather_df['temperature'].mean():.2f}°C")
        print(f"   Avg humidity: {weather_df['humidity'].mean():.2f}%")
        print("\n   Sample data:")
        print(weather_df.head())
    
    print("\n" + "=" * 60)
    print("✅ CSV DATA PIPELINE TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    test_csv_data_reading()

