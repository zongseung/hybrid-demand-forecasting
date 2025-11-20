"""
Test script to verify system components
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.connection import get_db_connection, read_sql_to_dataframe
from src.utils.config import settings
import requests


def test_database():
    """Test database connection"""
    print("Testing database connection...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()
        print(f"✅ Database connected: {version[0][:50]}...")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def test_tables():
    """Test database tables"""
    print("\nTesting database tables...")
    try:
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
        df = read_sql_to_dataframe(query)
        
        expected_tables = [
            'raw_demand', 'weather_history', 'calendar_info',
            'predictions', 'model_metrics', 'model_deployments', 'flow_runs'
        ]
        
        existing_tables = df['table_name'].tolist()
        
        for table in expected_tables:
            if table in existing_tables:
                print(f"✅ Table exists: {table}")
            else:
                print(f"❌ Table missing: {table}")
        
        return len(existing_tables) >= len(expected_tables)
        
    except Exception as e:
        print(f"❌ Table check failed: {e}")
        return False


def test_api():
    """Test FastAPI server"""
    print("\nTesting FastAPI server...")
    try:
        url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/health"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API server is running")
            print(f"   Status: {data.get('status')}")
            print(f"   Database: {data.get('database_connected')}")
            print(f"   Model: {data.get('model_loaded')}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API server not reachable: {e}")
        return False


def test_model_directory():
    """Test model directories"""
    print("\nTesting model directories...")
    try:
        prod_path = settings.PRODUCTION_MODEL_PATH
        temp_path = settings.TEMP_MODEL_PATH
        
        if os.path.exists(prod_path):
            print(f"✅ Production model directory exists: {prod_path}")
        else:
            print(f"⚠️  Production model directory not found: {prod_path}")
        
        if os.path.exists(temp_path):
            print(f"✅ Temp model directory exists: {temp_path}")
        else:
            print(f"⚠️  Temp model directory not found: {temp_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model directory check failed: {e}")
        return False


def test_imports():
    """Test Python imports"""
    print("\nTesting Python imports...")
    try:
        from src.models.hybrid_forecaster import HybridForecaster
        print("✅ HybridForecaster imported")
        
        from src.data.fetch_demand import fetch_demand_data
        print("✅ Data fetchers imported")
        
        from flows.data_ingestion import hourly_data_ingestion_flow
        print("✅ Prefect flows imported")
        
        from api.main import app
        print("✅ FastAPI app imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("DEMAND FORECASTING SYSTEM - COMPONENT TEST")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "Database": test_database(),
        "Tables": test_tables(),
        "Model Directories": test_model_directory(),
        "API": test_api()
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
    
    all_passed = all(results.values())
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())



