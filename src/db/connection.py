"""
Database connection management
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Generator
import pandas as pd
from src.utils.config import settings


def get_db_connection():
    """Create a database connection"""
    return psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        database=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD
    )


@contextmanager
def get_db_cursor(dict_cursor: bool = False) -> Generator:
    """Context manager for database cursor"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor) if dict_cursor else conn.cursor()
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def execute_query(query: str, params: tuple = None, fetch: bool = False):
    """Execute a SQL query"""
    with get_db_cursor() as cursor:
        cursor.execute(query, params)
        if fetch:
            return cursor.fetchall()


def read_sql_to_dataframe(query: str, params: tuple = None) -> pd.DataFrame:
    """Read SQL query results into a pandas DataFrame"""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()


def insert_dataframe(df: pd.DataFrame, table_name: str):
    """Insert a pandas DataFrame into a database table"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Create insert statement
        columns = ', '.join(df.columns)
        placeholders = ', '.join(['%s'] * len(df.columns))
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        
        # Execute batch insert
        for _, row in df.iterrows():
            cursor.execute(insert_query, tuple(row))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()



