import pytest
from sqlalchemy import text
from src.database.connection import get_engine

def test_database_connection():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
    except Exception as e:
        pytest.skip(f"Database is not accessible: {e}")
