"""
Postgres connection manager for AlphaPulse.
Provides a singleton connection pool and context manager for transactions.
"""
import os
import threading
from contextlib import contextmanager
from typing import Optional

import psycopg2
import psycopg2.pool


_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()


def get_connection_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Get or create the Postgres connection pool."""
    global _pool

    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=2,
                    maxconn=10,
                    host=os.getenv("PG_HOST", "localhost"),
                    port=int(os.getenv("PG_PORT", "5433")),
                    dbname=os.getenv("PG_DATABASE", "alphapulse"),
                    user=os.getenv("PG_USER", "alphapulse"),
                    password=os.getenv("PG_PASSWORD", "alphapulse"),
                )
    return _pool


@contextmanager
def get_connection():
    """Get a connection from the pool. Use as context manager."""
    pool = get_connection_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def test_connection() -> bool:
    """Test database connectivity."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
    except Exception:
        return False
