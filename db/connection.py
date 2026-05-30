"""
Postgres connection manager for AlphaPulse.
Provides a singleton connection pool and context manager for transactions.
"""
import os
import re
import time
from contextlib import contextmanager

import psycopg2


def _runtime_schema() -> str:
    schema = os.getenv("PG_SCHEMA", "historical").strip() or "historical"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
        raise ValueError(f"Invalid PG_SCHEMA value: {schema!r}")
    return schema


def _connect_with_retry() -> psycopg2.extensions.connection:
    sslmode = os.getenv("PG_SSLMODE")
    conn_kwargs = dict(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5433")),
        dbname=os.getenv("PG_DATABASE", "alphapulse"),
        user=os.getenv("PG_USER", "alphapulse"),
        password=os.getenv("PG_PASSWORD", "alphapulse"),
        connect_timeout=int(os.getenv("PG_CONNECT_TIMEOUT", "5")),
        options=f"-c search_path={_runtime_schema()},public",
    )
    if sslmode:
        conn_kwargs["sslmode"] = sslmode
    last_error = None
    for attempt in range(3):
        try:
            return psycopg2.connect(**conn_kwargs)
        except psycopg2.OperationalError as exc:
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(1)
    raise last_error


@contextmanager
def get_connection():
    """Get a dedicated connection for the duration of the context."""
    conn = _connect_with_retry()
    try:
        yield conn
    finally:
        conn.close()


def test_connection() -> bool:
    """Test database connectivity."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
    except Exception:
        return False


def ensure_connection_or_exit() -> None:
    """Test database connectivity and exit with error if unavailable."""
    import sys
    if not test_connection():
        print("\n  FATAL: Postgres is not available.")
        print("  Run: docker-compose up -d")
        print("  Or check PG_HOST/PG_PORT/PG_DATABASE/PG_USER/PG_PASSWORD settings.")
        sys.exit(1)
