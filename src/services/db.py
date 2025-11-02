"""Postgres connection helper.

Provides a small wrapper around psycopg2 to create connections from a DSN
string or an environment variable `DATABASE_DSN`.

Usage:
  from src.services.db import get_connection
  dsn = 'host=127.0.0.1 port=5435 dbname=saleco_dw user=postgres password=xxxxxxx connect_timeout=10 sslmode=prefer'
  with get_connection(dsn) as conn:
      with conn.cursor() as cur:
          cur.execute('SELECT 1')
          print(cur.fetchone())
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg2
from psycopg2 import OperationalError


def connect_dsn(dsn: Optional[str] = None):
    """Return a new psycopg2 connection created from a DSN string.

    If `dsn` is not provided the function will look for the `DATABASE_DSN`
    environment variable.
    """
    if not dsn:
        dsn = os.getenv("DATABASE_DSN")
        if not dsn:
            raise ValueError(
                "No DSN provided. Set DATABASE_DSN env var or pass dsn argument."
            )

    try:
        # psycopg2 accepts the DSN string directly
        conn = psycopg2.connect(dsn)
        return conn
    except OperationalError:
        # Surface a clear error to the caller
        raise


@contextmanager
def get_connection(dsn: Optional[str] = None) -> Iterator[psycopg2.extensions.connection]:
    """Context manager that yields a psycopg2 connection and ensures it's closed.

    Example:
        with get_connection(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                print(cur.fetchone())
    """
    conn = connect_dsn(dsn)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            # Best-effort close; don't mask earlier exceptions
            pass
