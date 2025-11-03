"""Small script to test DB connectivity.

It will read DSN from the environment variable `DATABASE_DSN`. If not set,
it falls back to the example DSN provided by the user.
"""

from __future__ import annotations

import os
from src.services.db import get_connection

DSN_FALLBACK = (
    "host=127.0.0.1 port=5435 dbname=saleco_dw user=postgres "
    "password=xxxxxxx connect_timeout=10 sslmode=prefer"
)


def main() -> None:
    dsn = os.getenv("DATABASE_DSN", DSN_FALLBACK)

    try:
        with get_connection(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                print("SELECT 1 ->", cur.fetchone())
    except Exception as exc:
        print("Connection failed:", exc)


if __name__ == "__main__":
    main()
