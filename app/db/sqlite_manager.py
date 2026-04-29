"""
SQLite Manager for Brahman 2.0.
Thread-safe, WAL-mode, production-grade persistence layer.
"""
import sqlite3
import os
from pathlib import Path
from typing import List, Dict, Optional


class SQLiteManager:
    """
    Manages the brahman_v2.db connection pool.
    Configured for high-throughput concurrent reads via WAL mode.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Resolve relative to this file's location for portability
            base = Path(__file__).resolve().parent.parent / "data"
            db_path = str(base / "brahman_v2.db")

        self.db_path = db_path
        self._ensure_db_exists()
        self._conn = self._create_connection()
        self._apply_pragmas()

    def _ensure_db_exists(self):
        """Verify the database file exists before connecting."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. "
                "Run ingestion_engine.py first to populate the Dhātupāṭha."
            )

    def _create_connection(self) -> sqlite3.Connection:
        """Create a thread-safe SQLite connection."""
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,  # Required for FastAPI async workers
            timeout=10
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _apply_pragmas(self):
        """Apply production pragmas for performance and safety."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
            cursor.execute("PRAGMA temp_store=MEMORY")
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "readonly database" in str(e):
                import logging
                logging.getLogger("brahman").warning("Database is read-only; skipping write-dependent pragmas.")
            else:
                raise

    def get_all_dhatus(self) -> List[Dict]:
        """Load all verbal roots from the database."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT id, root, gana, pada, meaning FROM dhatus")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_dhatu_by_root(self, root: str) -> Optional[Dict]:
        """Look up a single dhātu by its root string."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT id, root, gana, pada, meaning FROM dhatus WHERE root = ?",
            (root,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_dhatu_count(self) -> int:
        """Return total number of dhātus in the database."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM dhatus")
        return cursor.fetchone()["cnt"]

    def close(self):
        """Gracefully close the connection."""
        if self._conn:
            self._conn.close()
