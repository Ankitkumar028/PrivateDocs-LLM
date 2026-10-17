"""API-key authentication with SHA-256 hashing and SQLite persistence."""

from __future__ import annotations

import datetime
import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from privatedocs.config import get_settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def hash_key(plain_key: str) -> str:
    """Hash API key using SHA-256 for secure local persistence."""
    return hashlib.sha256(plain_key.encode("utf-8")).hexdigest()


class AuthDB:
    """Manages API key storage, verification, and role-based permissions in SQLite."""

    def __init__(self, db_path: str = "data/auth_keys.db") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        import threading
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def add_key(self, plain_key: str, role: str, description: str = "") -> None:
        """Register or update an API key with given role."""
        k_hash = hash_key(plain_key)
        now = datetime.datetime.now(datetime.UTC).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO api_keys (key_hash, role, description, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (k_hash, role, description, now),
            )
            self._conn.commit()

    def create_api_key(self, role: str = "user", description: str = "") -> str:
        """Generate and store a new random hex API key."""
        import secrets

        plain_key = f"pd_{role}_{secrets.token_hex(16)}"
        self.add_key(plain_key, role, description)
        return plain_key

    def verify_api_key(self, plain_key: str | None) -> str | None:
        """Verify API key and return associated role ('admin' or 'user'), or None if invalid."""
        if not plain_key:
            return None
        k_hash = hash_key(plain_key)
        with self._lock:
            cursor = self._conn.execute("SELECT role FROM api_keys WHERE key_hash = ?", (k_hash,))
            row = cursor.fetchone()
            if row:
                return row[0]
        return None


# Global AuthDB instance
_auth_db: AuthDB | None = None


def get_auth_db() -> AuthDB:
    global _auth_db
    if _auth_db is None:
        settings = get_settings()
        _auth_db = AuthDB(db_path=settings.API_KEYS_DB_PATH)
        # Seed default admin and user keys from settings
        if settings.ADMIN_API_KEY:
            _auth_db.add_key(settings.ADMIN_API_KEY, "admin", "Default Admin Key")
        if settings.USER_API_KEY:
            _auth_db.add_key(settings.USER_API_KEY, "user", "Default User Key")
    return _auth_db


async def get_current_user_role(
    api_key: str | None = Security(API_KEY_HEADER),
    auth_db: AuthDB = Depends(get_auth_db),
) -> str:
    """Dependency that authenticates any valid user or admin key."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing 'X-API-Key' header",
        )
    role = auth_db.verify_api_key(api_key)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key",
        )
    return role


async def require_admin_role(
    role: Annotated[str, Depends(get_current_user_role)],
) -> str:
    """Dependency enforcing that caller has 'admin' privileges."""
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this endpoint",
        )
    return role
