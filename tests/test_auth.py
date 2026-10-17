"""Unit tests for SQLite API-key authentication and role-based permissions."""

import pytest

from privatedocs.api.auth import AuthDB, hash_key


@pytest.fixture
def auth_db():
    return AuthDB(db_path=":memory:")


def test_hash_key():
    h1 = hash_key("secret-key-1")
    h2 = hash_key("secret-key-1")
    h3 = hash_key("secret-key-2")

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # SHA-256 hex string


def test_create_and_verify_api_key(auth_db):
    user_key = auth_db.create_api_key(role="user", description="Test User")
    admin_key = auth_db.create_api_key(role="admin", description="Test Admin")

    assert auth_db.verify_api_key(user_key) == "user"
    assert auth_db.verify_api_key(admin_key) == "admin"
    assert auth_db.verify_api_key("nonexistent-key") is None
    assert auth_db.verify_api_key("") is None
    assert auth_db.verify_api_key(None) is None


def test_add_key_override(auth_db):
    key = "my-custom-key"
    auth_db.add_key(key, role="user", description="Initial")
    assert auth_db.verify_api_key(key) == "user"

    # Promote to admin
    auth_db.add_key(key, role="admin", description="Promoted")
    assert auth_db.verify_api_key(key) == "admin"
