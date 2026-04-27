"""
Brahman App — Centralized Authentication

Provides a reusable FastAPI dependency for API key validation.
Used by both verify.py and solnet.py to enforce consistent auth.

In production (BRAHMAN_ENV=production):
  - BRAHMAN_API_KEY must be set explicitly
  - Known default keys are rejected at startup
In development:
  - A temporary secure key is generated and printed
"""
import os
import sys
import secrets

from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

BRAHMAN_ENV = os.environ.get("BRAHMAN_ENV", "development")
API_KEY = os.environ.get("BRAHMAN_API_KEY")

if not API_KEY:
    if BRAHMAN_ENV == "production":
        print("\n✗ FATAL: BRAHMAN_API_KEY is not set.")
        print("  Production mode requires an explicit API key.")
        sys.exit(1)
    else:
        API_KEY = secrets.token_hex(32)
        print(f"⚠ [app] BRAHMAN_API_KEY not set (env={BRAHMAN_ENV}). Temporary key: {API_KEY}")

_DEFAULT_KEYS = {"brahman-dev-secret-key-123", "test", "dev", "password", "secret"}
if API_KEY in _DEFAULT_KEYS:
    if BRAHMAN_ENV == "production":
        print(f"\n✗ FATAL: API key is a known default. Not allowed in production.")
        sys.exit(1)
    else:
        print(f"⚠ [app] API key matches a known default. Do NOT use in production.")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(_api_key_header)):
    """FastAPI dependency — rejects requests without a valid API key."""
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )
