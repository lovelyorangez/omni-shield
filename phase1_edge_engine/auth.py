"""
API key authentication for OMNI-SHIELD.

Setup:
  1. Generate a key:
       python3 -c "import secrets; print(secrets.token_hex(32))"
  2. Hash it:
       python3 -c "import hashlib; print(hashlib.sha256(b'YOUR_KEY').hexdigest())"
  3. Set the env variable (comma-separated for multiple keys):
       export OMNI_API_KEYS="hash1,hash2"

If OMNI_API_KEYS is unset the server runs in dev mode (all requests allowed).
"""

import hashlib
import os
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Load valid key hashes from environment.
# Strip whitespace so "hash1, hash2" (with spaces) works correctly.
VALID_KEY_HASHES: set[str] = set(
    k.strip()
    for k in os.environ.get("OMNI_API_KEYS", "").split(",")
    if k.strip()
)

# Dev mode when no keys are configured
AUTH_REQUIRED = bool(VALID_KEY_HASHES)


def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
    """FastAPI dependency — returns a redacted key string on success."""
    if not AUTH_REQUIRED:
        return "dev"
    if not api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    if key_hash not in VALID_KEY_HASHES:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key[:8] + "..."
