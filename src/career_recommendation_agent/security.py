from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time


DEFAULT_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"pbkdf2_sha256${base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, salt_raw, digest_raw = password_hash.split("$", 2)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_raw.encode())
        expected = base64.urlsafe_b64decode(digest_raw.encode())
    except ValueError:
        return False

    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return hmac.compare_digest(actual, expected)


def create_access_token(user_id: int, ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS) -> str:
    payload = {
        "sub": user_id,
        "exp": int(time.time()) + ttl_seconds,
    }
    payload_raw = _b64(json.dumps(payload, separators=(",", ":")).encode())
    signature = _sign(payload_raw)
    return f"{payload_raw}.{signature}"


def verify_access_token(token: str) -> int | None:
    try:
        payload_raw, signature = token.split(".", 1)
    except ValueError:
        return None

    if not hmac.compare_digest(_sign(payload_raw), signature):
        return None

    try:
        payload = json.loads(base64.urlsafe_b64decode(_pad(payload_raw)).decode("utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None

    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    user_id = payload.get("sub")
    return int(user_id) if isinstance(user_id, int) else None


def _sign(payload_raw: str) -> str:
    secret = os.getenv("APP_SECRET_KEY", "dev-only-change-this-secret")
    digest = hmac.new(secret.encode("utf-8"), payload_raw.encode("utf-8"), hashlib.sha256).digest()
    return _b64(digest)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("utf-8").rstrip("=")


def _pad(value: str) -> bytes:
    return (value + "=" * (-len(value) % 4)).encode("utf-8")

