"""Authentication helpers for public HTTP API endpoints."""

import secrets
from dataclasses import dataclass

from fastapi import Header, HTTPException

from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

ALLOWED_AUTH_CHANNELS = {"api", "feishu"}


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    channel: str


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


async def get_api_auth_context(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_channel: str | None = Header(default=None, alias="X-Channel"),
) -> AuthContext | None:
    """Authenticate a public API endpoint and build a trusted identity context.

    If API_KEY is not configured, keep local/test compatibility by returning
    None so the endpoint can use its legacy identity behavior.
    """
    configured_api_key = settings.api_key
    if not configured_api_key:
        logger.warning("API_KEY is not configured; using local compatibility identity")
        return None

    provided_api_key = _extract_bearer_token(authorization) or (x_api_key.strip() if x_api_key else None)
    if not provided_api_key or not secrets.compare_digest(provided_api_key, configured_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    trusted_user_id = x_user_id.strip() if x_user_id else ""
    if not trusted_user_id:
        raise HTTPException(status_code=401, detail="Missing trusted user identity")

    trusted_channel = (x_channel or "api").strip().lower()
    if trusted_channel not in ALLOWED_AUTH_CHANNELS:
        raise HTTPException(status_code=400, detail="X-Channel must be api or feishu")

    return AuthContext(user_id=trusted_user_id, channel=trusted_channel)


get_ask_auth_context = get_api_auth_context
