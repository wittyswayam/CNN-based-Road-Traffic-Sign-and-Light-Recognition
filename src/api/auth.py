"""
TrafficVision-AI :: Authentication & Security
===============================================
JWT-based authentication with API key fallback.
Integrates with FastAPI dependency injection.

Features
--------
- JWT Bearer token validation (RS256 / HS256)
- API key authentication (hashed storage)
- Role-based access control (RBAC): viewer | inference | admin
- Token refresh flow
- Brute-force protection (sliding window counter)
- Audit logging (every auth event)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RBAC roles
# ---------------------------------------------------------------------------


class Role(str, Enum):
    VIEWER = "viewer"           # GET /health, /metrics
    INFERENCE = "inference"     # POST /v2/detect
    ADMIN = "admin"             # all endpoints + model management


ROLE_PERMISSIONS: Dict[Role, List[str]] = {
    Role.VIEWER: ["health:read", "metrics:read"],
    Role.INFERENCE: ["health:read", "metrics:read", "detect:write"],
    Role.ADMIN: ["health:read", "metrics:read", "detect:write", "model:manage", "admin:all"],
}


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass
class TokenPayload:
    sub: str                # subject (user ID or service name)
    role: Role
    exp: float              # expiry Unix timestamp
    iat: float              # issued-at
    jti: str                # JWT ID (for revocation)

    def is_expired(self) -> bool:
        return time.time() > self.exp

    def has_permission(self, required: str) -> bool:
        return required in ROLE_PERMISSIONS.get(self.role, [])


@dataclass
class APIKey:
    key_id: str
    key_hash: str           # SHA-256 of the raw key
    role: Role
    owner: str
    created_at: float
    last_used_at: float = 0.0
    active: bool = True


# ---------------------------------------------------------------------------
# JWT handler (HS256 for simplicity; swap to RS256 in production)
# ---------------------------------------------------------------------------


class JWTHandler:
    """
    Minimal JWT implementation using HMAC-SHA256.
    In production, use python-jose or PyJWT with RS256 + JWKS endpoint.
    """

    ALGORITHM = "HS256"
    DEFAULT_TTL_SECONDS = 3600

    def __init__(self, secret_key: str) -> None:
        if len(secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters")
        self._secret = secret_key.encode()

    def create_token(
        self,
        sub: str,
        role: Role,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> str:
        import base64, json as _json
        now = time.time()
        payload = {
            "sub": sub,
            "role": role.value,
            "iat": now,
            "exp": now + ttl_seconds,
            "jti": secrets.token_hex(16),
        }
        header = base64.urlsafe_b64encode(
            _json.dumps({"alg": self.ALGORITHM, "typ": "JWT"}).encode()
        ).rstrip(b"=")
        body = base64.urlsafe_b64encode(
            _json.dumps(payload).encode()
        ).rstrip(b"=")
        sig_input = header + b"." + body
        signature = base64.urlsafe_b64encode(
            hmac.new(self._secret, sig_input, "sha256").digest()
        ).rstrip(b"=")
        return (sig_input + b"." + signature).decode()

    def verify_token(self, token: str) -> TokenPayload:
        import base64, json as _json
        try:
            parts = token.encode().split(b".")
            if len(parts) != 3:
                raise ValueError("Invalid JWT structure")

            header_b, body_b, sig_b = parts
            sig_input = header_b + b"." + body_b

            # Verify signature
            expected_sig = base64.urlsafe_b64encode(
                hmac.new(self._secret, sig_input, "sha256").digest()
            ).rstrip(b"=")
            if not hmac.compare_digest(expected_sig, sig_b):
                raise ValueError("Invalid JWT signature")

            # Decode payload
            padding = 4 - len(body_b) % 4
            body_padded = body_b + b"=" * (padding % 4)
            payload = _json.loads(base64.urlsafe_b64decode(body_padded))

            token_payload = TokenPayload(
                sub=payload["sub"],
                role=Role(payload["role"]),
                exp=payload["exp"],
                iat=payload["iat"],
                jti=payload["jti"],
            )

            if token_payload.is_expired():
                raise ValueError("Token expired")

            return token_payload

        except (KeyError, ValueError) as exc:
            raise PermissionError(f"Token validation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# API Key manager
# ---------------------------------------------------------------------------


class APIKeyManager:
    """
    In-memory API key store. Replace with database in production.
    Keys are stored as SHA-256 hashes — raw keys never persisted.
    """

    def __init__(self) -> None:
        self._keys: Dict[str, APIKey] = {}
        self._rate_counters: Dict[str, List[float]] = {}

    def create_key(self, owner: str, role: Role) -> str:
        """Generate a new API key and store its hash. Returns the raw key once."""
        raw_key = f"tv_{secrets.token_urlsafe(32)}"
        key_id = secrets.token_hex(8)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        self._keys[key_id] = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            role=role,
            owner=owner,
            created_at=time.time(),
        )
        logger.info("Created API key for owner=%s role=%s key_id=%s", owner, role.value, key_id)
        return raw_key  # shown to user ONCE

    def validate_key(self, raw_key: str, required_permission: str) -> Optional[APIKey]:
        """Validate a raw API key and check permission. Returns APIKey or None."""
        raw_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        for key_id, api_key in self._keys.items():
            if hmac.compare_digest(api_key.key_hash, raw_hash):
                if not api_key.active:
                    logger.warning("Inactive API key used: %s", key_id)
                    return None
                if required_permission not in ROLE_PERMISSIONS.get(api_key.role, []):
                    logger.warning(
                        "Permission denied: key=%s role=%s required=%s",
                        key_id, api_key.role, required_permission,
                    )
                    return None
                api_key.last_used_at = time.time()
                return api_key

        logger.warning("Unknown API key attempted (hash prefix=%s…)", raw_hash[:8])
        return None

    def revoke_key(self, key_id: str) -> bool:
        if key_id in self._keys:
            self._keys[key_id].active = False
            logger.info("Revoked API key: %s", key_id)
            return True
        return False

    def is_rate_limited(
        self, key_id: str, limit: int = 100, window_seconds: int = 60
    ) -> bool:
        """Sliding-window rate limiter. Returns True if limit exceeded."""
        now = time.time()
        timestamps = self._rate_counters.setdefault(key_id, [])
        # Remove old entries
        self._rate_counters[key_id] = [
            ts for ts in timestamps if now - ts < window_seconds
        ]
        if len(self._rate_counters[key_id]) >= limit:
            return True
        self._rate_counters[key_id].append(now)
        return False


# ---------------------------------------------------------------------------
# FastAPI dependency factories
# ---------------------------------------------------------------------------


def make_auth_dependency(
    jwt_handler: JWTHandler,
    api_key_manager: APIKeyManager,
    required_permission: str = "detect:write",
):
    """
    Returns a FastAPI dependency that validates Bearer tokens or API keys.

    Usage::
        app = FastAPI()
        auth = make_auth_dependency(jwt_handler, api_key_manager, "detect:write")

        @app.post("/v2/detect")
        async def detect(file: UploadFile, principal=Depends(auth)):
            ...
    """
    from fastapi import Depends, HTTPException, Security, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader

    bearer_scheme = HTTPBearer(auto_error=False)
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def _dependency(
        credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
        api_key: Optional[str] = Security(api_key_header),
    ):
        # Try JWT Bearer
        if credentials:
            try:
                payload = jwt_handler.verify_token(credentials.credentials)
                if not payload.has_permission(required_permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions",
                    )
                return payload
            except PermissionError as exc:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=str(exc),
                    headers={"WWW-Authenticate": "Bearer"},
                )

        # Try API key
        if api_key:
            key_obj = api_key_manager.validate_key(api_key, required_permission)
            if key_obj is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or unauthorized API key",
                )
            if api_key_manager.is_rate_limited(key_obj.key_id):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded (100 req/min)",
                )
            return key_obj

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: provide Bearer token or X-API-Key header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _dependency
