"""Sliding-window per-key rate limiting with 429 Retry-After handling."""

from __future__ import annotations

import collections
import logging
import time
from threading import Lock

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from privatedocs.api.metrics import RATE_LIMITED_TOTAL
from privatedocs.config import get_settings

logger = logging.getLogger(__name__)
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class RateLimiter:
    """Thread-safe in-memory sliding window rate limiter."""

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60.0
        self._requests: dict[str, collections.deque[float]] = collections.defaultdict(collections.deque)
        self._lock = Lock()

    def check_rate_limit(self, key: str) -> None:
        """Check if request is within rate limit; raises HTTP 429 if exceeded."""
        now = time.monotonic()
        with self._lock:
            timestamps = self._requests[key]
            # Prune timestamps older than window
            cutoff = now - self.window_seconds
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()

            if len(timestamps) >= self.requests_per_minute:
                # Oldest timestamp determines when a slot opens up
                oldest = timestamps[0]
                retry_after = max(1, int(oldest + self.window_seconds - now))
                RATE_LIMITED_TOTAL.inc()
                logger.warning("Rate limit exceeded for key hash snippet '%s...'. Retry-After: %ds", key[:8], retry_after)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please throttle requests.",
                    headers={"Retry-After": str(retry_after)},
                )

            timestamps.append(now)


_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        settings = get_settings()
        _rate_limiter = RateLimiter(requests_per_minute=settings.RATE_LIMIT_PER_MINUTE)
    return _rate_limiter


async def check_rate_limit(
    request: Request,
    api_key: str | None = Security(API_KEY_HEADER),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    """FastAPI dependency checking rate limit for the incoming API key or client IP."""
    key = api_key or (request.client.host if request.client else "unknown_client")
    limiter.check_rate_limit(key)
