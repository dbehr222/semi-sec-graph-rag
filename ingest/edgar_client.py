"""Polite HTTP client for SEC EDGAR: identity header, throttle, retries, disk cache.

Identity comes from the EDGAR_IDENTITY env var (loaded from .env), e.g.
    EDGAR_IDENTITY="Jane Doe jane@example.com"
The SEC requires a name + contact email in the User-Agent of every request.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import threading
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = ROOT / "data" / "cache" / "http"

MAX_REQUESTS_PER_SECOND = 8  # SEC's published limit is 10
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = 5

# Archive documents never change once filed; index endpoints (submissions JSON) do.
IMMUTABLE_PREFIX = "https://www.sec.gov/Archives/"
MUTABLE_TTL_SECONDS = 24 * 3600


class MissingIdentityError(RuntimeError):
    pass


def load_identity() -> str:
    load_dotenv()
    identity = (os.environ.get("EDGAR_IDENTITY") or "").strip()
    if not identity:
        raise MissingIdentityError(
            "EDGAR_IDENTITY is not set. The SEC requires a name and contact email "
            "in the User-Agent. Add a line like\n"
            '    EDGAR_IDENTITY="Your Name you@example.com"\n'
            "to .env (which is gitignored) or export it in your shell."
        )
    return identity


class _Throttle:
    """Spaces requests at least 1/rate seconds apart (thread-safe)."""

    def __init__(self, rate: float):
        self.interval = 1.0 / rate
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = self._next
            self._next = now + self.interval


class EdgarClient:
    def __init__(
        self,
        identity: str | None = None,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
        rate: float = MAX_REQUESTS_PER_SECOND,
    ):
        self.identity = identity or load_identity()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._throttle = _Throttle(rate)
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": self.identity, "Accept-Encoding": "gzip, deflate"}
        )
        self.network_requests = 0  # for tests / diagnostics

    # -- cache ---------------------------------------------------------------
    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / hashlib.sha1(url.encode()).hexdigest()

    def _cached(self, url: str) -> bytes | None:
        path = self._cache_path(url)
        if not path or not path.exists():
            return None
        if not url.startswith(IMMUTABLE_PREFIX):
            if time.time() - path.stat().st_mtime > MUTABLE_TTL_SECONDS:
                return None
        return path.read_bytes()

    def _store(self, url: str, content: bytes) -> None:
        path = self._cache_path(url)
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(content)
        tmp.replace(path)

    # -- fetching ------------------------------------------------------------
    def get_bytes(self, url: str) -> bytes:
        cached = self._cached(url)
        if cached is not None:
            return cached

        for attempt in range(MAX_RETRIES + 1):
            self._throttle.wait()
            self.network_requests += 1
            try:
                resp = self._session.get(url, timeout=30)
            except requests.RequestException as exc:
                if attempt == MAX_RETRIES:
                    raise
                log.warning("network error on %s (%s); retrying", url, exc)
            else:
                if resp.status_code == 200:
                    self._store(url, resp.content)
                    return resp.content
                if resp.status_code not in RETRY_STATUSES or attempt == MAX_RETRIES:
                    resp.raise_for_status()
                log.warning("HTTP %s on %s; retrying", resp.status_code, url)
            # exponential backoff with jitter: ~1, 2, 4, 8, 16 s
            time.sleep(2**attempt + random.uniform(0, 0.5))
        raise RuntimeError("unreachable")

    def get_json(self, url: str) -> dict:
        import json

        return json.loads(self.get_bytes(url))
