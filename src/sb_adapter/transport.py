from __future__ import annotations

from urllib.parse import urlparse


def validate_service_url(base_url: str) -> None:
    """Validate only the URL shape required by the HTTP client."""
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"unsupported service URL: {base_url!r}")
