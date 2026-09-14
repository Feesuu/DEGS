from __future__ import annotations

from urllib.parse import urlparse


def validate_service_url(base_url: str) -> None:
    """Validate only the URL shape required by the HTTP client."""
    if base_url.startswith("mock://"):
        return
    parsed = urlparse(base_url)
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        return
    raise ValueError(f"unsupported service URL: {base_url!r}")
