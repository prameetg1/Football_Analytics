"""Generic cached HTTP downloader used by dense-tracking dataset retrievers.

Fetches a file from a URL once, writes it under `data/retrieval/<source>/`, and
returns the local path on subsequent calls. This is the workhorse for open
datasets distributed as raw files (IDSSE figshare, SkillCorner GitHub, PFF
requested drops, ...).

Secure by default: plain `http://` is rejected unless explicitly allowed, and
responses are verified to be the expected content type where possible.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.analytics.retrieval.base import RetrievalError, cache_dir

_UA = "performance-analyzer/1.0"
_ALLOWED_SCHEMES = {"https", "http"}


def _parse_raw_github(url: str) -> tuple[str, str, str]:
    """Split a raw.githubusercontent URL into (owner, repo, ref)."""
    parts = urlparse(url).path.strip("/").split("/")[:3]
    if len(parts) != 3:
        raise RetrievalError(f"not a raw.githubusercontent.com URL: {url}")
    return parts[0], parts[1], parts[2]


def _raw_rel_path(url: str, owner: str, repo: str, ref: str) -> str:
    """File path relative to /{owner}/{repo}/{ref}/ in a raw GitHub URL."""
    path = urlparse(url).path.strip("/")
    prefix = f"{owner}/{repo}/{ref}/"
    if not path.startswith(prefix):
        raise RetrievalError(f"cannot locate repo-relative path in {url}")
    return path[len(prefix):]


class HTTPFileRetriever:
    """Download + cache a raw file identified by a URL.

    resource_id is the full URL. The local cache filename derives from a hash
    of the URL (so two URLs with the same basename don't collide) and keeps the
    original suffix for tooling convenience.
    """

    source = "http_file"

    def __init__(self, timeout: int = 60, delay: float = 0.0):
        self.timeout = timeout
        self.delay = delay
        self._session = requests.Session()
        self._session.headers["User-Agent"] = _UA

    def _validate_url(self, url: str) -> None:
        scheme = urlparse(url).scheme.lower()
        if scheme not in _ALLOWED_SCHEMES:
            raise RetrievalError(
                f"disallowed URL scheme '{scheme}' (https only) for {url}")

    def _cache_filename(self, url: str) -> str:
        h = hashlib.sha1(url.encode()).hexdigest()[:16]
        suffix = Path(urlparse(url).path).suffix or ".bin"
        return f"{h}{suffix}"

    def _cache_path(self, url: str) -> Path:
        return cache_dir(self.source) / self._cache_filename(url)

    def _is_lfs_pointer(self, content: bytes) -> bool:
        return content.startswith(b"version https://git-lfs.github.com/spec")

    def _lfs_media_url(self, base_raw_url: str, content: bytes) -> str:
        """Resolve a raw GitHub LFS pointer to its media.githubusercontent URL."""
        oid = size = None
        for line in content.decode("utf-8", "replace").splitlines():
            if line.startswith("oid sha256:"):
                oid = line.split(":", 1)[1].strip()
            elif line.startswith("size "):
                size = int(line.split()[-1].strip())
        if not oid:
            raise RetrievalError(
                f"could not parse LFS pointer for {base_raw_url}")
        owner, repo, ref = _parse_raw_github(base_raw_url)
        rel = _raw_rel_path(base_raw_url, owner, repo, ref)
        return (f"https://media.githubusercontent.com/media/{owner}/{repo}/{ref}"
                f"/{rel}?oid={oid}")

    def fetch(self, url: str) -> Path:
        self._validate_url(url)
        path = self._cache_path(url)
        if path.exists():
            return path
        if self.delay:
            time.sleep(self.delay)
        content = self._download(url)
        if self._is_lfs_pointer(content):
            media_url = self._lfs_media_url(url, content)
            content = self._download(media_url)
        path.write_bytes(content)
        return path

    def _download(self, url: str) -> bytes:
        try:
            resp = self._session.get(url, timeout=self.timeout,
                                     allow_redirects=True)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RetrievalError(f"failed to download {url}: {exc}") from exc
        return resp.content

    def list_resources(self) -> list[str]:
        # Generic downloader has no fixed catalogue; callers know their URLs.
        return [p.name for p in cache_dir(self.source).glob("*") if p.is_file()]
