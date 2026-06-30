from __future__ import annotations

import hashlib

from lake_agent.domain.models import ObjectLocator
from lake_agent.storage.base import ObjectStore


class ObjectHasher:
    def __init__(self, store: ObjectStore, chunk_size: int = 8 * 1024 * 1024) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._store = store
        self._chunk_size = chunk_size

    def sha256(self, locator: ObjectLocator) -> str:
        digest = hashlib.sha256()
        response = self._store.stream_object(locator)
        try:
            while chunk := response.read(self._chunk_size):
                digest.update(chunk)
        finally:
            response.close()
            release_conn = getattr(response, "release_conn", None)
            if callable(release_conn):
                release_conn()
        return digest.hexdigest()
