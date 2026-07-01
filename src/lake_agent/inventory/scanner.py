from __future__ import annotations

from typing import Iterator

from lake_agent.domain.models import FileMetadata
from lake_agent.storage.base import ObjectStore


class ObjectScanner:
    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    def scan(self, prefix: str = "") -> Iterator[FileMetadata]:
        yield from self._store.list_objects(prefix)

    def stat(self, obj: FileMetadata) -> FileMetadata:
        return self._store.stat_object(obj)
