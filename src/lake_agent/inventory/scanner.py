from __future__ import annotations

from typing import Iterator

from lake_agent.domain.models import DiscoveredObject
from lake_agent.storage.base import ObjectStore


class ObjectScanner:
    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    def scan(self, bucket: str, prefix: str = "") -> Iterator[DiscoveredObject]:
        yield from self._store.list_objects(bucket, prefix)

    def stat(self, obj: DiscoveredObject) -> DiscoveredObject:
        return self._store.stat_object(obj.locator)
