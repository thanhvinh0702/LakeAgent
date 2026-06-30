from __future__ import annotations

from typing import BinaryIO, Iterator, Protocol

from lake_agent.domain.models import DiscoveredObject, ObjectLocator


class ObjectStore(Protocol):
    """Small interface used by inventory code and easily faked in tests."""

    def list_objects(
        self,
        bucket: str,
        prefix: str = "",
    ) -> Iterator[DiscoveredObject]: ...

    def stat_object(self, locator: ObjectLocator) -> DiscoveredObject: ...

    def read_range(
        self,
        locator: ObjectLocator,
        offset: int,
        length: int,
    ) -> bytes: ...

    def stream_object(self, locator: ObjectLocator) -> BinaryIO: ...
