from __future__ import annotations

from typing import BinaryIO, Iterator, Protocol

from lake_agent.domain.models import FileMetadata


class ObjectStore(Protocol):
    """Small interface used by inventory code and easily faked in tests."""

    def list_objects(
        self,
        prefix: str = "",
    ) -> Iterator[FileMetadata]: ...

    def stat_object(self, obj: FileMetadata) -> FileMetadata: ...

    def read_range(
        self,
        obj: FileMetadata,
        offset: int,
        length: int,
    ) -> bytes: ...

    def stream_object(self, obj: FileMetadata) -> BinaryIO: ...
