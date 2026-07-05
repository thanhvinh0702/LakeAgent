from __future__ import annotations

import mimetypes
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterator

from lake_agent.domain.models import FileMetadata


class LocalFileStore:
    def __init__(self, root_dir: str) -> None:
        root_path = Path(root_dir).expanduser().resolve()
        if not root_path.exists():
            raise ValueError(f"DATALAKE_DIR does not exist: {root_path}")
        if not root_path.is_dir():
            raise ValueError(f"DATALAKE_DIR is not a directory: {root_path}")
        self._root_dir = root_path

    def list_objects(self, prefix: str = "") -> Iterator[FileMetadata]:
        base_dir = self._resolve_prefix(prefix)
        if not base_dir.exists():
            return
        for path in sorted(base_dir.rglob("*")):
            if not path.is_file():
                continue
            yield self._metadata_for_path(path)

    def stat_object(self, obj: FileMetadata) -> FileMetadata:
        path = self._resolve_object_path(obj.object_key)
        return self._metadata_for_path(path)

    def read_range(
        self,
        obj: FileMetadata,
        offset: int,
        length: int,
    ) -> bytes:
        if offset < 0 or length < 0:
            raise ValueError("offset and length must be non-negative")
        path = self._resolve_object_path(obj.object_key)
        with path.open("rb") as file_obj:
            file_obj.seek(offset)
            return file_obj.read(length)

    def stream_object(self, obj: FileMetadata) -> BinaryIO:
        path = self._resolve_object_path(obj.object_key)
        return path.open("rb")

    def rename_object(self, obj: FileMetadata, new_object_key: str) -> FileMetadata:
        source_path = self._resolve_object_path(obj.object_key)
        destination_path = self._resolve_destination_path(new_object_key)
        if destination_path.exists():
            raise FileExistsError(destination_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.rename(destination_path)
        return replace(obj, object_key=new_object_key)

    def _resolve_prefix(self, prefix: str) -> Path:
        if not prefix:
            return self._root_dir
        path = (self._root_dir / prefix).resolve()
        if self._root_dir not in path.parents and path != self._root_dir:
            raise ValueError(f"Prefix escapes DATALAKE_DIR: {prefix}")
        return path

    def _resolve_object_path(self, object_key: str) -> Path:
        path = (self._root_dir / object_key).resolve()
        if self._root_dir not in path.parents and path != self._root_dir:
            raise ValueError(f"Object path escapes DATALAKE_DIR: {object_key}")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _resolve_destination_path(self, object_key: str) -> Path:
        path = (self._root_dir / object_key).resolve()
        if self._root_dir not in path.parents and path != self._root_dir:
            raise ValueError(f"Object path escapes DATALAKE_DIR: {object_key}")
        return path

    def _metadata_for_path(self, path: Path) -> FileMetadata:
        stat = path.stat()
        content_type, _ = mimetypes.guess_type(path.name)
        return FileMetadata(
            object_key=path.relative_to(self._root_dir).as_posix(),
            size_bytes=stat.st_size,
            last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            declared_content_type=content_type,
        )
