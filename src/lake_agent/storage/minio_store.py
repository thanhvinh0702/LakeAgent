from __future__ import annotations

from datetime import datetime
from typing import Any, BinaryIO, Iterator

from lake_agent.domain.models import DiscoveredObject, ObjectLocator


class MinioObjectStore:
    """MinIO implementation of the object-store interface.

    The SDK import is delayed so the domain and unit tests do not require the
    MinIO package to be installed.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        *,
        secure: bool = False,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from minio import Minio
            except ImportError as exc:  # pragma: no cover - integration guard
                raise RuntimeError(
                    "MinIO support requires the 'minio' package. "
                    "Install the project dependencies first."
                ) from exc
            client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
            )
        self._client = client

    def list_objects(
        self,
        bucket: str,
        prefix: str = "",
    ) -> Iterator[DiscoveredObject]:
        for item in self._client.list_objects(bucket, prefix=prefix, recursive=True):
            object_key = getattr(item, "object_name", None)
            if not object_key or object_key.endswith("/"):
                continue
            locator = ObjectLocator(
                bucket=bucket,
                object_key=object_key,
                version_id=getattr(item, "version_id", None),
            )
            yield DiscoveredObject(
                locator=locator,
                etag=_clean_etag(getattr(item, "etag", None)),
                size_bytes=int(getattr(item, "size", 0) or 0),
                last_modified=_as_datetime(getattr(item, "last_modified", None)),
            )

    def stat_object(self, locator: ObjectLocator) -> DiscoveredObject:
        kwargs: dict[str, str] = {}
        if locator.version_id:
            kwargs["version_id"] = locator.version_id
        stat = self._client.stat_object(
            locator.bucket,
            locator.object_key,
            **kwargs,
        )
        metadata = {
            str(key): str(value)
            for key, value in (getattr(stat, "metadata", None) or {}).items()
        }
        return DiscoveredObject(
            locator=locator,
            etag=_clean_etag(getattr(stat, "etag", None)),
            size_bytes=int(getattr(stat, "size", 0) or 0),
            last_modified=_as_datetime(getattr(stat, "last_modified", None)),
            declared_content_type=getattr(stat, "content_type", None),
            user_metadata=metadata,
        )

    def read_range(
        self,
        locator: ObjectLocator,
        offset: int,
        length: int,
    ) -> bytes:
        if offset < 0 or length < 0:
            raise ValueError("offset and length must be non-negative")
        kwargs: dict[str, str | int] = {"offset": offset, "length": length}
        if locator.version_id:
            kwargs["version_id"] = locator.version_id
        response = self._client.get_object(
            locator.bucket,
            locator.object_key,
            **kwargs,
        )
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def stream_object(self, locator: ObjectLocator) -> BinaryIO:
        kwargs: dict[str, str] = {}
        if locator.version_id:
            kwargs["version_id"] = locator.version_id
        return self._client.get_object(
            locator.bucket,
            locator.object_key,
            **kwargs,
        )


def _clean_etag(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value).strip('"')


def _as_datetime(value: object | None) -> datetime | None:
    return value if isinstance(value, datetime) else None
