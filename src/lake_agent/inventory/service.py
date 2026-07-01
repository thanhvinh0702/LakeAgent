from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Mapping

from lake_agent.domain.enums import FileStatus
from lake_agent.domain.models import FileMetadata
from lake_agent.inventory.identifier import ObjectIdentifier
from lake_agent.inventory.scanner import ObjectScanner
from lake_agent.persistence.repositories import InventoryRepository


class InventoryService:
    def __init__(
        self,
        scanner: ObjectScanner,
        identifier: ObjectIdentifier,
        repository: InventoryRepository,
        *,
        stat_new_or_changed: bool = True,
    ) -> None:
        self._scanner = scanner
        self._identifier = identifier
        self._repository = repository
        self._stat_new_or_changed = stat_new_or_changed

    def run(self, prefix: str = "") -> dict[str, Any]:
        normalized_prefix = _normalize_prefix(prefix)
        scanned_at = datetime.now(timezone.utc)
        discovered_count = 0
        identified_count = 0
        unchanged_count = 0
        error_count = 0

        for listed_object in self._scanner.scan(normalized_prefix):
            discovered_count += 1
            previous = self._repository.find_object(listed_object.identity)

            if previous and _is_unchanged(previous, listed_object):
                self._repository.save(
                    replace(
                        listed_object,
                        status=FileStatus.IDENTIFIED,
                        is_present=True,
                    ),
                    scanned_at,
                )
                unchanged_count += 1
                continue

            current_object = listed_object
            try:
                if self._stat_new_or_changed:
                    current_object = self._scanner.stat(listed_object)

                current_object = self._identifier.identify(current_object)
                self._repository.save(
                    replace(
                        current_object,
                        status=FileStatus.IDENTIFIED,
                        is_present=True,
                    ),
                    scanned_at,
                )
                identified_count += 1
            except Exception as exc:
                error_count += 1
                self._repository.save(
                    replace(
                        current_object,
                        warnings=(str(exc),),
                        status=FileStatus.ERROR,
                        is_present=True,
                    ),
                    scanned_at,
                )

        self._repository.mark_missing(normalized_prefix, scanned_at)
        return {
            "prefix": normalized_prefix,
            "discovered_count": discovered_count,
            "identified_count": identified_count,
            "unchanged_count": unchanged_count,
            "error_count": error_count,
        }


def _is_unchanged(
    previous: Mapping[str, Any],
    current: FileMetadata,
) -> bool:
    if previous.get("status") != FileStatus.IDENTIFIED.value:
        return False
    if previous.get("size_bytes") != current.size_bytes:
        return False
    if previous.get("version_id") != current.version_id:
        return False

    previous_etag = previous.get("etag")
    if previous_etag and current.etag:
        return previous_etag == current.etag

    previous_modified = previous.get("last_modified")
    return _same_timestamp(previous_modified, current.last_modified)


def _same_timestamp(left: object, right: datetime | None) -> bool:
    if left is None or right is None:
        return False
    return left == right


def _normalize_prefix(prefix: str) -> str:
    cleaned = prefix.strip().replace("\\", "/").strip("/")
    if not cleaned:
        return ""
    normalized = PurePosixPath(cleaned).as_posix()
    if normalized in {".", ""}:
        return ""
    return normalized
