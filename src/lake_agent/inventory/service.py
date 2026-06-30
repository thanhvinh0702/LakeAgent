from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Mapping, Any

from lake_agent.domain.enums import FileStatus
from lake_agent.domain.models import DiscoveredObject, InventoryRunResult
from lake_agent.inventory.hasher import ObjectHasher
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
        hasher: ObjectHasher | None = None,
        hash_content: bool = False,
        stat_new_or_changed: bool = True,
    ) -> None:
        if hash_content and hasher is None:
            raise ValueError("hasher is required when hash_content=True")
        self._scanner = scanner
        self._identifier = identifier
        self._repository = repository
        self._hasher = hasher
        self._hash_content = hash_content
        self._stat_new_or_changed = stat_new_or_changed

    def run(self, bucket: str, prefix: str = "") -> InventoryRunResult:
        run_id = self._repository.create_run(bucket, prefix)
        discovered_count = 0
        identified_count = 0
        unchanged_count = 0
        error_count = 0

        try:
            for listed_object in self._scanner.scan(bucket, prefix):
                discovered_count += 1
                previous = self._repository.find_object(
                    listed_object.locator.identity
                )

                if previous and _is_unchanged(previous, listed_object):
                    self._repository.mark_seen(listed_object, run_id)
                    unchanged_count += 1
                    continue

                current_object = listed_object
                try:
                    if self._stat_new_or_changed:
                        current_object = self._scanner.stat(listed_object)

                    result = self._identifier.identify(current_object)
                    if self._hash_content:
                        assert self._hasher is not None
                        result = replace(
                            result,
                            sha256=self._hasher.sha256(current_object.locator),
                        )
                    self._repository.upsert_identified(
                        current_object,
                        result,
                        run_id,
                    )
                    identified_count += 1
                except Exception as exc:
                    error_count += 1
                    self._repository.upsert_failed_object(
                        current_object,
                        run_id,
                        str(exc),
                    )
                    self._repository.record_error(
                        run_id,
                        current_object.bucket,
                        current_object.object_key,
                        current_object.version_id,
                        "identify",
                        exc,
                    )

            # Missing detection is safe only after listing reached its natural end.
            self._repository.mark_listing_completed(run_id)
            self._repository.mark_unseen_missing(run_id, bucket, prefix)
            self._repository.complete_run(
                run_id,
                discovered_count=discovered_count,
                identified_count=identified_count,
                unchanged_count=unchanged_count,
                error_count=error_count,
            )
        except Exception as exc:
            self._repository.fail_run(run_id, exc)
            raise

        return InventoryRunResult(
            run_id=run_id,
            bucket=bucket,
            prefix=prefix,
            status=FileStatus.COMPLETED,
            discovered_count=discovered_count,
            identified_count=identified_count,
            unchanged_count=unchanged_count,
            error_count=error_count,
        )


def _is_unchanged(
    previous: Mapping[str, Any],
    current: DiscoveredObject,
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
