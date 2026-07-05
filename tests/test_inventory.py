from __future__ import annotations

import io
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from lake_agent.domain.enums import Modality
from lake_agent.domain.models import (
    DiscoveredObject,
    IdentificationResult,
    ObjectLocator,
)
from lake_agent.inventory.identifier import ObjectIdentifier
from lake_agent.inventory.scanner import ObjectScanner
from lake_agent.inventory.service import InventoryService
from lake_agent.persistence.repositories import InventoryRepository
from lake_agent.storage.local_store import LocalFileStore


class FakeObjectStore:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.range_reads = 0

    def list_objects(self, bucket: str, prefix: str = ""):
        for key, content in self.objects.items():
            if key.startswith(prefix):
                yield DiscoveredObject(
                    locator=ObjectLocator(bucket, key),
                    etag=f"etag-{len(content)}-{key}",
                    size_bytes=len(content),
                    last_modified=datetime(2026, 6, 30, tzinfo=UTC),
                )

    def stat_object(self, locator: ObjectLocator) -> DiscoveredObject:
        content = self.objects[locator.object_key]
        return DiscoveredObject(
            locator=locator,
            etag=f"etag-{len(content)}-{locator.object_key}",
            size_bytes=len(content),
            last_modified=datetime(2026, 6, 30, tzinfo=UTC),
            declared_content_type="application/octet-stream",
            user_metadata={"source": "test"},
        )

    def read_range(self, locator: ObjectLocator, offset: int, length: int) -> bytes:
        self.range_reads += 1
        return self.objects[locator.object_key][offset : offset + length]

    def stream_object(self, locator: ObjectLocator):
        return io.BytesIO(self.objects[locator.object_key])


class FakeRepository:
    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}
        self.runs: dict[str, dict] = {}
        self._run_number = 0

    def create_run(self, bucket: str, prefix: str) -> str:
        self._run_number += 1
        run_id = f"run-{self._run_number}"
        self.runs[run_id] = {"bucket": bucket, "prefix": prefix}
        return run_id

    def find_object(self, identity: str):
        return self.objects.get(identity)

    def mark_seen(self, obj: DiscoveredObject, run_id: str) -> None:
        self.objects[obj.locator.identity]["last_seen_run_id"] = run_id

    def upsert_identified(self, obj, result, run_id: str) -> None:
        self.objects[obj.locator.identity] = {
            "object_id": obj.locator.object_id,
            "etag": obj.etag,
            "size_bytes": obj.size_bytes,
            "version_id": obj.version_id,
            "last_modified": obj.last_modified,
            "status": "identified",
            "sha256": result.sha256,
            "last_seen_run_id": run_id,
            "format": result.detected_format,
        }

    def upsert_failed_object(self, obj, run_id: str, message: str) -> None:
        self.objects[obj.locator.identity] = {
            "etag": obj.etag,
            "size_bytes": obj.size_bytes,
            "version_id": obj.version_id,
            "last_modified": obj.last_modified,
            "status": "error",
            "last_seen_run_id": run_id,
        }

    def record_error(self, *args, **kwargs) -> None:
        pass

    def mark_listing_completed(self, run_id: str) -> None:
        self.runs[run_id]["listing_completed"] = True

    def mark_unseen_missing(self, run_id: str, bucket: str, prefix: str) -> None:
        pass

    def complete_run(self, run_id: str, **counts) -> None:
        self.runs[run_id].update(counts)

    def fail_run(self, run_id: str, error: Exception) -> None:
        self.runs[run_id]["error"] = str(error)


class RecordingResult:
    def fetchone(self):
        return None


class PlaceholderCheckingConnection:
    def execute(self, query: str, params=None):
        if params is not None:
            expected = query.count("%s")
            if expected != len(params):
                raise AssertionError(
                    f"SQL expects {expected} parameters, received {len(params)}"
                )
        return RecordingResult()


class ObjectIdentifierTest(unittest.TestCase):
    def test_signature_wins_over_wrong_extension(self) -> None:
        store = FakeObjectStore({"wrong.txt": b"%PDF-1.7\ncontent"})
        obj = next(store.list_objects("datalake"))

        result = ObjectIdentifier(store).identify(obj)

        self.assertEqual("pdf", result.detected_format)
        self.assertEqual(Modality.DOCUMENT, result.modality)
        self.assertTrue(result.warnings)

    def test_sql_is_identified_as_textual_sql_script(self) -> None:
        store = FakeObjectStore({"class_grades.sql": b"CREATE TABLE scores (id INT);"})
        obj = next(store.list_objects("datalake"))

        result = ObjectIdentifier(store).identify(obj)

        self.assertEqual("sql", result.detected_format)
        self.assertEqual(Modality.SQL_SCRIPT, result.modality)
        self.assertEqual("utf-8", result.encoding)


class InventoryServiceTest(unittest.TestCase):
    def test_second_run_skips_unchanged_objects(self) -> None:
        store = FakeObjectStore(
            {
                "tables/sales.csv": b"name,sales\nA,10\n",
                "docs/report.pdf": b"%PDF-1.7\n",
            }
        )
        repository = FakeRepository()
        scanner = ObjectScanner(store)
        service = InventoryService(
            scanner,
            ObjectIdentifier(store),
            repository,
        )

        first = service.run("datalake")
        reads_after_first = store.range_reads
        second = service.run("datalake")

        self.assertEqual(2, first["identified_count"])
        self.assertEqual(0, first["unchanged_count"])
        self.assertEqual(0, second["identified_count"])
        self.assertEqual(2, second["unchanged_count"])
        self.assertEqual(reads_after_first, store.range_reads)

    def test_inventory_renames_wrong_extension_to_detected_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            original_path = root / "docs" / "report.txt"
            original_path.parent.mkdir(parents=True, exist_ok=True)
            original_path.write_bytes(b"%PDF-1.7\ncontent")

            repository = FakeRepository()
            store = LocalFileStore(tmp_dir)
            service = InventoryService(
                ObjectScanner(store),
                ObjectIdentifier(store),
                repository,
            )

            result = service.run()

            renamed_path = root / "docs" / "report.pdf"
            self.assertEqual(1, result["identified_count"])
            self.assertFalse(original_path.exists())
            self.assertTrue(renamed_path.exists())
            saved = repository.objects['["docs/report.pdf",""]']
            self.assertEqual("pdf", saved["format"])


class InventoryRepositoryTest(unittest.TestCase):
    def test_repository_statements_have_matching_parameter_counts(self) -> None:
        connection = PlaceholderCheckingConnection()
        repository = InventoryRepository(connection)
        locator = ObjectLocator("datalake", "tables/sales.csv")
        obj = DiscoveredObject(
            locator=locator,
            etag="abc",
            size_bytes=10,
            last_modified=datetime(2026, 6, 30, tzinfo=UTC),
        )
        result = IdentificationResult(
            locator=locator,
            detected_mime_type="text/csv",
            detected_format="csv",
            modality=Modality.TABULAR,
            encoding="utf-8",
            confidence=0.85,
        )

        repository.create_run("datalake", "")
        repository.find_object(locator.identity)
        repository.mark_seen(obj, "00000000-0000-0000-0000-000000000001")
        repository.upsert_identified(
            obj,
            result,
            "00000000-0000-0000-0000-000000000001",
        )
        repository.upsert_failed_object(
            obj,
            "00000000-0000-0000-0000-000000000001",
            "failed",
        )
        repository.record_error(
            "00000000-0000-0000-0000-000000000001",
            "datalake",
            obj.object_key,
            None,
            "identify",
            ValueError("failed"),
        )
        repository.mark_listing_completed(
            "00000000-0000-0000-0000-000000000001"
        )
        repository.mark_unseen_missing(
            "00000000-0000-0000-0000-000000000001",
            "datalake",
            "",
        )
        repository.complete_run(
            "00000000-0000-0000-0000-000000000001",
            discovered_count=1,
            identified_count=1,
            unchanged_count=0,
            error_count=0,
        )
        repository.fail_run(
            "00000000-0000-0000-0000-000000000001",
            ValueError("failed"),
        )


if __name__ == "__main__":
    unittest.main()
