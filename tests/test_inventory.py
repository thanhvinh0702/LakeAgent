from __future__ import annotations

import io
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from lake_agent.domain.enums import FileStatus, Modality
from lake_agent.domain.models import FileMetadata
from lake_agent.inventory.identifier import ObjectIdentifier
from lake_agent.inventory.scanner import ObjectScanner
from lake_agent.inventory.service import InventoryService
from lake_agent.persistence.repositories import InventoryRepository
from lake_agent.storage.local_store import LocalFileStore


class FakeObjectStore:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.range_reads = 0

    def list_objects(self, prefix: str = ""):
        for key, content in self.objects.items():
            if key.startswith(prefix):
                yield FileMetadata(
                    object_key=key,
                    etag=f"etag-{len(content)}-{key}",
                    size_bytes=len(content),
                    last_modified=datetime(2026, 6, 30, tzinfo=UTC),
                )

    def stat_object(self, obj: FileMetadata) -> FileMetadata:
        content = self.objects[obj.object_key]
        return FileMetadata(
            object_key=obj.object_key,
            etag=f"etag-{len(content)}-{obj.object_key}",
            size_bytes=len(content),
            last_modified=datetime(2026, 6, 30, tzinfo=UTC),
            declared_content_type="application/octet-stream",
            user_metadata={"source": "test"},
        )

    def read_range(self, obj: FileMetadata, offset: int, length: int) -> bytes:
        self.range_reads += 1
        return self.objects[obj.object_key][offset : offset + length]

    def stream_object(self, obj: FileMetadata):
        return io.BytesIO(self.objects[obj.object_key])

    def rename_object(self, obj: FileMetadata, new_object_key: str) -> FileMetadata:
        content = self.objects.pop(obj.object_key)
        self.objects[new_object_key] = content
        return FileMetadata(
            object_key=new_object_key,
            etag=obj.etag,
            size_bytes=obj.size_bytes,
            last_modified=obj.last_modified,
            declared_content_type=obj.declared_content_type,
            user_metadata=obj.user_metadata,
        )


class FakeRepository:
    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}

    def find_object(self, identity: str):
        return self.objects.get(identity)

    def save(self, obj: FileMetadata, scanned_at: datetime) -> None:
        self.objects[obj.identity] = {
            "object_id": obj.object_id,
            "etag": obj.etag,
            "size_bytes": obj.size_bytes,
            "version_id": obj.version_id,
            "last_modified": obj.last_modified,
            "status": obj.status.value,
            "format": obj.detected_format,
        }

    def mark_missing(self, prefix: str, scanned_at: datetime) -> None:
        pass


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
        obj = next(store.list_objects())

        result = ObjectIdentifier(store).identify(obj)

        self.assertEqual("pdf", result.detected_format)
        self.assertEqual(Modality.DOCUMENT, result.modality)
        self.assertTrue(result.warnings)

    def test_sql_is_identified_as_textual_sql_script(self) -> None:
        store = FakeObjectStore({"class_grades.sql": b"CREATE TABLE scores (id INT);"})
        obj = next(store.list_objects())

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

        first = service.run("")
        reads_after_first = store.range_reads
        second = service.run("")

        self.assertEqual(2, first["identified_count"])
        self.assertEqual(0, first["unchanged_count"])
        self.assertEqual(0, second["identified_count"])
        self.assertEqual(2, second["unchanged_count"])
        self.assertEqual(reads_after_first, store.range_reads)

    def test_inventory_keeps_original_extension_when_signature_disagrees(self) -> None:
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

            self.assertEqual(1, result["identified_count"])
            self.assertTrue(original_path.exists())
            self.assertFalse((root / "docs" / "report.pdf").exists())
            saved = repository.objects['["docs/report.txt",""]']
            self.assertEqual("pdf", saved["format"])


class InventoryRepositoryTest(unittest.TestCase):
    def test_repository_statements_have_matching_parameter_counts(self) -> None:
        connection = PlaceholderCheckingConnection()
        repository = InventoryRepository(connection)
        
        obj = FileMetadata(
            object_key="tables/sales.csv",
            etag="abc",
            size_bytes=10,
            last_modified=datetime(2026, 6, 30, tzinfo=UTC),
            detected_mime_type="text/csv",
            detected_format="csv",
            modality=Modality.TABULAR,
            encoding="utf-8",
            identification_confidence=0.85,
        )

        repository.find_object(obj.identity)
        repository.save(obj, datetime.now(UTC))
        repository.mark_missing("datalake", datetime.now(UTC))


if __name__ == "__main__":
    unittest.main()
