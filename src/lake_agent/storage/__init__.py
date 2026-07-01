"""Object-storage adapters."""

from lake_agent.storage.base import ObjectStore
from lake_agent.storage.local_store import LocalFileStore

__all__ = ["LocalFileStore", "ObjectStore"]
