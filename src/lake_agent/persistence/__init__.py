"""Metadata persistence backed by PostgreSQL."""

from lake_agent.persistence.database import PostgresDatabase
from lake_agent.persistence.repositories import InventoryRepository

__all__ = ["InventoryRepository", "PostgresDatabase"]
