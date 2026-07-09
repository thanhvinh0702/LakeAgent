from __future__ import annotations

from pydantic import BaseModel, Field


class EnrichedDatabaseTableProfile(BaseModel):
    table_id: str
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)


class EnrichedDatabaseResult(BaseModel):
    file_summary: str | None = None
    file_keywords: list[str] = Field(default_factory=list)
    tables: list[EnrichedDatabaseTableProfile] = Field(default_factory=list)
