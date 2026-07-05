from __future__ import annotations

from pydantic import BaseModel, Field


class EnrichedTextResult(BaseModel):
    file_summary: str | None = None
    file_keywords: list[str] = Field(default_factory=list)
