"""
api/schemas.py — Request/Response Pydantic models
"""
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class ResearchRequest(BaseModel):
    query:      str        = Field(..., min_length=5, max_length=500, description="Research question")
    model_size: str        = Field(default="0.6b", pattern="^(0\\.6b|1\\.5b)$")
    max_sources: int       = Field(default=8, ge=1, le=20)


class SourceItem(BaseModel):
    url:   str
    title: str
    score: float = 0.0


class ResearchResponse(BaseModel):
    session_id:   str
    query:        str
    model_used:   str
    status:       str
    final_report: Optional[str] = None
    key_findings: list[str]     = []
    sources_used: list[SourceItem] = []
    sub_queries:  list[str]     = []
    created_at:   str           = Field(default_factory=lambda: datetime.utcnow().isoformat())
    duration_sec: Optional[float] = None


class ModelInfo(BaseModel):
    id:          str
    description: str
    vram:        str
    speed:       str
    loaded:      bool


class HealthResponse(BaseModel):
    status:        str = "ok"
    version:       str
    models_loaded: list[str]
    gpu_available: bool


class ErrorResponse(BaseModel):
    detail: str
    code:   int