"""Health endpoint response schema (SDD section 7.1)."""

from typing import Literal

from pydantic import BaseModel

DependencyStatus = Literal["ok", "error"]
# `offline`: fake mode, where nothing is embedded remotely, so there is no endpoint to check.
EmbeddingStatus = Literal["ok", "error", "offline"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    postgres: DependencyStatus
    qdrant: DependencyStatus
    llm_provider: Literal["configured", "unconfigured"]
    embeddings: EmbeddingStatus
