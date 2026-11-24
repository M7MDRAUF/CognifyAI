"""Shared schema primitives."""

from typing import Annotated

from pydantic import Field

# A score constrained to the [0, 1] interval (SDD evaluation fields).
Score = Annotated[float, Field(ge=0.0, le=1.0)]
