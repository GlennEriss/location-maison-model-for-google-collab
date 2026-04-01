from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetComposition(BaseModel):
    total_examples: int
    standard_cases_ratio: float
    hard_cases_ratio: float


class ExampleRecord(BaseModel):
    example_id: str
    split: str
    difficulty: str
    instruction: str = Field(default="Transforme cette annonce immobiliere en JSON structure.")
    description: str
    target_json: dict[str, Any]
    metadata: dict[str, Any]


class DatasetManifest(BaseModel):
    target_size: int
    generated_examples: int
    split_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    property_type_counts: dict[str, int]
    source_counts: dict[str, int] = Field(default_factory=dict)
    output_files: dict[str, str]
