"""
All models here have been adapted from CPR's knowledge graph project.

https://github.com/climatepolicyradar/knowledge-graph
"""

from typing import Optional

from pydantic import BaseModel, Field


class Span(BaseModel):
    """Represents a span within a text."""

    text: str = Field(..., description="The text of the span")
    start_index: int = Field(
        ..., ge=0, description="The start index of the span within the text"
    )
    end_index: int = Field(
        ..., gt=0, description="The end index of the span within the text"
    )
    type: Optional[str]
    fine_grained_type: Optional[str]
    id: Optional[str]
    probability: Optional[float]
    wikipedia_title: Optional[str]
    wikidata_id: Optional[str]


class LabelledPassage(BaseModel):
    """Represents a passage of text which has been labelled by an annotator"""

    id: str = Field(..., title="ID", description="The unique identifier of the passage")
    text: str = Field(..., title="Text", description="The text of the passage")
    spans: list[Span] = Field(
        default_factory=list,
        title="Spans",
        description="The spans in the passage which have been labelled by the annotator",
        repr=False,
    )
    metadata: dict = Field(
        default_factory=dict,
        title="Metadata",
        description="Additional data, eg translation status or dataset",
        repr=False,
    )
