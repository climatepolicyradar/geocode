import os
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from pydantic import BaseModel, Field
from refined.inference.processor import Refined
from rich import print as rprint
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn


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


def generate_span_annotations(
    dataset_dir: str,
    skip_already_processed: bool = True,
    num_documents: Optional[int] = None,
    output_dir: Path = Path("./data/output"),
):
    """Generates a JSON file for each document in the dataset containing the spans found by refined."""

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        rprint(f"[bold green]Created output directory {output_dir}")

    rprint("[bold blue]Loading Refined model")
    refined = Refined.from_pretrained("wikipedia_model", entity_set="wikipedia")

    all_qids = []

    rprint("[bold blue]Loading parquet files...")
    parquet_files = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith(".parquet")
    ]

    df_list = []

    for parquet_file in parquet_files:
        df = pd.read_parquet(
            parquet_file,
            columns=[
                "document_id",
                "text_block.text",
                "text_block.language",
                "text_block.text_block_id",
                "text_block.type",
                "document_metadata.languages",
            ],
        )
        df_list.append(df)

    documents = pd.concat(df_list, ignore_index=True)

    rprint("[bold blue]Processing and filtering dataset...")
    documents = documents[
        ~documents["text_block.type"].isin(["pageNumber", "pageHeader", "pageFooter"])
    ]
    # NOTE: this disables geocoding within tables. Tables are represented as individual cells so you probably want to merge them into tables, or just
    # run this model on non-numeric cells.
    documents = documents[~documents["text_block.type"].isin(["TableCell"])]
    documents = documents[
        documents["document_metadata.languages"].apply(
            lambda x: "English" in x and len(x) == 1
        )
    ]

    memory_usage = documents.memory_usage(deep=True).sum() / (1024 * 1024)
    rprint(
        f"[bold green]Dataset size: {len(documents)} rows, {memory_usage:.2f} MB in memory"
    )

    if num_documents is not None:
        document_ids = documents["document_id"].unique()[:num_documents]
        documents = documents[documents["document_id"].isin(document_ids)]
        rprint(f"Limiting to {num_documents} documents")

    rprint(f"Loaded {len(documents)} English documents from parquet files.")

    document_groups = list(documents.groupby("document_id"))

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Processing documents", total=len(document_groups))

        for document_id, document_df in document_groups:
            if skip_already_processed and Path(f"./data/{document_id}.json").exists():
                progress.update(task, advance=1, description=f"Skipped {document_id}")
                continue

            progress.update(
                task,
                description=f"Processing {document_id} ({len(document_df)} text blocks)",
            )

            # Use a dict here to avoid processing duplicate text blocks
            passages: dict[str, LabelledPassage] = {}

            text_blocks = document_df["text_block.text"].tolist() or []
            text_block_task = progress.add_task(
                f"[cyan]Text blocks for {document_id}",
                total=len(text_blocks),
                visible=len(text_blocks) > 1,
            )

            debug__text_block_limit = None
            debug__text_block_limit = 500
            for i, text in enumerate(text_blocks):
                if debug__text_block_limit is not None and i > debug__text_block_limit:
                    break

                if i > 0:
                    progress.update(
                        text_block_task,
                        completed=i,
                        description=f"[cyan]Text block {i + 1}/{len(text_blocks)}",
                    )

                if text not in passages:
                    passages[text] = LabelledPassage(
                        id=f"{document_id}_{hash(text)}",
                        text=text,
                        spans=[],
                        metadata={"document_id": document_id},
                    )

                for span in refined.process_text(text):
                    if span.predicted_entity is not None:
                        if span.predicted_entity.wikidata_entity_id:
                            all_qids.append(span.predicted_entity.wikidata_entity_id)

                        passages[text].spans.append(
                            Span(
                                text=span.text,
                                start_index=span.start,
                                end_index=span.start + span.ln,
                                type=span.coarse_mention_type or "unknown",
                                fine_grained_type=span.predicted_entity_types[0][1]
                                if span.predicted_entity_types
                                else None,
                                id=span.predicted_entity.wikidata_entity_id
                                or "unknown",
                                probability=span.entity_linking_model_confidence_score,
                                wikipedia_title=span.predicted_entity.wikipedia_entity_title,
                                wikidata_id=span.predicted_entity.wikidata_entity_id,
                            )
                        )

            progress.remove_task(text_block_task)
            progress.update(task, advance=1)

            output_path = output_dir / f"{document_id}.jsonl"
            with open(output_path, "w") as f:
                for passage in passages.values():
                    f.write(passage.model_dump_json() + "\n")
            rprint(f"[bold green]Wrote {len(passages)} passages to {output_path}")


if __name__ == "__main__":
    typer.run(generate_span_annotations)
