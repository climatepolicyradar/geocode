import os
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from refined.inference.processor import Refined
from rich import print as rprint
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from models import LabelledPassage, Span

# Use this flag to limit the processing time for quick iteration
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
if DEBUG:
    rprint(
        "[bold yellow]Running in debug mode. Turn off when ready to run on the full dataset."
    )


def find_entities_refined(
    dataset_dir: Annotated[
        str,
        typer.Argument(
            help="Path to the directory containing parquet files, cloned from https://huggingface.co/datasets/ClimatePolicyRadar/all-document-text-data."
        ),
    ],
    skip_already_processed: bool = True,
    num_documents: Optional[int] = None,
    doc_geography_iso: Annotated[
        Optional[str],
        typer.Option(help="ISO 3166-1 alpha-3 code to filter documents by."),
    ] = None,
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
                "document_metadata.geographies",
            ],
        )
        df_list.append(df)

    documents = pd.concat(df_list, ignore_index=True)
    del df_list

    if doc_geography_iso:
        rprint(
            f"[bold blue]Filtering to documents with geography {doc_geography_iso}..."
        )
        documents = documents[
            documents["document_metadata.geographies"].apply(
                lambda x: x is not None and doc_geography_iso.upper() in x
            )
        ]
        rprint(
            f"[bold green]Filtered to {len(documents['document_id'].unique())} documents"
        )

    rprint("[bold blue]Processing and filtering dataset...")
    # Documents with no text have one row in the dataset, with a null value for text blocks
    documents = documents[
        documents["text_block.text"].notna() & documents["text_block.text"].str.strip()
        != ""
    ]
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

            debug__text_block_limit = 500 if DEBUG else None

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
                    try:
                        passages[text] = LabelledPassage(
                            id=f"{document_id}_{hash(text)}",
                            text=text,
                            spans=[],
                            metadata={"document_id": document_id},
                        )
                    except Exception as e:
                        rprint(
                            f"[red]Error processing text {text} for {document_id}. Skipping."
                        )
                        rprint(f"[red]Error: {e}")
                        continue

                for span in refined.process_text(text):
                    if span.predicted_entity is not None:
                        if span.predicted_entity.wikidata_entity_id:
                            all_qids.append(span.predicted_entity.wikidata_entity_id)

                        # TODO: this *probably* isn't needed, but is more just a failsafe
                        # because the code hasn't been run on the entire dataset yet.
                        try:
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
                        except Exception as e:
                            rprint(
                                f"[red]Error processing span {span} for {document_id}. Skipping."
                            )
                            rprint(f"[red]Error: {e}")
                            continue

            progress.remove_task(text_block_task)
            progress.update(task, advance=1)

            output_path = output_dir / f"{document_id}.jsonl"
            with open(output_path, "w") as f:
                for passage in passages.values():
                    f.write(passage.model_dump_json() + "\n")
            rprint(
                f"[bold green]Wrote {len(passages)} labelled passages to {output_path}"
            )


if __name__ == "__main__":
    typer.run(find_entities_refined)
