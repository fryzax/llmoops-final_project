"""Data transformation component for Vertex AI pipeline.

This component prepares a summarization dataset with the following input schema (fixed columns):
- id: string (hex SHA1 of the source URL) [optional]
- article: string (full article body)
- highlights: string (reference summary)

It formats each example into a Phi-3 chat-style "messages" field:
[
    {"role": "user", "content": "Summarize the following article:\n<article>"},
    {"role": "assistant", "content": "<highlights>"}
]
and then performs a train/test split.
"""

from kfp.dsl import OutputPath, component


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas>=2.3.2",
        "datasets==4.0.0",
        "gcsfs",
    ],
)
def data_transformation_component(
    raw_dataset_uri: str,
    train_test_split_ratio: float,
    train_dataset: OutputPath("Dataset"),  # type: ignore
    test_dataset: OutputPath("Dataset"),  # type: ignore
) -> None:
    """Format and split summarization data for Phi-3 fine-tuning."""
    import logging

    import pandas as pd
    from datasets import Dataset

    def format_dataset_to_phi_messages(dataset: Dataset) -> Dataset:
        """Format dataset to Phi messages structure from article/highlights.

        Assumes fixed lowercase columns: 'article' and 'highlights'.
        Keeps optional 'id' column for traceability (not used in training).
        """

        def build_prompt(article: str) -> str:
            return f"Summarize the following article:\n{article}"

        def format_dataset(examples):
            """Format a single example to Phi messages structure."""
            converted_sample = [
                {"role": "user", "content": build_prompt(examples["prompt"])},
                {"role": "assistant", "content": examples["completion"]},
            ]
            return {"messages": converted_sample}

        # Directly use fixed columns 'article' and 'highlights'
        ds = dataset.rename_column("article", "prompt").rename_column(
            "highlights", "completion"
        )

        # Map to messages; keep 'id' if present for traceability
        mapped = ds.map(format_dataset)
        cols_to_remove = [c for c in ["prompt", "completion"] if c in mapped.column_names]
        return mapped.remove_columns(cols_to_remove)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting data transformation process...")

    logger.info(f"Reading from {raw_dataset_uri}")
    dataset = Dataset.from_pandas(pd.read_csv(raw_dataset_uri))

    logger.info("Formatting and splitting dataset...")
    formatted_dataset = format_dataset_to_phi_messages(dataset)
    split_dataset = formatted_dataset.train_test_split(test_size=train_test_split_ratio)

    logger.info(f"Writing train dataset to {train_dataset}...")
    split_dataset["train"].to_csv(train_dataset, index=False)

    logger.info(f"Writing test dataset to {test_dataset}...")
    split_dataset["test"].to_csv(test_dataset, index=False)

    logger.info("Data transformation process completed successfully")
