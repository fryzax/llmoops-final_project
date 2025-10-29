"""Main constants of the project."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT_PATH = Path(__file__).parents[1]

# GCP Configuration
PROJECT_ID: str | None = os.getenv("GCP_PROJECT_ID")
PROJECT_NUMBER: str | None = os.getenv("GCP_PROJECT_NUMBER")
REGION: str = os.getenv("GCP_REGION", "europe-west2")
BUCKET_NAME: str | None = os.getenv("GCP_BUCKET_NAME")
ENDPOINT_ID: str | None = os.getenv("GCP_ENDPOINT_ID")

# Paths

RAW_DATASET_URI: str = f"gs://{BUCKET_NAME}/data_newsmind_test.csv"
PIPELINE_ROOT_PATH: str = f"{BUCKET_NAME}/vertexai-pipeline-cnn/"
