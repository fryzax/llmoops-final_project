"""Main constants of the project."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT_PATH = Path(__file__).parents[1]

# GCP Configuration
PROJECT_ID: str | None = os.getenv("GCP_PROJECT_ID")
REGION: str = os.getenv("GCP_REGION", "europe-west9")
BUCKET_NAME: str | None = os.getenv("GCP_BUCKET_NAME")
ENDPOINT_ID: str | None = os.getenv("GCP_ENDPOINT_ID")
PROJECT_NUMBER: str | None = os.getenv("GCP_PROJECT_NUMBER")

# Paths
RAW_DATASET_URI: str = f"gs://{BUCKET_NAME}/yoda_sentences.csv"
PIPELINE_ROOT_PATH: str = f"{BUCKET_NAME}/vertexai-pipeline-root/"
ARTIFACT_URI = "gs://llmops_souesme_arroyo_fernandes/vertexai-pipeline-root/54825872111/pipeline-test-20251022082046/fine-tuning-component_-3099071654299435008/model/"
