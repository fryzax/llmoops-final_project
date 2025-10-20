"""Validation scripts to test GCP authentication and service connectivity."""

from google.cloud import aiplatform, storage

from src.constants import (
    BUCKET_NAME,
    PROJECT_ID,
    REGION,
)


def validate_vertex_ai_connectivity() -> bool:
    """Validate Vertex AI API connectivity."""
    try:
        aiplatform.init(project=PROJECT_ID, location=REGION)
        print("✓ Vertex AI API connectivity successful")
        return True
    except Exception as error:
        print(f"✗ Vertex AI API connectivity failed: {error}")
        return False


def validate_bucket_access(bucket_name: str) -> bool:
    """Validate access to a specific GCS bucket."""
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        bucket.reload()
        print(f"✓ GCS bucket access successful: {bucket_name}")
        return True
    except Exception as error:
        print(f"✗ GCS bucket access failed: {error}")
        return False


def run_all_validations() -> None:
    """Run all validation checks and return overall success status."""
    print("Running GCP setup validation...")
    print("=" * 50)

    validations = [
        validate_vertex_ai_connectivity(),
        validate_bucket_access(BUCKET_NAME) if BUCKET_NAME else False,
    ]

    if sum(validations) == len(validations):
        print("✓ All validations passed! GCP setup is ready.")
    else:
        print("✗ Some validations failed. Please check your GCP configuration.")


if __name__ == "__main__":
    run_all_validations()
