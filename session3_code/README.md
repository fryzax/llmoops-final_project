# LLMOps Practice Answers Repository
> Practice answers for the LLMOps class for Albert School!

## Project requirements

### Astral/uv

- Install [uv](https://github.com/astral-sh/uv) to manage your Python version, virtual environments, dependencies and
tooling configs: see [installation docs](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).


## Installation

### Python virtual environment and dependencies

Run the following command:
```bash
uv sync
```

## GCP Setup

### Prerequisites

1. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) if not already installed
2. Create or identify a GCP project for the LLMOps class

### Authentication and Project Configuration

1. Authenticate with your Google account:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

2. Set your project ID:
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

### Enable Required APIs

Enable the necessary GCP services:

```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage-component.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com
```

### Environment Variables

Create a `.env` file in the project root that matches the .env.example file.
Upload the dataset to the matching GCS bucket and filename!