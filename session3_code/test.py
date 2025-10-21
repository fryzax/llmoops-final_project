import pandas as pd
import os
from pathlib import Path


BUCKET_NAME: str | None = os.getenv("GCP_BUCKET_NAME")

df = pd.read_csv(f"gs://{BUCKET_NAME}/yoda_sentences.csv")
print(df.head())