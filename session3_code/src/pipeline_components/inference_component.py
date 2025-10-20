"""Test set inference component for Vertex AI."""

from kfp.dsl import Dataset, Input, Model, OutputPath, component


@component(
    base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel",
    packages_to_install=[
        "google-cloud-storage>=2.10.0",
        "transformers==4.46.*",
        "peft==0.13.2",
        "datasets==4.0.0",
        "pandas==2.2.2",
        "gcsfs",
    ],
)
def inference_component(
    dataset: Input[Dataset],
    model: Input[Model],
    predictions: OutputPath("Dataset"),  # type: ignore
):
    """Computes predictions on the test dataset."""
    import logging
    import re
    from pathlib import Path
    from typing import Any

    import pandas as pd
    import torch
    from google.cloud import storage
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def download_model(model_uri: str, local_dir: str):
        """Download model from GCS to local directory."""
        bucket_name, prefix = model_uri.replace("gs://", "").split("/", 1)

        bucket = storage.Client().get_bucket(bucket_name)
        for blob in bucket.list_blobs(prefix=prefix):
            filename = blob.name.split("/")[-1]
            if filename != "":
                blob.download_to_filename(f"{local_dir}/{filename}")

    def build_prompt(tokenizer: AutoTokenizer, sentence: str):
        """Build a prompt from a sentence applying the chat template."""
        return tokenizer.apply_chat_template(  # type: ignore
            [
                {"role": "user", "content": sentence},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        **kwargs: Any,
    ):
        tokenized_input = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)  # type: ignore

        generation_output = model.generate(  # type: ignore
            **tokenized_input,
            eos_token_id=tokenizer.eos_token_id,  # type: ignore
            **kwargs,
        )
        return tokenizer.batch_decode(  # type: ignore
            generation_output, skip_special_tokens=False
        )[0]

    def extract_response(generated_text: str) -> str:
        """Extract the model's response from the generated text."""
        return re.findall(
            r"(?:<\|assistant\|>)([^<]*)",
            generated_text,
        )[0]

    local_dir = Path("model")
    local_dir.mkdir(parents=True, exist_ok=True)
    repo_id = "microsoft/Phi-3-mini-4k-instruct"

    logger.info(f"Downloading model from {model.uri} to {local_dir}...")
    download_model(model.uri, str(local_dir))

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    model_instance = AutoModelForCausalLM.from_pretrained(
        local_dir, torch_dtype=torch.float16
    ).eval()

    logger.info(f"Loading dataset from {dataset.path}...")
    test_dataset = pd.read_csv(dataset.path).assign(
        messages=lambda df: df["messages"].apply(lambda x: eval(x.replace("\n", ",")))
    )

    predictions_df = []
    for _, row in tqdm(test_dataset.iterrows(), total=test_dataset.shape[0]):
        user_input = row["messages"][0]["content"]
        reference = row["messages"][1]["content"]
        response = extract_response(
            generate(
                model_instance,
                tokenizer,
                build_prompt(tokenizer, user_input),
                max_new_tokens=64,
            )
        )
        predictions_df.append(
            {
                "user_input": user_input,
                "reference": reference,
                "response": response,
            }
        )

    logger.info(f"Writing predictions to {predictions}...")
    pd.DataFrame(predictions_df).to_csv(predictions, index=False)
