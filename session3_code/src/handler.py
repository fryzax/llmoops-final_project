"""Handler for Hugging Face model inference requests."""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/opt/huggingface/model"


class EndpointHandler:
    """Handler for processing inference requests using a Hugging Face model."""

    def __init__(self, model_dir: str = MODEL_DIR) -> None:
        """Load tokenizer and model from the specified directory."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, device_map="cuda:0"
        ).eval()

    def generate(self, prompt, skip_special_tokens=False, **kwargs: Any) -> str:
        """Generate text based on the input prompt."""
        tokenized_input = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)
        generation_output = self.model.generate(
            **tokenized_input,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        return self.tokenizer.batch_decode(
            generation_output, skip_special_tokens=skip_special_tokens
        )[0]

    def __call__(self, data: dict[str, Any]) -> dict[str, list[Any]]:
        """Process inference requests containing image and text prompts."""
        return {
            "predictions": [
                self.generate(instance["input"], **data.get("parameters", {}))
                for instance in data["instances"]
            ]
        }
