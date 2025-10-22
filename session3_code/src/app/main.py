"""Chainlit app integrating a custom LLM chat model API."""

import re
import subprocess

import chainlit as cl
import requests
from chainlit.message import Message
from transformers import AutoTokenizer

from src.constants import ENDPOINT_ID, PROJECT_NUMBER

MODEL_REPO_ID = "microsoft/Phi-3-mini-4k-instruct"
ENDPOINT_URL = f"https://europe-west2-aiplatform.googleapis.com/v1/projects/{PROJECT_NUMBER}/locations/europe-west2/endpoints/{ENDPOINT_ID}:predict"


@cl.set_starters  # type: ignore
async def set_starters():
    """Set starter messages for the Chainlit app."""
    return [
        cl.Starter(
            label="Message #1 - Lightsaber talk",
            message="Paint the handle of your lightsaber on your hip dull gray.",
        ),
        cl.Starter(
            label="Message #2 - Not very nice",
            message="He is not very nice.",
        ),
        cl.Starter(
            label="Message #3 - Motivational quote",
            message="You must believe in the force.",
        ),
    ]


@cl.on_message
async def handle_message(message: Message):
    """Handle incoming messages from the user."""
    await cl.Message(content=call_model_api(message)).send()


def build_prompt(tokenizer: AutoTokenizer, sentence: str):
    """Build a prompt from a sentence applying the chat template."""
    return tokenizer.apply_chat_template(  # type: ignore
        [
            {"role": "user", "content": sentence},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_response(generated_text: str) -> str:
    """Extract the model's response from the generated text."""
    return re.findall(
        r"(?:<\|assistant\|>)([^<]*)",
        generated_text,
    )[0]


def call_model_api(message: Message) -> str:
    """Call the custom LLM chat model API."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)

    access_token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True
    ).strip()

    templated_input = build_prompt(tokenizer, message.content)
    model_input = {
        "instances": [{"input": templated_input}],
        "parameters": {
            "maxOutputTokens": 64,
            "temperature": 0.1,
            "topP": 0.8,
        },
    }
    response = requests.post(
        ENDPOINT_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=model_input,
    ).json()
    raw_model_response = response["predictions"][0]

    extracted_response = extract_response(raw_model_response)

    return extracted_response
