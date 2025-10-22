"""Chainlit app integrating a custom LLM chat model API."""

import re

import chainlit as cl
import requests
from chainlit.message import Message
from google.auth import default
from google.auth.transport.requests import Request
from transformers import AutoTokenizer

from src.constants import ENDPOINT_ID, PROJECT_NUMBER

MODEL_REPO_ID = "microsoft/Phi-3-mini-4k-instruct"
ENDPOINT_URL = f"https://europe-west2-aiplatform.googleapis.com/v1/projects/{PROJECT_NUMBER}/locations/europe-west2/endpoints/{ENDPOINT_ID}:predict"


@cl.set_starters  # type: ignore
async def set_starters():
    """Set starter messages for the Chainlit app."""
    return [
        cl.Starter(
            label="Tech News Article",
            message="Artificial intelligence has made significant strides in recent years, with large language models demonstrating unprecedented capabilities in understanding and generating human-like text. These models are being deployed across various industries, from healthcare to finance, revolutionizing how we interact with technology. However, concerns about ethical implications and potential misuse remain at the forefront of discussions among researchers and policymakers.",
        ),
        cl.Starter(
            label="Business News",
            message="The global economy showed mixed signals in the latest quarter, with some sectors experiencing robust growth while others faced challenges. Technology companies continued to dominate market performance, driven by innovations in cloud computing and artificial intelligence. Meanwhile, traditional retail struggled to adapt to changing consumer preferences and the ongoing shift to e-commerce platforms.",
        ),
        cl.Starter(
            label="Science Discovery",
            message="Scientists have discovered a new method for producing clean energy that could revolutionize the renewable energy sector. The breakthrough involves a novel approach to solar panel efficiency, potentially increasing energy output by up to 40%. Researchers believe this technology could be scaled for commercial use within the next five years, marking a significant step forward in the fight against climate change.",
        ),
    ]


@cl.on_message
async def handle_message(message: Message):
    """Handle incoming messages from the user."""
    await cl.Message(content=call_model_api(message)).send()


def build_prompt(tokenizer: AutoTokenizer, article: str):
    """Build a prompt from an article applying the chat template."""
    summarization_prompt = f"Summarize the following article:\n{article}"
    return tokenizer.apply_chat_template(  # type: ignore
        [
            {"role": "user", "content": summarization_prompt},
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
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)

        # Get access token using Google Auth library
        credentials, project = default()
        credentials.refresh(Request())
        access_token = credentials.token

        templated_input = build_prompt(tokenizer, message.content)
        model_input = {
            "instances": [{"input": templated_input}],
            "parameters": {
                "temperature": 0.1,
                "top_p": 0.8,              # ✅ Correct
                "max_new_tokens": 512,   
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

        # Debug: check what's in the response
        if "predictions" not in response:
            return f"❌ Error from API: {response.get('error', response)}"

        raw_model_response = response["predictions"][0]
        extracted_response = extract_response(raw_model_response)
        return extracted_response

    except Exception as e:
        return f"❌ Error: {str(e)}"
