"""
Chainlit app integrating a custom LLM chat model API + Langfuse v3 tracing (Python SDK).
"""

import re
import asyncio
import json
import logging
import requests
import chainlit as cl
from chainlit.message import Message
from datetime import datetime
from google.auth import default
from google.auth.transport.requests import Request
from transformers import AutoTokenizer
from langfuse import Langfuse, observe
from src.constants import ENDPOINT_ID, PROJECT_NUMBER
import time


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL_REPO_ID = "microsoft/Phi-3-mini-4k-instruct"
ENDPOINT_URL = (
    f"https://europe-west2-aiplatform.googleapis.com/v1/projects/"
    f"{PROJECT_NUMBER}/locations/europe-west2/endpoints/{ENDPOINT_ID}:predict"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
langfuse = Langfuse()  



def build_prompt(article: str) -> str:
    """
    Construit un prompt contextuel avec contrainte stricte de 5 lignes maximum.
    """
    summarization_prompt = (
        "You are a professional journalist assistant. "
        "Summarize the following article in **no more than 5 short lines**. "
        "Each line should be a full sentence. "
        "Avoid details, examples, and numbers unless essential. "
        "Focus only on the main idea, key facts, and implications.\n\n"
        f"Article:\n{article}\n\n"
        "Now write the summary:"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": summarization_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_response_from_prediction(prediction) -> str:
    """Extrait le texte utile du retour Vertex AI."""
    if isinstance(prediction, str):
        m = re.findall(r"(?:<\|assistant\|>)([^<]*)", prediction)
        return m[0].strip() if m else prediction

    if isinstance(prediction, dict):
        for key in ("generated_text", "output_text", "content", "text", "output"):
            if key in prediction and isinstance(prediction[key], str):
                return prediction[key]
        try:
            return json.dumps(prediction, ensure_ascii=False)
        except Exception:
            return str(prediction)

    return str(prediction)


def call_model_api(templated_input: str) -> tuple[str, dict]:
    """Appelle Vertex AI endpoint pour obtenir une prédiction."""
    credentials, _ = default()
    credentials.refresh(Request())
    access_token = credentials.token

    payload = {
        "instances": [{"input": templated_input}],
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.8,
            "max_new_tokens": 512,
        },
    }

    from requests.adapters import HTTPAdapter, Retry
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    r = session.post(
        ENDPOINT_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )

    resp = r.json()

    if "predictions" not in resp:
        logging.error(f"API error: {resp}")
        return f"❌ API error: {resp.get('error', resp)}", resp

    raw_model_response = resp["predictions"][0]
    extracted = extract_response_from_prediction(raw_model_response)
    return extracted, resp


@cl.on_chat_start
async def on_chat_start():
    """
    Initialise la session Chainlit.
    """
    await cl.Message(
        content="👋 Bienvenue ! Collez votre article ci-dessous pour obtenir un résumé clair et concis."
    ).send()


@cl.on_message
async def handle_message(message: Message):
    """
    Gère chaque message utilisateur (article collé).
    """

    if len(message.content.split()) > 2000:
        await cl.Message(
            content="⚠️ Votre article est trop long. Veuillez le réduire à 2000 mots maximum."
        ).send()
        return

    logging.info(f"User message received ({len(message.content.split())} mots)")

    templated_input = build_prompt(message.content)
    await cl.Message(content="🧠 Résumé en cours, merci de patienter...").send()

   
    @observe(name="vertex_ai_summarization", as_type="generation")
    def llm_call(prompt: str):
        """
        Appelle le modèle Vertex AI et logge toutes les métriques clés
        dans Langfuse (au niveau metadata pour qu'elles soient indexées).
        """
        start = time.perf_counter()
        output, raw_resp = call_model_api(prompt)
        duration = time.perf_counter() - start

        # Comptage des tokens
        input_tokens = len(tokenizer.encode(prompt))
        output_tokens = len(tokenizer.encode(output))
        total_tokens = input_tokens + output_tokens

        # ✅ Toutes les métriques importantes dans metadata
        return {
            "output": output,
            "metadata": {
                "model": MODEL_REPO_ID,
                "endpoint_id": ENDPOINT_ID,
                "project_number": PROJECT_NUMBER,
                "latency_seconds": round(duration, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "article_word_count": len(message.content.split()),
                "timestamp": datetime.utcnow().isoformat(),
            }
        }


    #Call the model : 1st call
    try:
        logging.info("Calling llm_call with Langfuse observation...")
        result = await asyncio.to_thread(llm_call, templated_input)
        answer = result["output"]

        # Text cleaning
        answer = re.sub(r"<\|.*?\|>", "", answer).strip()
        lines = [l.strip() for l in answer.splitlines() if l.strip()]

        # If text is too long, re-summarize
        if len(lines) > 5:
            short_text = " ".join(lines)
            logging.info(f"Résumé trop long ({len(lines)} lignes) → Re-summarizing.")

            short_prompt = (
                "Re-summarize the following text in **no more than 5 short lines**, "
                "keeping only the essential facts and key ideas:\n\n"
                f"{short_text}"
            )

            short_input = tokenizer.apply_chat_template(
                [{"role": "user", "content": short_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

            # 2nd model call, still observed by Langfuse    
            short_result = await asyncio.to_thread(llm_call, short_input)
            answer = re.sub(r"<\|.*?\|>", "", short_result["output"]).strip()

        #Final Truncature if still too long
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 5:
            answer = " ".join(sentences[:5]) + " [...]"
        answer = "\n".join(answer.split(". "))

        await cl.Message(content=answer).send()
        logging.info("✅ Résumé envoyé avec succès.")

    except Exception as e:
        err_msg = f"❌ Erreur : {e}"
        logging.exception(err_msg)
        await cl.Message(content=err_msg).send()
