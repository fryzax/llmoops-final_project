"""
Chainlit app integrating a custom LLM chat model API.
"""

# ============================================================
#                      IMPORTS
# ============================================================

import re
import logging
import requests
import chainlit as cl
from chainlit.message import Message

from google.auth import default
from google.auth.transport.requests import Request
from transformers import AutoTokenizer

from src.constants import ENDPOINT_ID, PROJECT_NUMBER


# ============================================================
#                CONFIGURATION & INITIALIZATION
# ============================================================

# Logging local
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL_REPO_ID = "microsoft/Phi-3-mini-4k-instruct"

ENDPOINT_URL = (
    f"https://europe-west2-aiplatform.googleapis.com/v1/projects/"
    f"{PROJECT_NUMBER}/locations/europe-west2/endpoints/{ENDPOINT_ID}:predict"
)

# On instancie le tokenizer UNE SEULE FOIS (gain de perf important)
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)


# ============================================================
#                     HELPER FUNCTIONS
# ============================================================

def build_prompt(article: str) -> str:
    """
    Construit un prompt contextuel et adapté à la tâche de résumé.
    """
    summarization_prompt = (
        "You are a helpful assistant specialized in summarizing news articles.\n\n"
        "Your task: produce a clear and concise summary in 3-5 bullet points.\n"
        "Focus on key facts, actors, and implications.\n\n"
        f"Article:\n{article}"
    )

    # Utilisation du template chat du tokenizer pour formater proprement
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": summarization_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_response(generated_text: str) -> str:
    """
    Extrait uniquement la partie réponse assistant du texte brut renvoyé par le modèle.
    """
    m = re.findall(r"(?:<\|assistant\|>)([^<]*)", generated_text)
    return m[0].strip() if m else generated_text


def call_model_api(templated_input: str) -> tuple[str, dict]:
    """
    Appelle le modèle hébergé sur Vertex AI et renvoie (réponse, réponse brute).
    """
    # Récupération d'un token d'accès OAuth pour GCP
    credentials, _ = default()
    credentials.refresh(Request())
    access_token = credentials.token

    # Payload avec paramètres de génération
    payload = {
        "instances": [{"input": templated_input}],
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.8,
            "max_new_tokens": 512,
        },
    }

    # Gestion des retries automatiques (résilience)
    from requests.adapters import HTTPAdapter, Retry
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    # Appel HTTP vers Vertex AI Endpoint
    r = session.post(
        ENDPOINT_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    resp = r.json()

    # Gestion d'erreur si la réponse n'est pas au bon format
    if "predictions" not in resp:
        logging.error(f"API error: {resp}")
        return f"❌ API error: {resp.get('error', resp)}", resp

    raw_model_response = resp["predictions"][0]
    extracted = extract_response(raw_model_response)
    return extracted, resp


# ============================================================
#                      CHAT HANDLERS
# ============================================================

@cl.on_chat_start
async def on_chat_start():
    """
    Initialise la session de chat.
    """
    await cl.Message(
        content="👋 Bienvenue ! Collez votre article ci-dessous pour obtenir un résumé clair et concis."
    ).send()


@cl.on_message
async def handle_message(message: Message):
    """
    Gère chaque message utilisateur (article collé).
    """

    # Vérification de la longueur du texte (garde-fou)
    if len(message.content.split()) > 2000:
        warning_msg = "⚠️ Votre article est trop long. Veuillez le réduire à 2000 mots maximum."
        await cl.Message(content=warning_msg).send()
        return

    # Log du message utilisateur
    logging.info(f"User message received ({len(message.content.split())} mots)")

    try:
        # Construction du prompt
        templated_input = build_prompt(message.content)

        # Message de progression utilisateur
        await cl.Message(content="🧠 Résumé en cours, merci de patienter...").send()

        # Simulation d'un streaming (affichage progressif)
        msg = cl.Message(content="")
        await msg.send()

        # Appel modèle
        answer, raw_resp = call_model_api(templated_input)

        # On stream le texte mot par mot pour un effet "ChatGPT-like"
        for token in answer.split():
            await msg.stream_token(token + " ")
        await msg.update()

        # Logging local
        logging.info("Model response sent successfully")

    except Exception as e:
        err = f"❌ Erreur: {e}"
        logging.error(err)
        await cl.Message(content=err).send()
