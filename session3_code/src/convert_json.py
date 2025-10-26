"""convert_json.py
Script robuste pour convertir les exports Langfuse (JSON / NDJSON) vers CSV ou SQLite.

Usage:
  python convert_json.py --input <path-to-json> [--csv out.csv] [--sqlite out.db]

This script attempts to parse multiple possible export shapes and extract useful
fields such as trace_id, timestamp, model, input/output text, and token counts.
"""

import argparse
import json
import os
from datetime import datetime
import sqlite3
from typing import Any, Dict, List

import pandas as pd


def load_json_flex(path: str) -> List[Dict[str, Any]]:
    """Charge un fichier JSON ou NDJSON et retourne une liste d'objets.

    Supporte:
    - JSON standard (object avec clé 'data' ou 'observations' ou liste au root)
    - NDJSON (un JSON par ligne)
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items

    if isinstance(obj, dict):
        for key in ("data", "observations", "items", "rows"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        return [obj]

    if isinstance(obj, list):
        return obj

    # Fallback
    return [obj]


def extract_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract useful fields from a Langfuse observation/generation/event item.

    The function is defensive: it tries multiple common key names.
    """
    def get(d, *keys, default=None):
        if keys and not all(isinstance(k, str) for k in keys):
            for i in range(len(keys)-1, -1, -1):
                if not isinstance(keys[i], str):
                    default = keys[i]
                    keys = keys[:i]
                    break

        for k in keys:
            if isinstance(d, dict) and k in d:
                return d[k]
        return default

    attrs = get(item, "attributes", {}) or {}

    observation_id = get(item, "id") or get(attrs, "id")
    trace_id = get(item, "trace_id") or get(attrs, "trace_id") or get(item, "trace")
    trace_name = get(item, "name") or get(attrs, "name") or get(attrs, "trace_name")

    timestamp = get(attrs, "timestamp") or get(item, "timestamp")
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()

    input_text = get(item, "input") or get(attrs, "input") or get(item, "prompt") or get(attrs, "prompt") or get(item, "content")
    if isinstance(input_text, dict):
        input_text = json.dumps(input_text, ensure_ascii=False)

    output_text = get(item, "output") or get(attrs, "output") or get(item, "response") or get(attrs, "response") or get(item, "generated_text")
    if isinstance(output_text, dict):
        output_text = json.dumps(output_text, ensure_ascii=False)

    model = get(attrs, "model") or get(item, "model")
    latency_seconds = get(attrs, "latency_seconds") or get(attrs, "latency")
    input_tokens = get(attrs, "input_tokens") or get(attrs, "inputTokenCount")
    output_tokens = get(attrs, "output_tokens") or get(attrs, "outputTokenCount")
    total_tokens = get(attrs, "total_tokens") or get(attrs, "totalTokenCount")

    input_word_count = None
    output_word_count = None
    if isinstance(input_text, str):
        input_word_count = len(input_text.split())
    if isinstance(output_text, str):
        output_word_count = len(output_text.split())

    raw = json.dumps(item, ensure_ascii=False)

    return {
        "observation_id": observation_id,
        "trace_id": trace_id,
        "trace_name": trace_name,
        "timestamp": timestamp,
        "model": model,
        "latency_seconds": latency_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_text": input_text,
        "output_text": output_text,
        "input_word_count": input_word_count,
        "output_word_count": output_word_count,
        "raw": raw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=False, default="src/1761477612243-lf-observations-export-cmh270g6p000mad07qlm3ynvv.json", help="Chemin vers le fichier JSON export Langfuse")
    parser.add_argument("--csv", "-c", required=False, help="Chemin de sortie CSV")
    parser.add_argument("--sqlite", "-s", required=False, help="Chemin fichier SQLite (db)")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Fichier introuvable: {input_path}")
        return

    print(f"Chargement de: {input_path}")
    items = load_json_flex(input_path)
    print(f"Objets extraits: {len(items)}")

    rows = [extract_fields(it) for it in items]
    df = pd.DataFrame(rows)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"✅ CSV écrit: {args.csv}")

    if args.sqlite:
        conn = sqlite3.connect(args.sqlite)
        df.to_sql("observations", conn, if_exists="replace", index=False)
        conn.close()
        print(f"✅ SQLite écrit: {args.sqlite} (table: observations)")

    if not args.csv and not args.sqlite:
        print(df.head().to_string())


if __name__ == "__main__":
    main()

