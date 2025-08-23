from pathlib import Path
import os
import yaml
import json
import asyncio
from typing import Any, Dict, Optional

import chromadb
from chromadb.utils import embedding_functions

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.readers.json import JSONReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("api_key")
model = os.getenv("model")
# ------------------------------
# Load Prompt from YAML
# ------------------------------
def load_prompt(yaml_path: str) -> str:
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    prompt = config.get("prompt", "").strip()
    if not prompt:
        raise ValueError("`prompt` missing or empty in YAML config.")
    return prompt


# ------------------------------
# Suggestion Generation Function
# ------------------------------
async def suggestion_generation(
    user_id: str,
    file_path: str,
    yaml_path: str,
    groq_api_key: Optional[str] = None,
    persist_dir: str = "./chroma_data",
    reset_index: bool = False,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Build/Query a per-user Chroma collection and return JSON suggestions.

    Args:
        user_id: unique identifier for the user/session (e.g., "u_123").
        file_path: path to JSON data to index (same format you used with JSONReader).
        yaml_path: path to YAML config containing `prompt`.
        groq_api_key: your Groq API key (or set via env var GROQ_API_KEY).
        persist_dir: Chroma persistence directory.
        reset_index: if True, drops and recreates the user's collection.
        top_k: retrieval depth.

    Returns:
        dict parsed from model JSON, or error dict.
    """

    # --- LLM (Groq) + Embeddings (HuggingFace) per-call (concurrency-safe) ---
    llm = Groq(model= model , api_key=groq_api_key)
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # --- Load documents to index (per user) ---
    loader = JSONReader()
    documents = loader.load_data(Path(file_path))

    # --- Chroma client & per-user collection ---
    client = chromadb.PersistentClient(path=persist_dir)
    collection_name = f"analysis_{user_id}"

    if reset_index:
        # Drop collection if exists
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass  # ok if it doesn't exist

    # Create or get the user's collection
    collection = client.get_or_create_collection(name=collection_name)

    # Wrap in LlamaIndex vector store
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # --- Build/Update index for this user ---
    # This will upsert embeddings as needed for the documents
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        llm=llm,
        embed_model=embed_model,
    )

    # --- Retrieve & Generate with YAML prompt ---
    prompt = load_prompt(yaml_path)
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(prompt)

    # --- Parse model output as JSON ---
    try:
        raw = (response.response or "").strip()
        if raw.startswith("```"):
            # Strip code fences (and optional language tag)
            raw = raw.strip("`").lstrip("json").strip()
        parsed = json.loads(raw)
        return parsed
    except Exception as e:
        return {
            "error": "Failed to parse LLM response as JSON",
            "detail": response.response if hasattr(response, "response") else str(response),
            "exception": str(e),
        }


# ------------------------------
# Entrypoint (example)
# ------------------------------
if __name__ == "__main__":
    # Example simultaneous usage:
    #   asyncio.gather(
    #       suggestion_generation("userA", "input_userA.json", "config.yaml", groq_api_key="..."),
    #       suggestion_generation("userB", "input_userB.json", "config.yaml", groq_api_key="..."),
    #   )
    result = asyncio.run(
        suggestion_generation(
            user_id="demo_user",
            file_path="input.json",
            yaml_path="config.yaml",
            groq_api_key="YOUR_GROQ_API_KEY",
            persist_dir="./chroma_data",
            reset_index=False,
            top_k=5,
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
