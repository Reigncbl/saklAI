from pathlib import Path
import os
import yaml
import json
import asyncio
from typing import Any, Dict, Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select


# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.readers.json import JSONReader  # ✅ correct import path
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.vector_stores import SimpleVectorStore
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
    vector_store_path: str = "./rag_store",
    reset_index: bool = False,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Build/Query a per-user file-based vector store and return JSON suggestions.

    Args:
        user_id: unique identifier for the user/session (e.g., "u_123").
        file_path: path to JSON data to index (same format you used with JSONReader).
        yaml_path: path to YAML config containing `prompt`.
        groq_api_key: your Groq API key (or set via env var GROQ_API_KEY).
        vector_store_path: base path for vector store files (will append user_id).
        reset_index: if True, clears the user's vector store before re-indexing.
        top_k: retrieval depth.

    Returns:
        dict parsed from model JSON, or error dict.
    """

    # --- LLM (Groq) + Embeddings (HuggingFace) ---
    llm = Groq(model=model, api_key=groq_api_key)
    
    # Store embedding model locally in server folder (dynamic path)
    current_dir = Path(__file__).parent  # services folder
    server_dir = current_dir.parent      # server folder
    model_cache_dir = server_dir / "models"
    model_cache_dir.mkdir(exist_ok=True)
    
    embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-large",
        cache_folder=str(model_cache_dir)
    )

    # --- Load documents ---
    loader = JSONReader()
    documents = loader.load_data(Path(file_path))

    # --- Simple Vector Store with file persistence ---
    vector_store_file = f"{vector_store_path}_{user_id}.json"
    
    if reset_index and os.path.exists(vector_store_file):
        os.remove(vector_store_file)
    
    vector_store = SimpleVectorStore()
    
    # Load existing data if available
    if os.path.exists(vector_store_file):
        vector_store = SimpleVectorStore.from_persist_path(vector_store_file)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # --- Build Index ---
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        llm=llm,
        embed_model=embed_model,
    )
    
    # Persist the vector store
    vector_store.persist(persist_path=vector_store_file)

    # --- Query with YAML Prompt ---
    prompt = load_prompt(yaml_path)
    query_engine = index.as_query_engine(similarity_top_k=top_k, llm=llm)
    response = query_engine.query(prompt)

    # --- Parse JSON output ---
    try:
        raw = (response.response or "").strip()
        if raw.startswith("```"):
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

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("api_key")
model = os.getenv("model")

if __name__ == "__main__":
    # Example usage - use dynamic paths to find existing files
    current_dir = Path(__file__).parent  # services folder
    server_dir = current_dir.parent      # server folder
    project_dir = server_dir.parent      # project root folder
    
    # Look for conversation data in project root
    test_json_file = project_dir / "conversation_data.json"
    # Look for config in Prompts folder
    test_yaml_file = server_dir / "Prompts" / "config.yaml"
    
    if test_json_file.exists() and test_yaml_file.exists():
        print(f"Found JSON file: {test_json_file}")
        print(f"Found YAML file: {test_yaml_file}")
        print("Running RAG suggestion generation...\n")
        
        result = asyncio.run(
            suggestion_generation(
                user_id="demo_user",
                file_path=str(test_json_file),
                yaml_path=str(test_yaml_file),
                groq_api_key=api_key,
                vector_store_path="./rag_store",
                reset_index=True,
                top_k=5,
            )
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Test files not found. Expected locations:")
        print(f"- JSON file: {test_json_file}")
        print(f"- YAML file: {test_yaml_file}")
        print(f"\nJSON exists: {test_json_file.exists()}")
        print(f"YAML exists: {test_yaml_file.exists()}")
        print("\nExample usage:")
        print("result = await suggestion_generation(")
        print("    user_id='your_user_id',")
        print("    file_path='your_data.json',")
        print("    yaml_path='your_prompt.yaml',")
        print("    groq_api_key='your_api_key'")
        print(")")
