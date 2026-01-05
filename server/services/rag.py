from pathlib import Path
import os
import yaml
import json
import asyncio
from typing import Any, Dict, Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select


# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.readers.file import PDFReader  # Updated to use PDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("api_key")
model = os.getenv("model", "moonshotai/kimi-k2-instruct")  # Default to kimi-k2-instruct


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
    yaml_path: str,
    groq_api_key: Optional[str] = None,
    vector_store_path: str = "./rag_store",
    reset_index: bool = False,
    top_k: int = 5,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    conversation_context: str = "",
) -> Dict[str, Any]:
    """
    Build/Query a per-user PDF-based vector store and return JSON suggestions.

    Args:
        user_id: unique identifier for the user/session (e.g., "u_123").
        yaml_path: path to YAML config containing `prompt`.
        groq_api_key: your Groq API key (or set via env var GROQ_API_KEY).
        vector_store_path: base path for vector store files (will append user_id).
        reset_index: if True, clears the user's vector store before re-indexing.
        top_k: retrieval depth.
        embedding_model: HuggingFace embedding model name. Options:
            - "sentence-transformers/all-MiniLM-L6-v2" (fastest, English-optimized)
            - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (fast, multilingual)
            - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" (slow, best quality)
        conversation_context: previous conversation context to provide continuity.

    Returns:
        dict parsed from model JSON, or error dict.
    """

    # --- LLM (Groq) + Embeddings (HuggingFace) ---
    llm = Groq(
        model=model, 
        api_key=groq_api_key,
        temperature=0.1,  # Lower temperature for more consistent responses
        max_tokens=800    # Reduced token limit for faster responses
    )
    
    # Store embedding model locally in server folder (dynamic path)
    current_dir = Path(__file__).parent  # services folder
    server_dir = current_dir.parent      # server folder
    model_cache_dir = server_dir / "models"
    model_cache_dir.mkdir(exist_ok=True)
    
    embed_model = HuggingFaceEmbedding(
        model_name=embedding_model,
        cache_folder=str(model_cache_dir)
    )

    # Set global settings for consistency
    Settings.llm = llm
    Settings.embed_model = embed_model

    # --- Load documents from BPI PDF ---
    # Define the path to the BPI PDF file
    current_dir = Path(__file__).parent  # services folder
    server_dir = current_dir.parent      # server folder
    pdf_file_path = server_dir / "BPI" / "BPI Product Data for RAG_.pdf"
    
    if not pdf_file_path.exists():
        return {
            "error": "BPI PDF file not found",
            "expected_path": str(pdf_file_path),
            "detail": "Please ensure the BPI Product Data for RAG_.pdf file exists in the server/BPI folder"
        }
    
    loader = PDFReader()
    documents = loader.load_data(pdf_file_path)

    # --- Optimized Vector Store with enhanced chunking ---
    vector_store_file = f"{vector_store_path}_{user_id}.json"
    
    if reset_index and os.path.exists(vector_store_file):
        os.remove(vector_store_file)
    
    vector_store = SimpleVectorStore()
    
    # Load existing data if available
    if os.path.exists(vector_store_file):
        vector_store = SimpleVectorStore.from_persist_path(vector_store_file)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # --- Build Index with Optimized Chunking ---
    if not os.path.exists(vector_store_file):
        print("🔧 Building optimized index with enhanced chunking...")
        
        # Use optimized sentence splitter for banking content
        node_parser = SentenceSplitter(
            chunk_size=1024,  # Balanced size based on analysis
            chunk_overlap=100,  # Good context preservation
            paragraph_separator="\n\n",
            secondary_chunking_regex=r"\d+\.\d+\s+[A-Z]",  # Split on section headers
        )
        
        # Parse documents into nodes
        nodes = node_parser.get_nodes_from_documents(documents)
        
        # Add banking-specific metadata for better retrieval
        for i, node in enumerate(nodes):
            content = node.get_content(metadata_mode=MetadataMode.NONE)
            
            # Categorize content by banking product type
            if any(term in content.lower() for term in ['savings', 'deposit', 'account']):
                node.metadata['product_category'] = 'deposits'
            elif any(term in content.lower() for term in ['credit', 'card', 'loan']):
                node.metadata['product_category'] = 'credit'
            elif any(term in content.lower() for term in ['investment', 'wealth', 'fund']):
                node.metadata['product_category'] = 'investments'
            else:
                node.metadata['product_category'] = 'general'
            
            # Extract section numbers for reference
            import re
            section_match = re.search(r'(\d+\.\d+)', content[:100])
            if section_match:
                node.metadata['section'] = section_match.group(1)
            
            node.metadata['chunk_index'] = i
            node.metadata['optimized'] = True
        
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
        )
    else:
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
        )
    
    # Persist the vector store
    vector_store.persist(persist_path=vector_store_file)

    # --- Query with YAML Prompt ---
    prompt = load_prompt(yaml_path)
    
    # Optimize prompt with conversation context for better performance
    if conversation_context:
        # Truncate context to prevent token overflow and improve speed
        context_preview = conversation_context[:500] if len(conversation_context) > 500 else conversation_context
        prompt = f"{prompt}\n\nRECENT CONTEXT: {context_preview}\n\nConsider this context for personalized responses."
    
    # Add performance optimization instructions with banking focus
    prompt += """

BANKING RAG OPTIMIZATION:
- Reference specific BPI products, fees, and requirements from retrieved context
- Use section numbers when available for precise information sourcing
- Focus on accurate financial details and eligibility criteria
- If information isn't in context, clearly state limitations
- Prioritize most relevant and recent product information
"""
    
    query_engine = index.as_query_engine(
        similarity_top_k=top_k, 
        llm=llm,
        response_mode="compact"  # Use compact mode for faster responses
    )
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

