"""
Optimized RAG Implementation with Enhanced Chunking Strategy
Based on BPI Product Data Analysis
"""
from pathlib import Path
import os
import yaml
import json
import asyncio
from typing import Any, Dict, Optional, List

from sqlmodel import Field, Session, SQLModel, create_engine, select

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.readers.file import PDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import MetadataMode
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("api_key")
model = os.getenv("model", "moonshotai/kimi-k2-instruct")


class OptimizedChunkingStrategy:
    """
    Optimized chunking strategy based on BPI PDF analysis
    """
    
    def __init__(self, embed_model):
        self.embed_model = embed_model
        
    def get_banking_optimized_parser(self, strategy: str = "balanced"):
        """
        Get optimized node parser based on strategy
        
        Strategies:
        - precision: Small chunks for precise Q&A (512 chars)
        - balanced: Medium chunks for balanced performance (1024 chars) 
        - context: Large chunks for context-heavy tasks (1536 chars)
        - semantic: Content-aware semantic chunking
        - hierarchical: Section-aware hierarchical chunking
        """
        
        if strategy == "precision":
            return SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50,
                paragraph_separator="\n\n",
                secondary_chunking_regex=r"\d+\.\d+\s+[A-Z]",  # Match section headers
            )
            
        elif strategy == "balanced":
            return SentenceSplitter(
                chunk_size=1024,
                chunk_overlap=100,
                paragraph_separator="\n\n",
                secondary_chunking_regex=r"\d+\.\d+\s+[A-Z]",
            )
            
        elif strategy == "context":
            return SentenceSplitter(
                chunk_size=1536,
                chunk_overlap=150,
                paragraph_separator="\n\n",
                secondary_chunking_regex=r"\d+\.\d+\s+[A-Z]",
            )
            
        elif strategy == "semantic":
            return SemanticSplitterNodeParser(
                embed_model=self.embed_model,
                buffer_size=1,
                breakpoint_percentile_threshold=95
            )
            
        elif strategy == "hierarchical":
            return HierarchicalNodeParser.from_defaults(
                chunk_sizes=[2048, 1024, 512],
                chunk_overlap=100
            )
            
        else:
            # Default to balanced
            return self.get_banking_optimized_parser("balanced")
    
    def create_ingestion_pipeline(self, strategy: str = "balanced"):
        """Create optimized ingestion pipeline with metadata extraction"""
        
        node_parser = self.get_banking_optimized_parser(strategy)
        
        # Create pipeline with metadata extractors for banking content
        pipeline = IngestionPipeline(
            transformations=[
                node_parser,
                TitleExtractor(nodes=5, llm=None),  # Extract titles from context
                # QuestionsAnsweredExtractor(questions=3, llm=None),  # Extract key questions
            ]
        )
        
        return pipeline


# Load Prompt from YAML
def load_prompt(yaml_path: str) -> str:
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    prompt = config.get("prompt", "").strip()
    if not prompt:
        raise ValueError("`prompt` missing or empty in YAML config.")
    return prompt


async def optimized_suggestion_generation(
    user_id: str,
    yaml_path: str,
    groq_api_key: Optional[str] = None,
    vector_store_path: str = "./rag_store",
    reset_index: bool = False,
    top_k: int = 5,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    conversation_context: str = "",
    chunking_strategy: str = "balanced",
    similarity_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Optimized RAG with enhanced chunking for BPI banking data
    
    Args:
        user_id: Unique user identifier
        yaml_path: Path to prompt configuration
        groq_api_key: API key for Groq
        vector_store_path: Path for vector store persistence
        reset_index: Whether to rebuild the index
        top_k: Number of similar chunks to retrieve
        embedding_model: HuggingFace embedding model
        conversation_context: Previous conversation context
        chunking_strategy: Chunking strategy ('precision', 'balanced', 'context', 'semantic', 'hierarchical')
        similarity_threshold: Minimum similarity threshold for retrieval
    """
    
    try:
        # Initialize LLM with optimized parameters
        llm = Groq(
            model=model,
            api_key=groq_api_key,
            temperature=0.1,
            max_tokens=800
        )
        
        # Initialize embedding model with caching
        current_dir = Path(__file__).parent
        server_dir = current_dir.parent
        model_cache_dir = server_dir / "models"
        model_cache_dir.mkdir(exist_ok=True)
        
        embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            cache_folder=str(model_cache_dir)
        )
        
        # Set global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        # Load BPI PDF with error handling
        pdf_file_path = server_dir / "BPI" / "BPI Product Data for RAG_.pdf"
        
        if not pdf_file_path.exists():
            return {
                "error": "BPI PDF file not found",
                "expected_path": str(pdf_file_path),
                "detail": "Please ensure the BPI Product Data for RAG_.pdf file exists"
            }
        
        loader = PDFReader()
        documents = loader.load_data(pdf_file_path)
        
        # Enhanced vector store with strategy-specific naming
        vector_store_file = f"{vector_store_path}_{user_id}_{chunking_strategy}.json"
        
        if reset_index and os.path.exists(vector_store_file):
            os.remove(vector_store_file)
        
        vector_store = SimpleVectorStore()
        
        # Load existing vector store if available
        if os.path.exists(vector_store_file):
            vector_store = SimpleVectorStore.from_persist_path(vector_store_file)
            print(f"📚 Loaded existing vector store: {chunking_strategy}")
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Apply optimized chunking strategy
        chunking = OptimizedChunkingStrategy(embed_model)
        
        if not os.path.exists(vector_store_file):
            print(f"🔧 Building index with {chunking_strategy} chunking strategy...")
            
            if chunking_strategy == "hierarchical":
                # Special handling for hierarchical chunking
                pipeline = chunking.create_ingestion_pipeline(chunking_strategy)
                nodes = pipeline.run(documents=documents)
                # Get leaf nodes for indexing
                leaf_nodes = get_leaf_nodes(nodes)
                index = VectorStoreIndex(
                    leaf_nodes,
                    storage_context=storage_context,
                    embed_model=embed_model,
                )
            else:
                # Standard chunking approaches
                pipeline = chunking.create_ingestion_pipeline(chunking_strategy)
                nodes = pipeline.run(documents=documents)
                
                # Add banking-specific metadata
                for i, node in enumerate(nodes):
                    # Extract section information from content
                    content = node.get_content(metadata_mode=MetadataMode.NONE)
                    
                    # Detect product categories
                    if any(term in content.lower() for term in ['savings', 'deposit', 'account']):
                        node.metadata['product_category'] = 'deposits'
                    elif any(term in content.lower() for term in ['credit', 'card', 'loan']):
                        node.metadata['product_category'] = 'credit'
                    elif any(term in content.lower() for term in ['investment', 'wealth', 'fund']):
                        node.metadata['product_category'] = 'investments'
                    else:
                        node.metadata['product_category'] = 'general'
                    
                    # Extract section numbers
                    import re
                    section_match = re.search(r'(\d+\.\d+)', content[:100])
                    if section_match:
                        node.metadata['section'] = section_match.group(1)
                    
                    node.metadata['chunk_strategy'] = chunking_strategy
                    node.metadata['chunk_index'] = i
                
                index = VectorStoreIndex(
                    nodes,
                    storage_context=storage_context,
                    embed_model=embed_model,
                )
        else:
            # Load existing index
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=embed_model,
            )
        
        # Persist the vector store
        vector_store.persist(persist_path=vector_store_file)
        
        # Load and optimize prompt
        prompt = load_prompt(yaml_path)
        
        # Add context with length optimization
        if conversation_context:
            context_preview = conversation_context[:400] if len(conversation_context) > 400 else conversation_context
            prompt = f"{prompt}\n\nRECENT CONTEXT: {context_preview}\n\nUse this context for personalized responses."
        
        # Add banking-specific instructions
        prompt += f"""

BANKING OPTIMIZATION INSTRUCTIONS:
- Focus on specific BPI products and services mentioned in the retrieved context
- Provide accurate financial details (fees, requirements, benefits)
- Use section numbers when referencing specific information
- If information is not in the retrieved context, clearly state limitations
- Prioritize recent and relevant product information
"""
        
        # Configure optimized query engine
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            llm=llm,
            response_mode="compact",
            node_postprocessors=[],  # Can add custom postprocessors here
        )
        
        # Execute query with timing
        import time
        start_time = time.time()
        response = query_engine.query(prompt)
        query_time = time.time() - start_time
        
        # Parse and return response with metadata
        try:
            raw = (response.response or "").strip()
            if raw.startswith("```"):
                raw = raw.strip("`").lstrip("json").strip()
            parsed = json.loads(raw)
            
            # Add optimization metadata
            parsed["_metadata"] = {
                "chunking_strategy": chunking_strategy,
                "query_time_seconds": round(query_time, 2),
                "chunks_retrieved": len(response.source_nodes) if hasattr(response, 'source_nodes') else 0,
                "embedding_model": embedding_model,
                "vector_store_path": vector_store_file
            }
            
            return parsed
            
        except Exception as e:
            return {
                "error": "Failed to parse LLM response as JSON",
                "detail": response.response if hasattr(response, "response") else str(response),
                "exception": str(e),
                "chunking_strategy": chunking_strategy,
                "query_time_seconds": round(query_time, 2)
            }
    
    except Exception as e:
        return {
            "error": "RAG processing failed",
            "exception": str(e),
            "chunking_strategy": chunking_strategy
        }


# Wrapper function for backward compatibility
async def suggestion_generation(
    user_id: str,
    yaml_path: str,
    groq_api_key: Optional[str] = None,
    vector_store_path: str = "./rag_store",
    reset_index: bool = False,
    top_k: int = 3,  # Reduced from 5 based on optimization
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    conversation_context: str = "",
) -> Dict[str, Any]:
    """
    Backward compatible wrapper using optimized balanced chunking
    """
    return await optimized_suggestion_generation(
        user_id=user_id,
        yaml_path=yaml_path,
        groq_api_key=groq_api_key,
        vector_store_path=vector_store_path,
        reset_index=reset_index,
        top_k=top_k,
        embedding_model=embedding_model,
        conversation_context=conversation_context,
        chunking_strategy="balanced"  # Use optimized balanced strategy by default
    )


# Utility function to test different chunking strategies
async def compare_chunking_strategies(
    user_id: str,
    yaml_path: str,
    groq_api_key: str,
    test_query: str = "What savings accounts does BPI offer?",
    strategies: List[str] = None
) -> Dict[str, Any]:
    """
    Compare different chunking strategies for performance analysis
    """
    
    if strategies is None:
        strategies = ["precision", "balanced", "context", "semantic"]
    
    results = {}
    
    for strategy in strategies:
        print(f"🧪 Testing strategy: {strategy}")
        
        result = await optimized_suggestion_generation(
            user_id=f"{user_id}_test",
            yaml_path=yaml_path,
            groq_api_key=groq_api_key,
            chunking_strategy=strategy,
            reset_index=True,  # Force rebuild for fair comparison
            vector_store_path="./rag_store_test"
        )
        
        results[strategy] = {
            "success": "error" not in result,
            "response_quality": len(result.get("detail", "")) if "error" not in result else 0,
            "metadata": result.get("_metadata", {}),
            "sample_response": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
        }
    
    return results
