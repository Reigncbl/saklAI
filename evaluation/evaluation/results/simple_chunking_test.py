"""
Simple test for optimized RAG chunking
"""
import asyncio
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Add server to path
sys.path.append('server')

# Load environment
load_dotenv()

from services.rag_optimized import optimized_suggestion_generation

async def test_chunking():
    """Test optimized chunking strategy"""
    
    print("🚀 Testing Optimized RAG Chunking")
    print("=" * 40)
    
    # Configuration
    user_id = "test_chunking"
    yaml_path = "server/Prompts/savings_accounts.yaml"
    api_key = os.getenv("api_key")
    
    if not api_key:
        print("❌ API key not found")
        return
    
    if not Path(yaml_path).exists():
        print(f"❌ YAML file not found: {yaml_path}")
        return
    
    # Test different chunking strategies
    strategies = ["precision", "balanced", "context"]
    
    for strategy in strategies:
        print(f"\n🔍 Testing {strategy} strategy...")
        
        try:
            result = await optimized_suggestion_generation(
                user_id=f"test_{strategy}",
                yaml_path=yaml_path,
                groq_api_key=api_key,
                chunking_strategy=strategy,
                reset_index=True,
                top_k=3,
                conversation_context="User wants information about BPI savings accounts"
            )
            
            if "error" not in result:
                metadata = result.get("_metadata", {})
                print(f"   ✅ Success: {strategy}")
                print(f"   ⏱️  Time: {metadata.get('query_time_seconds', 'N/A')}s")
                print(f"   📊 Chunks: {metadata.get('chunks_retrieved', 'N/A')}")
                print(f"   🧠 Strategy: {metadata.get('chunking_strategy', 'N/A')}")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
    
    print("\n✅ Chunking test complete!")

if __name__ == "__main__":
    asyncio.run(test_chunking())
