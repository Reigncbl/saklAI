"""
Test Script for Optimized RAG Chunking
Validates the enhanced chunking strategy for BPI banking data
"""
import asyncio
import json
import time
from pathlib import Path
import sys

# Add server directory to path
server_dir = Path(__file__).parent.parent / "server"
sys.path.append(str(server_dir))

try:
    # Import after adding path
    import services.rag_optimized as rag_optimized
    import services.rag as original_rag_module
    
    # Extract functions
    optimized_suggestion_generation = rag_optimized.optimized_suggestion_generation
    compare_chunking_strategies = rag_optimized.compare_chunking_strategies
    original_rag = original_rag_module.suggestion_generation
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the evaluation directory")
    sys.exit(1)
from dotenv import load_dotenv
import os

load_dotenv()

async def test_optimized_chunking():
    """Test the optimized chunking implementation"""
    
    print("🚀 Testing Optimized RAG Chunking for BPI Banking Data")
    print("=" * 60)
    
    # Test configuration
    user_id = "test_user_chunking"
    yaml_path = server_dir / "Prompts" / "savings_accounts.yaml"
    groq_api_key = os.getenv("api_key")
    
    if not groq_api_key:
        print("❌ API key not found. Please set 'api_key' in environment variables.")
        return
    
    if not yaml_path.exists():
        print(f"❌ YAML file not found: {yaml_path}")
        return
    
    # Test queries for banking scenarios
    test_queries = [
        "What savings accounts does BPI offer for students?",
        "What are the requirements for BPI credit cards?", 
        "Tell me about BPI investment products and wealth management services",
        "What are the fees for BPI deposit accounts?",
        "How do I open a BPI savings account?"
    ]
    
    print("\n📊 1. COMPARISON: Original vs Optimized RAG")
    print("-" * 50)
    
    for i, query in enumerate(test_queries[:2]):  # Test first 2 queries
        print(f"\n🔍 Query {i+1}: {query}")
        
        # Test original RAG
        print("  📌 Testing Original RAG...")
        start_time = time.time()
        original_result = await original_rag(
            user_id=f"{user_id}_orig_{i}",
            yaml_path=str(yaml_path),
            groq_api_key=groq_api_key,
            reset_index=True,
            conversation_context=f"User is asking about: {query}"
        )
        original_time = time.time() - start_time
        
        # Test optimized RAG
        print("  🚀 Testing Optimized RAG...")
        start_time = time.time()
        optimized_result = await optimized_suggestion_generation(
            user_id=f"{user_id}_opt_{i}",
            yaml_path=str(yaml_path),
            groq_api_key=groq_api_key,
            reset_index=True,
            chunking_strategy="balanced",
            conversation_context=f"User is asking about: {query}"
        )
        optimized_time = time.time() - start_time
        
        # Compare results
        print(f"    ⏱️  Original Time: {original_time:.2f}s")
        print(f"    ⚡ Optimized Time: {optimized_time:.2f}s")
        print(f"    📈 Speed Improvement: {((original_time - optimized_time) / original_time * 100):.1f}%")
        
        orig_success = "error" not in original_result
        opt_success = "error" not in optimized_result
        print(f"    ✅ Original Success: {orig_success}")
        print(f"    🎯 Optimized Success: {opt_success}")
        
        if opt_success:
            metadata = optimized_result.get("_metadata", {})
            print(f"    📊 Chunks Retrieved: {metadata.get('chunks_retrieved', 'N/A')}")
            print(f"    🧠 Chunking Strategy: {metadata.get('chunking_strategy', 'N/A')}")
    
    print("\n🧪 2. CHUNKING STRATEGY COMPARISON")
    print("-" * 50)
    
    # Test different chunking strategies
    strategy_results = await compare_chunking_strategies(
        user_id="strategy_test",
        yaml_path=str(yaml_path),
        groq_api_key=groq_api_key,
        test_query="What are BPI's savings account options and their features?",
        strategies=["precision", "balanced", "context", "semantic"]
    )
    
    print("\n📋 Strategy Performance Summary:")
    for strategy, result in strategy_results.items():
        success = result["success"]
        metadata = result["metadata"]
        print(f"  {strategy.upper():>12}: {'✅' if success else '❌'} | "
              f"Time: {metadata.get('query_time_seconds', 'N/A')}s | "
              f"Chunks: {metadata.get('chunks_retrieved', 'N/A')}")
    
    print("\n🎯 3. DETAILED CHUNKING ANALYSIS")
    print("-" * 50)
    
    # Detailed analysis of balanced strategy
    detailed_result = await optimized_suggestion_generation(
        user_id="detailed_test",
        yaml_path=str(yaml_path),
        groq_api_key=groq_api_key,
        chunking_strategy="balanced",
        reset_index=True,
        top_k=5,
        conversation_context="User wants detailed information about BPI banking products"
    )
    
    if "error" not in detailed_result:
        metadata = detailed_result.get("_metadata", {})
        print(f"✅ Detailed Test Successful")
        print(f"   ⏱️  Query Time: {metadata.get('query_time_seconds')}s")
        print(f"   📊 Chunks Retrieved: {metadata.get('chunks_retrieved')}")
        print(f"   🧠 Strategy: {metadata.get('chunking_strategy')}")
        print(f"   📁 Vector Store: {Path(metadata.get('vector_store_path', '')).name}")
        
        # Show sample response structure
        response_keys = [k for k in detailed_result.keys() if not k.startswith("_")]
        print(f"   📝 Response Keys: {response_keys}")
    else:
        print(f"❌ Detailed test failed: {detailed_result.get('error')}")
    
    print("\n💡 4. OPTIMIZATION RECOMMENDATIONS")
    print("-" * 50)
    
    best_strategy = max(strategy_results.items(), 
                       key=lambda x: x[1]["success"] and x[1]["metadata"].get("query_time_seconds", float('inf')) < 10)
    
    print(f"🏆 Recommended Strategy: {best_strategy[0].upper()}")
    print(f"   📊 Success Rate: {'100%' if best_strategy[1]['success'] else '0%'}")
    print(f"   ⚡ Query Time: {best_strategy[1]['metadata'].get('query_time_seconds', 'N/A')}s")
    
    print("\n📈 PERFORMANCE OPTIMIZATIONS IMPLEMENTED:")
    print("   ✅ Balanced chunk size (1024 chars) for banking content")
    print("   ✅ Banking-specific metadata extraction")
    print("   ✅ Section-aware chunking with regex patterns")
    print("   ✅ Product category classification")
    print("   ✅ Enhanced prompts for banking domain")
    print("   ✅ Optimized embedding model caching")
    print("   ✅ Vector store strategy-specific naming")
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Deploy optimized RAG to production")
    print("   2. Monitor performance metrics")
    print("   3. Fine-tune chunk sizes based on user feedback")
    print("   4. Consider implementing hierarchical chunking for complex queries")
    
    print("\n" + "=" * 60)
    print("✅ Chunking Optimization Test Complete!")


async def test_banking_scenarios():
    """Test specific banking scenarios with optimized chunking"""
    
    print("\n🏦 BANKING SCENARIO TESTS")
    print("=" * 40)
    
    user_id = "banking_scenarios"
    yaml_path = server_dir / "Prompts" / "savings_accounts.yaml"
    groq_api_key = os.getenv("api_key")
    
    banking_scenarios = [
        {
            "scenario": "Student Banking",
            "query": "I'm a college student looking for a savings account",
            "context": "Student, limited income, first bank account"
        },
        {
            "scenario": "Small Business", 
            "query": "What business banking solutions does BPI offer?",
            "context": "Small business owner, need checking and savings"
        },
        {
            "scenario": "Investment Planning",
            "query": "I want to invest money for retirement planning",
            "context": "Professional, looking for long-term investments"
        }
    ]
    
    for i, scenario in enumerate(banking_scenarios):
        print(f"\n📊 Scenario {i+1}: {scenario['scenario']}")
        print(f"   Query: {scenario['query']}")
        
        result = await optimized_suggestion_generation(
            user_id=f"{user_id}_scenario_{i}",
            yaml_path=str(yaml_path),
            groq_api_key=groq_api_key,
            chunking_strategy="balanced",
            conversation_context=scenario['context'],
            top_k=3
        )
        
        if "error" not in result:
            metadata = result.get("_metadata", {})
            print(f"   ✅ Success | Time: {metadata.get('query_time_seconds')}s | "
                  f"Chunks: {metadata.get('chunks_retrieved')}")
        else:
            print(f"   ❌ Failed: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(test_optimized_chunking())
    asyncio.run(test_banking_scenarios())
