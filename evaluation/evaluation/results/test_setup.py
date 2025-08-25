#!/usr/bin/env python3
"""
Quick Setup Test for SaklAI Evaluation

This script verifies that the evaluation environment is properly configured.
Run this before running the full evaluation suite.
"""

import sys
import os
from pathlib import Path

# Add server path
current_dir = Path(__file__).parent
server_dir = current_dir.parent / "server"
sys.path.append(str(server_dir))

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        # Core evaluation imports
        from evaluation_script import SaklAIEvaluator, TestCase, EvaluationResult
        print("✅ Evaluation script imports OK")
        
        from performance_benchmark import PerformanceBenchmark
        print("✅ Performance benchmark imports OK")
        
        # SaklAI service imports
        from services.rag import suggestion_generation
        from services.translation_service import translate_to_english
        from services.classification_service import classify_with_langchain_agent
        print("✅ SaklAI service imports OK")
        
        # Additional dependencies
        import pandas as pd
        import psutil
        print("✅ Additional dependencies OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    print("\n🔧 Testing environment...")
    
    # Check API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("api_key")
    if api_key:
        print("✅ API key found")
    else:
        print("❌ API key not found in environment")
        return False
    
    # Check required directories
    server_dir = current_dir.parent / "server"
    if server_dir.exists():
        print("✅ Server directory found")
    else:
        print("❌ Server directory not found")
        return False
    
    # Check BPI PDF (optional)
    bpi_pdf = server_dir / "BPI" / "BPI Product Data for RAG_.pdf"
    if bpi_pdf.exists():
        print("✅ BPI PDF found")
    else:
        print("⚠️  BPI PDF not found (RAG tests may fail)")
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test translation
        from services.translation_service import translate_to_english
        api_key = os.getenv("api_key")
        
        result = translate_to_english("Hello world", api_key)
        if result:
            print("✅ Translation service working")
        else:
            print("❌ Translation service failed")
            return False
        
        # Test classification (async)
        import asyncio
        from services.classification_service import classify_with_langchain_agent
        
        async def test_classification():
            result = await classify_with_langchain_agent("Hello", api_key)
            return "template" in result
        
        classification_ok = asyncio.run(test_classification())
        if classification_ok:
            print("✅ Classification service working")
        else:
            print("❌ Classification service failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 SaklAI Evaluation Setup Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_imports()
    all_tests_passed &= test_environment()
    all_tests_passed &= test_basic_functionality()
    
    print("\n" + "=" * 50)
    
    if all_tests_passed:
        print("🎉 All tests passed! Evaluation environment is ready.")
        print("\nYou can now run:")
        print("  python run_evaluation.py --mode functional")
        print("  python run_evaluation.py --mode performance")
        print("  python run_evaluation.py --mode all")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing dependencies: pip install -r requirements.txt")
        print("  - Set API key in .env file")
        print("  - Ensure you're running from the evaluation directory")
        return 1

if __name__ == "__main__":
    sys.exit(main())
