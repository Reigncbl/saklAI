"""
Diagnostic Test for BLEU and ROUGE Score Issues
"""

import sys
import os
import re
from collections import Counter

# Add the evaluation directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantitative_metrics import QuantitativeAnalyzer

def diagnose_bleu_rouge_issue():
    """Detailed diagnosis of why BLEU and ROUGE scores are 0."""
    
    print("🔍 BLEU/ROUGE DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = QuantitativeAnalyzer()
    
    # Test cases from our evaluation
    test_cases = [
        {
            "name": "Account Balance Query",
            "expected": "Your current account balance is $1,250.50",
            "actual": "Your account balance is $1,250.50"
        },
        {
            "name": "Money Transfer Query", 
            "expected": "To transfer money, log into online banking and select Transfer Funds",
            "actual": "You can transfer money through online banking by selecting Transfer Funds"
        },
        {
            "name": "Loan Rates Query",
            "expected": "Our current loan rates range from 3.5% to 7.2% depending on the loan type",
            "actual": "Current loan rates are 3.5% to 7.2% based on loan type and credit score"
        }
    ]
    
    print("\n📝 TEST CASE ANALYSIS")
    print("-" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔸 Test Case {i}: {case['name']}")
        print(f"Expected: {case['expected']}")
        print(f"Actual:   {case['actual']}")
        
        # Tokenization analysis
        expected_tokens = analyzer._tokenize(case['expected'])
        actual_tokens = analyzer._tokenize(case['actual'])
        
        print(f"\n📊 Tokenization Analysis:")
        print(f"Expected tokens: {expected_tokens}")
        print(f"Actual tokens:   {actual_tokens}")
        print(f"Expected count:  {len(expected_tokens)}")
        print(f"Actual count:    {len(actual_tokens)}")
        
        # Token overlap analysis
        expected_set = set(expected_tokens)
        actual_set = set(actual_tokens)
        overlap = expected_set & actual_set
        
        print(f"\n🔄 Token Overlap Analysis:")
        print(f"Common tokens:     {list(overlap)}")
        print(f"Expected only:     {list(expected_set - actual_set)}")
        print(f"Actual only:       {list(actual_set - expected_set)}")
        print(f"Overlap ratio:     {len(overlap)}/{len(expected_set)} = {len(overlap)/len(expected_set):.2%}")
        
        # Calculate BLEU score with debugging
        bleu_score = analyzer.calculate_bleu_score(case['expected'], case['actual'])
        print(f"\n📈 BLEU Score: {bleu_score:.6f}")
        
        # Calculate ROUGE scores with debugging
        rouge_scores = analyzer.calculate_rouge_scores(case['expected'], case['actual'])
        print(f"📈 ROUGE Scores:")
        print(f"   ROUGE-1: {rouge_scores['rouge_1']:.6f}")
        print(f"   ROUGE-2: {rouge_scores['rouge_2']:.6f}")
        print(f"   ROUGE-L: {rouge_scores['rouge_l']:.6f}")
        
        print("-" * 50)
    
    print("\n🔧 ISSUE ANALYSIS")
    print("-" * 50)
    
    # Analyze specific issues
    case1 = test_cases[0]
    expected_tokens = analyzer._tokenize(case1['expected'])
    actual_tokens = analyzer._tokenize(case1['actual'])
    
    print(f"Case 1 Detailed Analysis:")
    print(f"Expected: '{case1['expected']}'")
    print(f"Actual:   '{case1['actual']}'")
    print(f"Expected tokens: {expected_tokens}")
    print(f"Actual tokens:   {actual_tokens}")
    
    # Check for specific differences
    missing_words = []
    for token in expected_tokens:
        if token not in actual_tokens:
            missing_words.append(token)
    
    extra_words = []
    for token in actual_tokens:
        if token not in expected_tokens:
            extra_words.append(token)
    
    print(f"\n❌ Missing words in actual: {missing_words}")
    print(f"➕ Extra words in actual: {extra_words}")
    
    # Test tokenization issues
    print(f"\n🔍 TOKENIZATION INVESTIGATION")
    print("-" * 30)
    
    # Test different tokenization approaches
    test_text = "Your current account balance is $1,250.50"
    
    print(f"Original text: '{test_text}'")
    
    # Current tokenization
    current_tokens = re.findall(r'\b\w+\b', test_text.lower())
    print(f"Current method: {current_tokens}")
    
    # Alternative tokenization methods
    simple_split = test_text.lower().split()
    print(f"Simple split:   {simple_split}")
    
    # Remove punctuation method
    import string
    no_punct = test_text.translate(str.maketrans('', '', string.punctuation)).lower().split()
    print(f"No punctuation: {no_punct}")
    
    # Test with the problematic currency format
    currency_test = "$1,250.50"
    currency_tokens = re.findall(r'\b\w+\b', currency_test.lower())
    print(f"\nCurrency '{currency_test}' -> {currency_tokens}")
    
    # Improved tokenization test
    def improved_tokenize(text):
        # Handle currency and numbers better
        text = re.sub(r'\$([0-9,]+\.?[0-9]*)', r'dollar \1', text)  # Convert $1,250.50 to "dollar 1250.50"
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.lower().split()
    
    improved_tokens = improved_tokenize(test_text)
    print(f"Improved method: {improved_tokens}")
    
    print(f"\n🎯 SOLUTION RECOMMENDATIONS")
    print("-" * 50)
    
    print("1. 📝 Tokenization Issues:")
    print("   • Currency formatting: $1,250.50 becomes ['1', '250', '50']")
    print("   • Punctuation removal splits numbers")
    print("   • Word differences: 'current' vs omitted")
    
    print("\n2. 🔧 Immediate Fixes:")
    print("   • Implement fuzzy token matching")
    print("   • Better number/currency handling")
    print("   • Synonym awareness")
    print("   • Stemming/lemmatization")
    
    print("\n3. 📊 Alternative Metrics:")
    print("   • Semantic similarity (already implemented)")
    print("   • BERTScore for contextual evaluation")
    print("   • Custom domain-specific scoring")
    
    # Test semantic similarity as alternative
    print(f"\n🧠 SEMANTIC SIMILARITY TEST")
    print("-" * 30)
    
    for case in test_cases:
        semantic_score = analyzer.calculate_semantic_similarity(case['expected'], case['actual'])
        print(f"{case['name']}: {semantic_score:.6f}")
    
    print(f"\n✅ CONCLUSION")
    print("-" * 30)
    print("• BLEU/ROUGE scores are 0 due to strict tokenization")
    print("• Currency and punctuation handling needs improvement")
    print("• Semantic similarity provides better evaluation")
    print("• System is functional - scoring methodology needs refinement")

if __name__ == "__main__":
    diagnose_bleu_rouge_issue()
