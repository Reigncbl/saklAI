#!/usr/bin/env python3
"""
Test script for quantitative metrics module
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from quantitative_metrics import QuantitativeAnalyzer, generate_quantitative_report

def test_quantitative_metrics():
    """Test the quantitative metrics functionality"""
    print("🧪 Testing Quantitative Metrics Module...")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = QuantitativeAnalyzer()
    
    # Test data
    sample_results = [
        {
            "test_id": "test_001",
            "test_type": "translation",
            "expected_output": "How to open a savings account?",
            "actual_output": "How can I open a savings account?",
            "response_time": 1.5,
            "success": True,
            "expected_category": "banking",
            "predicted_category": "banking"
        },
        {
            "test_id": "test_002", 
            "test_type": "translation",
            "expected_output": "Credit card application process",
            "actual_output": "Process for applying for credit cards",
            "response_time": 2.1,
            "success": True,
            "expected_category": "credit",
            "predicted_category": "credit"
        },
        {
            "test_id": "test_003",
            "test_type": "classification",
            "expected_output": "Template: savings.yaml, Category: banking",
            "actual_output": "Template: savings.yaml, Category: banking",
            "response_time": 0.8,
            "success": True,
            "expected_category": "banking",
            "predicted_category": "banking"
        }
    ]
    
    # Test individual metrics
    print("\n📊 Testing Individual Metrics:")
    print("-" * 30)
    
    # BLEU Score
    bleu = analyzer.calculate_bleu_score(
        "How to open a savings account?",
        "How can I open a savings account?"
    )
    print(f"BLEU Score: {bleu:.3f}")
    
    # ROUGE Scores
    rouge = analyzer.calculate_rouge_scores(
        "How to open a savings account?",
        "How can I open a savings account?"
    )
    print(f"ROUGE-1: {rouge['rouge_1']:.3f}")
    print(f"ROUGE-2: {rouge['rouge_2']:.3f}")
    print(f"ROUGE-L: {rouge['rouge_l']:.3f}")
    
    # Semantic Similarity
    semantic_sim = analyzer.calculate_semantic_similarity(
        "How to open a savings account?",
        "How can I open a savings account?"
    )
    print(f"Semantic Similarity: {semantic_sim:.3f}")
    
    # Performance Metrics
    response_times = [1.5, 2.1, 0.8, 1.2, 3.0]
    perf_metrics = analyzer.calculate_performance_metrics(response_times)
    print(f"Avg Response Time: {perf_metrics['avg_response_time']:.2f}s")
    print(f"95th Percentile: {perf_metrics['p95_response_time']:.2f}s")
    
    # Classification Metrics
    y_true = ["banking", "credit", "banking"]
    y_pred = ["banking", "credit", "banking"]
    class_metrics = analyzer.calculate_classification_metrics(y_true, y_pred)
    print(f"Classification F1: {class_metrics['f1_score']:.3f}")
    
    # Test comprehensive analysis
    print("\n📈 Testing Comprehensive Analysis:")
    print("-" * 40)
    
    metrics = analyzer.analyze_evaluation_results(sample_results)
    print(f"Overall BLEU: {metrics.bleu_score:.3f}")
    print(f"Overall Semantic Similarity: {metrics.semantic_similarity_score:.3f}")
    print(f"Task Completion Rate: {metrics.task_completion_rate:.1%}")
    print(f"System Reliability: {metrics.system_reliability:.1%}")
    
    # Test report generation
    print("\n📄 Testing Report Generation:")
    print("-" * 35)
    
    report = generate_quantitative_report(metrics, sample_results)
    print("✅ Report generated successfully")
    print(f"Report length: {len(report)} characters")
    
    # Print first few lines of report
    print("\nReport preview:")
    print("-" * 20)
    report_lines = report.split('\n')[:15]
    for line in report_lines:
        print(line)
    
    print("\n✅ All tests passed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_quantitative_metrics()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
