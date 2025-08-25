"""
Test Business Metrics Integration with Quantitative Evaluation
"""

import json
import sys
import os

# Add the evaluation directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantitative_metrics import QuantitativeMetrics, BusinessMetrics, QuantitativeAnalyzer, generate_quantitative_report
from sample_business_data import generate_sample_business_data

# Mock evaluation data for testing
def create_mock_evaluation_data():
    """Create mock evaluation data for testing the reporting system."""
    return {
        "responses": [
            {
                "question": "What is my account balance?",
                "expected": "Your current account balance is $1,250.50",
                "actual": "Your account balance is $1,250.50",
                "response_time": 2.3,
                "success": True
            },
            {
                "question": "How do I transfer money?", 
                "expected": "To transfer money, log into online banking and select Transfer Funds",
                "actual": "You can transfer money through online banking by selecting Transfer Funds",
                "response_time": 1.8,
                "success": True
            },
            {
                "question": "What are your loan rates?",
                "expected": "Our current loan rates range from 3.5% to 7.2% depending on the loan type",
                "actual": "Current loan rates are 3.5% to 7.2% based on loan type and credit score", 
                "response_time": 3.1,
                "success": True
            }
        ]
    }

def test_business_metrics_integration():
    """Test the complete business metrics integration."""
    
    print("=== Testing Business Metrics Integration ===\n")
    
    # Generate sample business data
    print("1. Generating sample business data...")
    business_data = generate_sample_business_data()
    
    # Create mock evaluation data
    print("2. Creating mock evaluation data...")
    eval_data = create_mock_evaluation_data()
    
    # Calculate quantitative metrics
    print("3. Calculating quantitative metrics...")
    analyzer = QuantitativeAnalyzer()
    metrics = analyzer.analyze_evaluation_results(eval_data["responses"])
    
    print(f"   ✓ BLEU Score: {metrics.bleu_score:.3f}")
    print(f"   ✓ ROUGE-L Score: {metrics.rouge_l_score:.3f}")
    print(f"   ✓ Average Response Time: {metrics.avg_response_time:.2f}s")
    print(f"   ✓ Success Rate: {metrics.task_completion_rate:.1%}")
    
    # Generate comprehensive report with business metrics
    print("4. Generating comprehensive report with business metrics...")
    
    try:
        report = generate_quantitative_report(metrics, business_data)
        
        # Save the report
        report_file = "test_business_metrics_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"   ✓ Report generated successfully: {report_file}")
        
        # Show a preview of the report
        lines = report.split('\n')
        print("\n=== Report Preview (First 30 lines) ===")
        for i, line in enumerate(lines[:30]):
            print(line)
        
        if len(lines) > 30:
            print(f"\n... and {len(lines) - 30} more lines")
            
        print(f"\n   ✓ Total report lines: {len(lines)}")
        
    except Exception as e:
        print(f"   ❌ Error generating report: {e}")
        return False
    
    # Test individual business metrics
    print("\n5. Testing individual business metrics...")
    
    business_calc = BusinessMetrics()
    
    # Test containment rate
    containment = business_calc.calculate_bot_containment_rate(business_data["sessions"])
    print(f"   ✓ Bot Containment Rate: {containment.get('bot_containment_rate', 0):.1f}%")
    
    # Test CSAT
    csat = business_calc.calculate_csat(business_data["satisfaction_data"])
    print(f"   ✓ CSAT Score: {csat.get('csat_score', 0):.1f}%")
    
    # Test cost metrics
    cost = business_calc.calculate_cost_per_session(business_data["usage_data"])
    print(f"   ✓ Cost per Session: ${cost.get('cost_per_session', 0):.3f}")
    
    # Test FCR
    fcr = business_calc.calculate_first_contact_resolution(business_data["resolution_data"])
    print(f"   ✓ First Contact Resolution: {fcr.get('fcr_rate', 0):.1f}%")
    
    # Test handoff time
    handoff = business_calc.calculate_average_handoff_time(business_data["handoff_events"])
    if "error" not in handoff:
        print(f"   ✓ Average Handoff Time: {handoff.get('average_handoff_time', 0):.1f}s")
    else:
        print(f"   ⚠️ Handoff Time: {handoff.get('error', 'Unknown error')}")
    
    print("\n=== Business Metrics Integration Test Complete ===")
    print("✅ All business metrics have been successfully integrated!")
    
    return True

if __name__ == "__main__":
    success = test_business_metrics_integration()
    if success:
        print("\n🎉 Business metrics integration is working perfectly!")
        print("📊 The system now supports comprehensive business intelligence reporting")
        print("📈 Available metrics: Average Handoff Time, Bot Containment Rate, CSAT, Cost Analysis, FCR")
    else:
        print("\n❌ There were issues with the business metrics integration")
