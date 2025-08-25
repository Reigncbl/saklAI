#!/usr/bin/env python3
"""
SaklAI Evaluation Runner

This script provides a unified interface to run all evaluation tasks:
1. Basic functionality tests
2. Comprehensive test suites 
3. Performance benchmarks
4. Generate reports

Usage:
    python run_evaluation.py [--mode all|functional|performance] [--config config.json]
"""

import asyncio
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Import evaluation modules
from evaluation_script import SaklAIEvaluator
from performance_benchmark import PerformanceBenchmark
from quantitative_metrics import generate_quantitative_report


class EvaluationRunner:
    """Main evaluation runner class"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.results_dir = Path("evaluation/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    async def run_functional_tests(self) -> dict:
        """Run functional evaluation tests"""
        print("🧪 Running Functional Tests...")
        print("=" * 50)
        
        evaluator = SaklAIEvaluator(self.config_path)
        summary = await evaluator.run_evaluation()
        
        # Save results
        results_file = self.results_dir / f"functional_results_{self.timestamp}.json"
        evaluator.save_results(str(results_file))
        
        # Export CSV
        csv_file = self.results_dir / f"functional_results_{self.timestamp}.csv"
        evaluator.export_to_csv(str(csv_file))
        
        return {
            "type": "functional",
            "summary": summary,
            "results_file": str(results_file),
            "csv_file": str(csv_file)
        }
    
    async def run_performance_tests(self, duration: int = 60, concurrent_users: int = 5) -> dict:
        """Run performance benchmark tests"""
        print("⚡ Running Performance Tests...")
        print("=" * 50)
        
        benchmark = PerformanceBenchmark()
        results = await benchmark.run_full_benchmark(duration, concurrent_users)
        
        # Save results
        results_file = self.results_dir / f"performance_results_{self.timestamp}.json"
        benchmark.save_benchmark_results(results, str(results_file))
        
        return {
            "type": "performance", 
            "results": results,
            "results_file": str(results_file)
        }
    
    def load_extended_test_cases(self) -> dict:
        """Load extended test cases from JSON file"""
        test_cases_file = Path("evaluation/test_cases.json")
        if test_cases_file.exists():
            with open(test_cases_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    async def run_comprehensive_tests(self) -> dict:
        """Run comprehensive tests using extended test cases"""
        print("🔬 Running Comprehensive Tests...")
        print("=" * 50)
        
        # Load extended test cases
        extended_tests = self.load_extended_test_cases()
        
        if not extended_tests:
            print("⚠️  No extended test cases found, running basic tests only")
            return await self.run_functional_tests()
        
        # Create evaluator with extended test cases
        evaluator = SaklAIEvaluator(self.config_path)
        
        # Override test cases with extended ones
        extended_test_list = []
        for suite_name, test_cases in extended_tests.get("test_suites", {}).items():
            for test_case in test_cases:
                # Convert dict to TestCase object format
                from evaluation_script import TestCase
                extended_test_list.append(TestCase(
                    id=test_case["id"],
                    input_text=test_case["input_text"],
                    expected_category=test_case.get("expected_category"),
                    expected_template=test_case.get("expected_template"),
                    expected_translation=test_case.get("expected_translation"),
                    language=test_case.get("language", "en"),
                    test_type=test_case["test_type"]
                ))
        
        # Replace default test cases with extended ones
        original_get_test_cases = evaluator.get_test_cases
        evaluator.get_test_cases = lambda: extended_test_list
        
        # Run evaluation
        summary = await evaluator.run_evaluation()
        
        # Save results
        results_file = self.results_dir / f"comprehensive_results_{self.timestamp}.json"
        evaluator.save_results(str(results_file))
        
        # Export CSV
        csv_file = self.results_dir / f"comprehensive_results_{self.timestamp}.csv"
        evaluator.export_to_csv(str(csv_file))
        
        return {
            "type": "comprehensive",
            "summary": summary,
            "results_file": str(results_file),
            "csv_file": str(csv_file),
            "test_suites_run": list(extended_tests.get("test_suites", {}).keys())
        }
    
    def generate_report(self, functional_results: dict = None, performance_results: dict = None) -> str:
        """Generate a comprehensive evaluation report"""
        report_file = self.results_dir / f"evaluation_report_{self.timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# SaklAI Evaluation Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            if functional_results:
                summary = functional_results["summary"]
                f.write(f"- **Functional Tests**: {summary.success_rate:.1%} success rate ({summary.successful_tests}/{summary.total_tests} tests)\n")
                f.write(f"- **Average Response Time**: {summary.average_response_time:.2f}s\n")
                
                if hasattr(summary, 'classification_accuracy') and summary.classification_accuracy:
                    f.write(f"- **Classification Accuracy**: {summary.classification_accuracy:.1%}\n")
                
                if hasattr(summary, 'translation_accuracy') and summary.translation_accuracy:
                    f.write(f"- **Translation Accuracy**: {summary.translation_accuracy:.1%}\n")
                
                # Add quantitative metrics summary
                if hasattr(summary, 'quantitative_metrics') and summary.quantitative_metrics:
                    qm = summary.quantitative_metrics
                    f.write(f"- **BLEU Score**: {qm.bleu_score:.3f}\n")
                    f.write(f"- **Semantic Similarity**: {qm.semantic_similarity_score:.3f}\n")
                    f.write(f"- **System Reliability**: {qm.system_reliability:.1%}\n")
            
            if performance_results:
                f.write(f"\n### Performance Highlights\n")
                for test_name, result in performance_results["results"].items():
                    success_rate = (1 - result.error_rate) * 100
                    f.write(f"- **{test_name.title()}**: {success_rate:.1f}% success, {result.operations_per_second:.1f} ops/s\n")
            
            # Detailed Results
            if functional_results:
                f.write(f"\n## Functional Test Results\n\n")
                summary = functional_results["summary"]
                
                f.write(f"### Overall Metrics\n")
                f.write(f"- Total Tests: {summary.total_tests}\n")
                f.write(f"- Successful: {summary.successful_tests}\n")
                f.write(f"- Failed: {summary.failed_tests}\n")
                f.write(f"- Success Rate: {summary.success_rate:.1%}\n")
                f.write(f"- Average Response Time: {summary.average_response_time:.3f}s\n\n")
                
                # Add comprehensive quantitative metrics section
                if hasattr(summary, 'quantitative_metrics') and summary.quantitative_metrics:
                    qm = summary.quantitative_metrics
                    f.write(f"### Quantitative Quality Metrics\n\n")
                    f.write(f"| Metric | Score | Interpretation |\n")
                    f.write(f"|--------|-------|----------------|\n")
                    f.write(f"| BLEU Score | {qm.bleu_score:.3f} | Translation/Generation Quality |\n")
                    f.write(f"| ROUGE-1 | {qm.rouge_1_score:.3f} | Unigram Overlap |\n")
                    f.write(f"| ROUGE-2 | {qm.rouge_2_score:.3f} | Bigram Overlap |\n")
                    f.write(f"| ROUGE-L | {qm.rouge_l_score:.3f} | Longest Common Subsequence |\n")
                    f.write(f"| Semantic Similarity | {qm.semantic_similarity_score:.3f} | Meaning Preservation |\n")
                    f.write(f"| Lexical Diversity | {qm.lexical_diversity:.3f} | Vocabulary Richness |\n")
                    f.write(f"| Classification F1 | {qm.classification_f1:.3f} | Classification Quality |\n\n")
                    
                    f.write(f"### Performance Distribution\n\n")
                    f.write(f"| Metric | Value |\n")
                    f.write(f"|--------|-------|\n")
                    f.write(f"| Average Response Time | {qm.avg_response_time:.2f}s |\n")
                    f.write(f"| Median Response Time | {qm.median_response_time:.2f}s |\n")
                    f.write(f"| 95th Percentile | {qm.p95_response_time:.2f}s |\n")
                    f.write(f"| 99th Percentile | {qm.p99_response_time:.2f}s |\n")
                    f.write(f"| Standard Deviation | {qm.standard_deviation:.2f}s |\n\n")
                    
                    f.write(f"### Statistical Analysis\n\n")
                    f.write(f"- **Confidence Interval (95%)**: [{qm.confidence_interval_lower:.3f}, {qm.confidence_interval_upper:.3f}]\n")
                    f.write(f"- **Task Completion Rate**: {qm.task_completion_rate:.1%}\n")
                    f.write(f"- **System Reliability**: {qm.system_reliability:.1%}\n")
                    f.write(f"- **Error Rate**: {qm.error_rate:.1%}\n\n")
                    
                    # Generate and append quantitative report
                    try:
                        # Convert summary results to format expected by quantitative report
                        analysis_data = []
                        if 'evaluator' in functional_results:
                            evaluator = functional_results['evaluator']
                            for result in evaluator.results:
                                analysis_record = {
                                    "test_id": result.test_id,
                                    "test_type": result.test_type,
                                    "expected_output": result.expected_output or "",
                                    "actual_output": result.actual_output,
                                    "response_time": result.response_time,
                                    "success": result.success,
                                    "expected_category": getattr(result, 'expected_category', None),
                                    "predicted_category": getattr(result, 'predicted_category', None),
                                    "similarity_score": result.similarity_score
                                }
                                analysis_data.append(analysis_record)
                        
                        quantitative_report = generate_quantitative_report(qm, analysis_data)
                        f.write("\n## Detailed Quantitative Analysis\n\n")
                        # Skip the title and first lines from quantitative report to avoid duplication
                        report_lines = quantitative_report.split('\n')[4:]  # Skip title and first summary
                        f.write('\n'.join(report_lines))
                        f.write("\n\n")
                    except Exception as e:
                        f.write(f"*Note: Could not generate detailed quantitative analysis: {e}*\n\n")
                
                f.write(f"### Test Type Breakdown\n")
                for test_type, stats in summary.test_type_breakdown.items():
                    f.write(f"**{test_type.upper()}**:\n")
                    f.write(f"- Tests: {stats['successful']}/{stats['total']} ({stats['success_rate']:.1%})\n")
                    f.write(f"- Avg Time: {stats['average_response_time']:.3f}s\n\n")
            
            if performance_results:
                f.write(f"## Performance Test Results\n\n")
                
                for test_name, result in performance_results["results"].items():
                    f.write(f"### {test_name.title()} Performance\n")
                    f.write(f"- Duration: {result.duration_seconds}s\n")
                    f.write(f"- Concurrent Users: {result.concurrent_users}\n")
                    f.write(f"- Total Operations: {result.total_operations}\n")
                    f.write(f"- Success Rate: {(1-result.error_rate):.1%}\n")
                    f.write(f"- Operations/Second: {result.operations_per_second:.2f}\n")
                    f.write(f"- Average Response Time: {result.average_response_time:.3f}s\n")
                    f.write(f"- 95th Percentile: {result.p95_response_time:.3f}s\n")
                    f.write(f"- Memory Usage: {result.average_memory_usage_mb:.1f} MB (peak: {result.peak_memory_usage_mb:.1f} MB)\n\n")
            
            # Enhanced Recommendations based on quantitative metrics
            f.write("## Recommendations\n\n")
            
            if functional_results and hasattr(functional_results['summary'], 'quantitative_metrics'):
                summary = functional_results["summary"]
                qm = summary.quantitative_metrics
                
                if summary.success_rate < 0.9:
                    f.write("⚠️ **Functional Issues**:\n")
                    f.write("- Success rate below 90%, investigate failing test cases\n")
                    f.write("- Review error logs for common failure patterns\n\n")
                
                if qm and qm.avg_response_time > 5.0:
                    f.write("⚠️ **Performance Issues**:\n")
                    f.write(f"- Average response time {qm.avg_response_time:.2f}s above 5 seconds\n")
                    f.write("- Consider optimizing RAG retrieval and LLM calls\n\n")
                
                if qm and qm.semantic_similarity_score < 0.6:
                    f.write("⚠️ **Quality Issues**:\n")
                    f.write(f"- Semantic similarity {qm.semantic_similarity_score:.3f} below 0.6 threshold\n")
                    f.write("- Review translation accuracy and response relevance\n\n")
                
                if qm and qm.classification_f1 < 0.8:
                    f.write("⚠️ **Classification Issues**:\n")
                    f.write(f"- F1 score {qm.classification_f1:.3f} below 0.8 threshold\n")
                    f.write("- Improve classification model training or category definitions\n\n")
                
                if qm and qm.system_reliability < 0.95:
                    f.write("⚠️ **Reliability Issues**:\n")
                    f.write(f"- System reliability {qm.system_reliability:.1%} below 95%\n")
                    f.write("- Enhance error handling and system robustness\n\n")
                
                # Positive recommendations
                if qm:
                    f.write("✅ **Strengths**:\n")
                    if qm.semantic_similarity_score > 0.8:
                        f.write("- Excellent semantic similarity preservation\n")
                    if qm.classification_f1 > 0.9:
                        f.write("- High-quality classification performance\n")
                    if qm.avg_response_time < 2.0:
                        f.write("- Fast response times\n")
                    if qm.system_reliability > 0.95:
                        f.write("- High system reliability\n")
                    f.write("\n")
            
            if performance_results:
                for test_name, result in performance_results["results"].items():
                    if result.error_rate > 0.1:
                        f.write(f"⚠️ **{test_name.title()} Issues**:\n")
                        f.write(f"- Error rate above 10% under load\n")
                        f.write(f"- Review system capacity and error handling\n\n")
            
            f.write("## Files Generated\n\n")
            if functional_results:
                f.write(f"- Functional Results: `{functional_results['results_file']}`\n")
                f.write(f"- Functional CSV: `{functional_results['csv_file']}`\n")
            
            if performance_results:
                f.write(f"- Performance Results: `{performance_results['results_file']}`\n")
            
            f.write(f"- This Report: `{report_file}`\n")
        
        return str(report_file)
    
    async def run_all(self, duration: int = 60, concurrent_users: int = 5, mode: str = "all") -> dict:
        """Run all evaluation tasks"""
        print("🚀 SaklAI Evaluation Suite")
        print(f"Timestamp: {self.timestamp}")
        print("=" * 60)
        
        results = {"timestamp": self.timestamp}
        
        if mode in ["all", "functional", "comprehensive"]:
            if mode == "comprehensive":
                functional_results = await self.run_comprehensive_tests()
            else:
                functional_results = await self.run_functional_tests()
            results["functional"] = functional_results
        else:
            functional_results = None
        
        if mode in ["all", "performance"]:
            performance_results = await self.run_performance_tests(duration, concurrent_users)
            results["performance"] = performance_results
        else:
            performance_results = None
        
        # Generate report
        report_file = self.generate_report(functional_results, performance_results)
        results["report_file"] = report_file
        
        print(f"\n🎉 Evaluation completed!")
        print(f"📄 Report: {report_file}")
        
        return results


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SaklAI Evaluation Runner")
    parser.add_argument("--mode", choices=["all", "functional", "performance", "comprehensive"], 
                       default="all", help="Evaluation mode")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--duration", type=int, default=60, help="Performance test duration (seconds)")
    parser.add_argument("--concurrent-users", type=int, default=5, help="Concurrent users for performance tests")
    
    args = parser.parse_args()
    
    try:
        runner = EvaluationRunner(args.config)
        results = await runner.run_all(args.duration, args.concurrent_users, args.mode)
        
        print(f"\n✅ All evaluations completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Evaluation interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
