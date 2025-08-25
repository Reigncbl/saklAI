#!/usr/bin/env python3
"""
SaklAI Performance Benchmark Script

This script performs load testing and performance benchmarking for:
1. RAG system under concurrent loads
2. Translation service performance 
3. Classification speed and accuracy
4. Memory usage and resource consumption

Usage:
    python performance_benchmark.py [--concurrent-users 10] [--duration 60] [--output benchmark_results.json]
"""

import asyncio
import json
import time
import sys
import os
import psutil
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import concurrent.futures

# Optional imports for Unix systems
try:
    import resource
except ImportError:
    resource = None  # Not available on Windows

# Add server path for imports
current_dir = Path(__file__).parent
server_dir = current_dir.parent / "server"
sys.path.append(str(server_dir))

from services.rag import suggestion_generation
from services.translation_service import translate_to_english
from services.classification_service import classify_with_langchain_agent
from dto.models import SuggestionRequest
from business.rag_processor import process_rag_suggestion

from dotenv import load_dotenv
load_dotenv()


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation"""
    operation_type: str
    response_time: float
    success: bool
    memory_usage_mb: float
    cpu_percent: float
    error_message: Optional[str] = None
    timestamp: str = ""


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark"""
    test_name: str
    duration_seconds: float
    concurrent_users: int
    total_operations: int
    successful_operations: int
    failed_operations: int
    operations_per_second: float
    average_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    average_memory_usage_mb: float
    peak_memory_usage_mb: float
    average_cpu_percent: float
    error_rate: float
    errors: List[str]


class PerformanceBenchmark:
    """Performance benchmarking class for SaklAI"""
    
    def __init__(self):
        self.api_key = os.getenv("api_key")
        if not self.api_key:
            raise ValueError("API key not found. Please set 'api_key' environment variable.")
        
        self.process = psutil.Process()
        self.metrics: List[PerformanceMetrics] = []
    
    def get_system_metrics(self) -> tuple:
        """Get current system metrics"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
        cpu_percent = self.process.cpu_percent()
        return memory_mb, cpu_percent
    
    async def benchmark_translation(self, duration_seconds: int, concurrent_users: int) -> BenchmarkResult:
        """Benchmark translation service performance"""
        print(f"🔄 Starting translation benchmark: {concurrent_users} users for {duration_seconds}s")
        
        test_messages = [
            "Paano mag-open ng savings account?",
            "Pwede ba mag-apply ng credit card online?",
            "Ano ang requirements para sa personal loan?",
            "Magkano ang minimum balance?",
            "May online banking ba kayo?",
            "Paano mag-transfer ng pera sa ibang banka?",
            "Ano ang interest rate ng time deposit?",
            "Pwede ba mag-withdraw anytime?",
            "May ATM ba sa malapit?",
            "Paano mag-register ng mobile banking?"
        ]
        
        start_time = time.time()
        operations = []
        
        async def worker():
            """Worker function for concurrent testing"""
            while time.time() - start_time < duration_seconds:
                message = test_messages[len(operations) % len(test_messages)]
                
                operation_start = time.time()
                memory_before, cpu_before = self.get_system_metrics()
                
                try:
                    result = translate_to_english(message, self.api_key)
                    success = len(result) > 0
                    error_msg = None
                except Exception as e:
                    success = False
                    error_msg = str(e)
                
                response_time = time.time() - operation_start
                memory_after, cpu_after = self.get_system_metrics()
                
                metric = PerformanceMetrics(
                    operation_type="translation",
                    response_time=response_time,
                    success=success,
                    memory_usage_mb=memory_after,
                    cpu_percent=cpu_after,
                    error_message=error_msg,
                    timestamp=datetime.now().isoformat()
                )
                operations.append(metric)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
        
        # Run concurrent workers
        tasks = [worker() for _ in range(concurrent_users)]
        await asyncio.gather(*tasks)
        
        return self._calculate_benchmark_result("Translation", operations, duration_seconds, concurrent_users)
    
    async def benchmark_classification(self, duration_seconds: int, concurrent_users: int) -> BenchmarkResult:
        """Benchmark classification service performance"""
        print(f"🔍 Starting classification benchmark: {concurrent_users} users for {duration_seconds}s")
        
        test_messages = [
            "What are the requirements for opening a savings account?",
            "How can I apply for a credit card?",
            "What loans do you offer for small businesses?",
            "How do I send money to Philippines?",
            "Can you help me with online banking?",
            "Hello, how are you today?",
            "Where is the nearest branch?",
            "What is your customer service number?",
            "I need help with my account balance",
            "What are your banking hours?"
        ]
        
        start_time = time.time()
        operations = []
        
        async def worker():
            """Worker function for concurrent testing"""
            while time.time() - start_time < duration_seconds:
                message = test_messages[len(operations) % len(test_messages)]
                
                operation_start = time.time()
                memory_before, cpu_before = self.get_system_metrics()
                
                try:
                    result = await classify_with_langchain_agent(message, self.api_key)
                    success = "template" in result and "category" in result
                    error_msg = None
                except Exception as e:
                    success = False
                    error_msg = str(e)
                
                response_time = time.time() - operation_start
                memory_after, cpu_after = self.get_system_metrics()
                
                metric = PerformanceMetrics(
                    operation_type="classification",
                    response_time=response_time,
                    success=success,
                    memory_usage_mb=memory_after,
                    cpu_percent=cpu_after,
                    error_message=error_msg,
                    timestamp=datetime.now().isoformat()
                )
                operations.append(metric)
                
                await asyncio.sleep(0.2)  # Slightly longer delay for LLM calls
        
        # Run concurrent workers
        tasks = [worker() for _ in range(concurrent_users)]
        await asyncio.gather(*tasks)
        
        return self._calculate_benchmark_result("Classification", operations, duration_seconds, concurrent_users)
    
    async def benchmark_rag(self, duration_seconds: int, concurrent_users: int) -> BenchmarkResult:
        """Benchmark RAG system performance"""
        print(f"🧠 Starting RAG benchmark: {concurrent_users} users for {duration_seconds}s")
        
        test_messages = [
            "What are the benefits of opening a savings account with BPI?",
            "How long does credit card approval take?",
            "What are the requirements for a business loan?",
            "What are the fees for sending money abroad?",
            "How do I download and set up the BPI mobile app?",
            "What documents do I need for account opening?",
            "What is the minimum balance for savings accounts?",
            "How can I check my account balance online?",
            "What are the interest rates for time deposits?",
            "How do I apply for a personal loan?"
        ]
        
        start_time = time.time()
        operations = []
        user_counter = 0
        
        async def worker():
            """Worker function for concurrent testing"""
            nonlocal user_counter
            worker_user_id = f"bench_user_{user_counter}"
            user_counter += 1
            
            while time.time() - start_time < duration_seconds:
                message = test_messages[len(operations) % len(test_messages)]
                
                operation_start = time.time()
                memory_before, cpu_before = self.get_system_metrics()
                
                try:
                    request = SuggestionRequest(
                        user_id=worker_user_id,
                        message=message,
                        prompt_type="auto"
                    )
                    result = await process_rag_suggestion(request)
                    success = result.get("status") == "success"
                    error_msg = None
                except Exception as e:
                    success = False
                    error_msg = str(e)
                
                response_time = time.time() - operation_start
                memory_after, cpu_after = self.get_system_metrics()
                
                metric = PerformanceMetrics(
                    operation_type="rag",
                    response_time=response_time,
                    success=success,
                    memory_usage_mb=memory_after,
                    cpu_percent=cpu_after,
                    error_message=error_msg,
                    timestamp=datetime.now().isoformat()
                )
                operations.append(metric)
                
                await asyncio.sleep(0.5)  # Longer delay for RAG operations
        
        # Run concurrent workers
        tasks = [worker() for _ in range(concurrent_users)]
        await asyncio.gather(*tasks)
        
        return self._calculate_benchmark_result("RAG", operations, duration_seconds, concurrent_users)
    
    async def benchmark_end_to_end(self, duration_seconds: int, concurrent_users: int) -> BenchmarkResult:
        """Benchmark end-to-end performance with mixed workloads"""
        print(f"🔄 Starting end-to-end benchmark: {concurrent_users} users for {duration_seconds}s")
        
        test_cases = [
            ("Paano mag-open ng savings account?", "translation_rag"),
            ("What are credit card requirements?", "classification_rag"),
            ("Kumusta ka?", "translation_direct"),
            ("Ano ang minimum balance sa savings?", "translation_rag"),
            ("Where is your nearest branch?", "classification_direct"),
            ("Pwede ba mag-apply ng loan online?", "translation_rag"),
            ("Good morning!", "classification_direct"),
            ("Magkano ang fees sa remittance?", "translation_rag"),
            ("How do I reset my password?", "classification_rag"),
            ("May promo ba kayo ngayon?", "translation_rag")
        ]
        
        start_time = time.time()
        operations = []
        user_counter = 0
        
        async def worker():
            """Worker function for concurrent mixed testing"""
            nonlocal user_counter
            worker_user_id = f"e2e_user_{user_counter}"
            user_counter += 1
            
            while time.time() - start_time < duration_seconds:
                message, expected_flow = test_cases[len(operations) % len(test_cases)]
                
                operation_start = time.time()
                memory_before, cpu_before = self.get_system_metrics()
                
                try:
                    request = SuggestionRequest(
                        user_id=worker_user_id,
                        message=message,
                        prompt_type="auto"
                    )
                    result = await process_rag_suggestion(request)
                    success = result.get("status") == "success"
                    error_msg = None
                except Exception as e:
                    success = False
                    error_msg = str(e)
                
                response_time = time.time() - operation_start
                memory_after, cpu_after = self.get_system_metrics()
                
                metric = PerformanceMetrics(
                    operation_type="end_to_end",
                    response_time=response_time,
                    success=success,
                    memory_usage_mb=memory_after,
                    cpu_percent=cpu_after,
                    error_message=error_msg,
                    timestamp=datetime.now().isoformat()
                )
                operations.append(metric)
                
                await asyncio.sleep(0.3)
        
        # Run concurrent workers
        tasks = [worker() for _ in range(concurrent_users)]
        await asyncio.gather(*tasks)
        
        return self._calculate_benchmark_result("End-to-End", operations, duration_seconds, concurrent_users)
    
    def _calculate_benchmark_result(self, test_name: str, operations: List[PerformanceMetrics], 
                                  duration: float, concurrent_users: int) -> BenchmarkResult:
        """Calculate benchmark results from operation metrics"""
        if not operations:
            return BenchmarkResult(
                test_name=test_name,
                duration_seconds=duration,
                concurrent_users=concurrent_users,
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
                operations_per_second=0.0,
                average_response_time=0.0,
                min_response_time=0.0,
                max_response_time=0.0,
                p95_response_time=0.0,
                p99_response_time=0.0,
                average_memory_usage_mb=0.0,
                peak_memory_usage_mb=0.0,
                average_cpu_percent=0.0,
                error_rate=1.0,
                errors=["No operations completed"]
            )
        
        successful_ops = [op for op in operations if op.success]
        failed_ops = [op for op in operations if not op.success]
        
        response_times = [op.response_time for op in operations]
        memory_usage = [op.memory_usage_mb for op in operations]
        cpu_usage = [op.cpu_percent for op in operations]
        
        # Calculate percentiles
        response_times_sorted = sorted(response_times)
        p95_idx = int(0.95 * len(response_times_sorted))
        p99_idx = int(0.99 * len(response_times_sorted))
        
        return BenchmarkResult(
            test_name=test_name,
            duration_seconds=duration,
            concurrent_users=concurrent_users,
            total_operations=len(operations),
            successful_operations=len(successful_ops),
            failed_operations=len(failed_ops),
            operations_per_second=len(operations) / duration,
            average_response_time=statistics.mean(response_times),
            min_response_time=min(response_times),
            max_response_time=max(response_times),
            p95_response_time=response_times_sorted[p95_idx] if p95_idx < len(response_times_sorted) else 0,
            p99_response_time=response_times_sorted[p99_idx] if p99_idx < len(response_times_sorted) else 0,
            average_memory_usage_mb=statistics.mean(memory_usage),
            peak_memory_usage_mb=max(memory_usage),
            average_cpu_percent=statistics.mean(cpu_usage),
            error_rate=len(failed_ops) / len(operations),
            errors=[op.error_message for op in failed_ops if op.error_message][:10]  # First 10 errors
        )
    
    def print_benchmark_result(self, result: BenchmarkResult):
        """Print benchmark results in a formatted way"""
        print(f"\n📊 {result.test_name} Benchmark Results")
        print("=" * 60)
        print(f"Duration: {result.duration_seconds}s")
        print(f"Concurrent Users: {result.concurrent_users}")
        print(f"Total Operations: {result.total_operations}")
        print(f"Successful: {result.successful_operations}")
        print(f"Failed: {result.failed_operations}")
        print(f"Success Rate: {(1 - result.error_rate):.1%}")
        print(f"Operations/Second: {result.operations_per_second:.2f}")
        
        print(f"\n⏱️  Response Times:")
        print(f"  Average: {result.average_response_time:.3f}s")
        print(f"  Min: {result.min_response_time:.3f}s")
        print(f"  Max: {result.max_response_time:.3f}s")
        print(f"  95th Percentile: {result.p95_response_time:.3f}s")
        print(f"  99th Percentile: {result.p99_response_time:.3f}s")
        
        print(f"\n💾 Resource Usage:")
        print(f"  Average Memory: {result.average_memory_usage_mb:.1f} MB")
        print(f"  Peak Memory: {result.peak_memory_usage_mb:.1f} MB")
        print(f"  Average CPU: {result.average_cpu_percent:.1f}%")
        
        if result.errors:
            print(f"\n❌ Sample Errors ({len(result.errors)}):")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"  - {error}")
    
    async def run_full_benchmark(self, duration: int = 60, concurrent_users: int = 5) -> Dict[str, BenchmarkResult]:
        """Run full performance benchmark suite"""
        print("🚀 Starting SaklAI Performance Benchmark Suite")
        print(f"Configuration: {concurrent_users} concurrent users, {duration}s duration per test")
        print("=" * 80)
        
        results = {}
        
        # Run individual benchmarks
        tests = [
            ("translation", self.benchmark_translation),
            ("classification", self.benchmark_classification),
            ("rag", self.benchmark_rag),
            ("end_to_end", self.benchmark_end_to_end)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🧪 Running {test_name} benchmark...")
            try:
                result = await test_func(duration, concurrent_users)
                results[test_name] = result
                self.print_benchmark_result(result)
            except Exception as e:
                print(f"❌ {test_name} benchmark failed: {e}")
                continue
        
        return results
    
    def save_benchmark_results(self, results: Dict[str, BenchmarkResult], output_path: str):
        """Save benchmark results to JSON file"""
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": os.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3)
            },
            "benchmark_results": {name: asdict(result) for name, result in results.items()}
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Benchmark results saved to: {output_file}")


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SaklAI Performance Benchmark")
    parser.add_argument("--concurrent-users", type=int, default=5, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--output", default="evaluation/benchmark_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    try:
        benchmark = PerformanceBenchmark()
        results = await benchmark.run_full_benchmark(args.duration, args.concurrent_users)
        benchmark.save_benchmark_results(results, args.output)
        
        print("\n🎉 Performance benchmark completed!")
        
        # Print summary
        print("\n📈 SUMMARY:")
        print("-" * 40)
        for test_name, result in results.items():
            success_rate = (1 - result.error_rate) * 100
            print(f"{test_name.upper()}: {success_rate:.1f}% success, {result.operations_per_second:.1f} ops/s")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
