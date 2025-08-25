#!/usr/bin/env python3
"""
SaklAI Evaluation Script for RAG and Translation Systems

This script evaluates the performance of:
1. RAG (Retrieval Augmented Generation) system
2. Translation service (Tagalog/Taglish to English)
3. Classification accuracy
4. Response quality metrics

Usage:
    python evaluation_script.py [--config config.json] [--output results.json]
"""

import asyncio
import json
import sys
import os
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

# Import quantitative metrics
from quantitative_metrics import QuantitativeAnalyzer, QuantitativeMetrics, generate_quantitative_report

# Add server path to sys.path for imports
current_dir = Path(__file__).parent
server_dir = current_dir.parent / "server"
sys.path.append(str(server_dir))

# Import SaklAI modules
from services.rag import suggestion_generation
from services.translation_service import translate_to_english
from services.classification_service import classify_with_langchain_agent, should_use_rag
from services.response_service import generate_direct_groq_response
from dto.models import SuggestionRequest
from business.rag_processor import process_rag_suggestion

# Import environment variables
from dotenv import load_dotenv
load_dotenv()


@dataclass
class TestCase:
    """Represents a single test case"""
    id: str
    input_text: str
    expected_category: Optional[str] = None
    expected_template: Optional[str] = None
    expected_translation: Optional[str] = None
    language: str = "en"  # en, tl, taglish
    test_type: str = "general"  # rag, translation, classification


@dataclass
class EvaluationResult:
    """Represents evaluation results for a single test case"""
    test_id: str
    test_type: str
    input_text: str
    expected_output: Optional[str]
    actual_output: str
    classification_result: Optional[Dict] = None
    translation_result: Optional[str] = None
    rag_result: Optional[Dict] = None
    response_time: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    similarity_score: Optional[float] = None
    # Quantitative metrics
    bleu_score: Optional[float] = None
    rouge_scores: Optional[Dict[str, float]] = None
    semantic_similarity: Optional[float] = None
    expected_category: Optional[str] = None
    predicted_category: Optional[str] = None


@dataclass
class EvaluationSummary:
    """Summary of all evaluation results"""
    total_tests: int
    successful_tests: int
    failed_tests: int
    success_rate: float
    average_response_time: float
    test_type_breakdown: Dict[str, Dict]
    classification_accuracy: Optional[float] = None
    translation_accuracy: Optional[float] = None
    # Comprehensive quantitative metrics
    quantitative_metrics: Optional[QuantitativeMetrics] = None


class SaklAIEvaluator:
    """Main evaluation class for SaklAI systems"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.api_key = os.getenv("api_key")
        self.results: List[EvaluationResult] = []
        
        # Initialize quantitative analyzer
        self.quantitative_analyzer = QuantitativeAnalyzer(
            embedding_model=self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
        if not self.api_key:
            raise ValueError("API key not found. Please set 'api_key' environment variable.")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load evaluation configuration"""
        default_config = {
            "vector_store_path": "./rag_store_eval",
            "test_user_id": "eval_user_001",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "top_k": 5,
            "similarity_threshold": 0.7,
            "timeout_seconds": 30
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _extract_clean_translation(self, verbose_translation: str) -> str:
        """Extract clean translation from verbose LLM response"""
        import re
        
        # Look for text in quotes
        quote_patterns = [
            r'"([^"]+)"',  # Text in double quotes
            r"'([^']+)'",  # Text in single quotes
        ]
        
        for pattern in quote_patterns:
            matches = re.findall(pattern, verbose_translation)
            if matches:
                # Return the first substantial match (longer than 10 chars)
                for match in matches:
                    if len(match.strip()) > 10:
                        return match.strip()
        
        # If no quotes found, try to extract from "Translation:" prefix
        if "Translation:" in verbose_translation:
            parts = verbose_translation.split("Translation:", 1)
            if len(parts) > 1:
                # Get everything after "Translation:" and before any parentheses
                translation_part = parts[1].split("(")[0].strip()
                # Remove quotes if present
                translation_part = translation_part.strip('"\'')
                if len(translation_part) > 5:
                    return translation_part
        
        # If still no good match, try to extract the first line after "Here's the translation:"
        if "Here's the translation:" in verbose_translation:
            lines = verbose_translation.split('\n')
            for i, line in enumerate(lines):
                if "Here's the translation:" in line:
                    # Look for the next non-empty line
                    for j in range(i+1, len(lines)):
                        next_line = lines[j].strip()
                        if next_line and not next_line.startswith('('):
                            # Remove quotes
                            next_line = next_line.strip('"\'')
                            if len(next_line) > 5:
                                return next_line
        
        # Fallback: return original if no extraction worked
        return verbose_translation
    
    def get_test_cases(self) -> List[TestCase]:
        """Get predefined test cases for evaluation"""
        return [
            # RAG Test Cases (Banking-related)
            TestCase(
                id="rag_001",
                input_text="What are the requirements for opening a savings account?",
                expected_category="savings",
                expected_template="savings_accounts.yaml",
                test_type="rag"
            ),
            TestCase(
                id="rag_002", 
                input_text="How can I apply for a credit card?",
                expected_category="credit",
                expected_template="credit_cards.yaml",
                test_type="rag"
            ),
            TestCase(
                id="rag_003",
                input_text="What loans do you offer for small businesses?",
                expected_category="loan",
                expected_template="loans.yaml",
                test_type="rag"
            ),
            TestCase(
                id="rag_004",
                input_text="How do I send money to Philippines through remittance?",
                expected_category="remittance",
                expected_template="remittances_ofw.yaml",
                test_type="rag"
            ),
            TestCase(
                id="rag_005",
                input_text="Can you help me with online banking setup?",
                expected_category="digital",
                expected_template="digital_banking.yaml",
                test_type="rag"
            ),
            
            # Translation Test Cases
            TestCase(
                id="trans_001",
                input_text="Paano mag-open ng savings account?",
                expected_translation="How to open a savings account?",
                language="tl",
                test_type="translation"
            ),
            TestCase(
                id="trans_002",
                input_text="Pwede ba mag-apply ng credit card online?",
                expected_translation="Can I apply for a credit card online?",
                language="taglish",
                test_type="translation"
            ),
            TestCase(
                id="trans_003",
                input_text="Ano ang requirements para sa personal loan?",
                expected_translation="What are the requirements for a personal loan?",
                language="tl",
                test_type="translation"
            ),
            TestCase(
                id="trans_004",
                input_text="May online banking ba kayo?",
                expected_translation="Do you have online banking?",
                language="tl",
                test_type="translation"
            ),
            
            # Classification Test Cases
            TestCase(
                id="class_001",
                input_text="Hello, how are you today?",
                expected_template="config.yaml",
                expected_category="general",
                test_type="classification"
            ),
            TestCase(
                id="class_002",
                input_text="I want to know about your branch locations",
                expected_template="general_banking.yaml",
                expected_category="banking",
                test_type="classification"
            ),
            TestCase(
                id="class_003",
                input_text="What's the interest rate for time deposits?",
                expected_template="savings_accounts.yaml",
                expected_category="savings",
                test_type="classification"
            ),
            
            # End-to-end Test Cases
            TestCase(
                id="e2e_001",
                input_text="Magkano ang minimum balance sa savings account?",
                expected_category="savings",
                expected_template="savings_accounts.yaml",
                language="taglish",
                test_type="end_to_end"
            ),
            TestCase(
                id="e2e_002",
                input_text="Paano mag-transfer ng pera sa ibang bansa?",
                expected_category="remittance",
                expected_template="remittances_ofw.yaml",
                language="tl",
                test_type="end_to_end"
            )
        ]
    
    async def evaluate_translation(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate translation functionality"""
        start_time = time.time()
        
        try:
            raw_translated = translate_to_english(test_case.input_text, self.api_key)
            # Extract clean translation from potentially verbose response
            translated = self._extract_clean_translation(raw_translated)
            response_time = time.time() - start_time
            
            # Calculate similarity if expected translation is provided
            similarity_score = None
            bleu_score = None
            rouge_scores = None
            semantic_similarity = None
            
            if test_case.expected_translation:
                similarity_score = self._calculate_similarity(translated, test_case.expected_translation)
                
                # Calculate quantitative metrics
                bleu_score = self.quantitative_analyzer.calculate_bleu_score(
                    test_case.expected_translation, translated
                )
                rouge_scores = self.quantitative_analyzer.calculate_rouge_scores(
                    test_case.expected_translation, translated
                )
                semantic_similarity = self.quantitative_analyzer.calculate_semantic_similarity(
                    test_case.expected_translation, translated
                )
            
            success = (similarity_score is None) or (similarity_score >= self.config["similarity_threshold"])
            
            return EvaluationResult(
                test_id=test_case.id,
                test_type=test_case.test_type,
                input_text=test_case.input_text,
                expected_output=test_case.expected_translation,
                actual_output=translated,
                translation_result=translated,
                response_time=response_time,
                success=success,
                similarity_score=similarity_score,
                bleu_score=bleu_score,
                rouge_scores=rouge_scores,
                semantic_similarity=semantic_similarity
            )
            
        except Exception as e:
            return EvaluationResult(
                test_id=test_case.id,
                test_type=test_case.test_type,
                input_text=test_case.input_text,
                expected_output=test_case.expected_translation,
                actual_output="",
                response_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def evaluate_classification(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate classification functionality"""
        start_time = time.time()
        
        try:
            classification = await classify_with_langchain_agent(test_case.input_text, self.api_key)
            response_time = time.time() - start_time
            
            # Check if classification matches expected results
            template_match = classification.get("template") == test_case.expected_template
            
            # More flexible category matching with synonyms
            category_match = True
            predicted_category = None
            if test_case.expected_category:
                actual_category = classification.get("category", "").lower()
                expected_category = test_case.expected_category.lower()
                predicted_category = actual_category
                
                # Define category synonyms for more flexible matching
                category_synonyms = {
                    "general": ["config", "general", "conversation", "greeting"],
                    "banking": ["banking", "general_banking", "bank"],
                    "savings": ["savings", "savings_account", "account", "banking_product"],
                    "credit": ["credit", "credit_card", "card"],
                    "loan": ["loan", "loans", "lending"],
                    "remittance": ["remittance", "remittances", "transfer"],
                    "digital": ["digital", "digital_banking", "online"]
                }
                
                # Check direct match first
                category_match = expected_category in actual_category or actual_category in expected_category
                
                # If no direct match, check synonyms
                if not category_match:
                    for synonym_group in category_synonyms.values():
                        if expected_category in synonym_group and actual_category in synonym_group:
                            category_match = True
                            break
            
            success = template_match and category_match
            
            return EvaluationResult(
                test_id=test_case.id,
                test_type=test_case.test_type,
                input_text=test_case.input_text,
                expected_output=f"Template: {test_case.expected_template}, Category: {test_case.expected_category}",
                actual_output=f"Template: {classification.get('template')}, Category: {classification.get('category')}",
                classification_result=classification,
                response_time=response_time,
                success=success,
                expected_category=test_case.expected_category,
                predicted_category=predicted_category
            )
            
        except Exception as e:
            return EvaluationResult(
                test_id=test_case.id,
                test_type=test_case.test_type,
                input_text=test_case.input_text,
                expected_output=f"Template: {test_case.expected_template}, Category: {test_case.expected_category}",
                actual_output="",
                response_time=time.time() - start_time,
                success=False,
                error_message=str(e),
                expected_category=test_case.expected_category,
                predicted_category=None
            )
    
    async def evaluate_rag(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate RAG functionality"""
        start_time = time.time()
        
        try:
            # Create a suggestion request
            request = SuggestionRequest(
                user_id=self.config["test_user_id"],
                message=test_case.input_text,
                prompt_type="auto"
            )
            
            # Process the request through the RAG system
            result = await process_rag_suggestion(request)
            response_time = time.time() - start_time
            
            # Evaluate success based on response structure and content
            success = (
                result.get("status") == "success" and
                "suggestions" in result and
                len(result["suggestions"]) > 0
            )
            
            # Check if correct template was used
            if test_case.expected_template and result.get("template_used"):
                template_match = result["template_used"] == test_case.expected_template
                success = success and template_match
            
            return EvaluationResult(
                test_id=test_case.id,
                test_type=test_case.test_type,
                input_text=test_case.input_text,
                expected_output=test_case.expected_template,
                actual_output=result.get("template_used", ""),
                rag_result=result,
                response_time=response_time,
                success=success
            )
            
        except Exception as e:
            return EvaluationResult(
                test_id=test_case.id,
                test_type=test_case.test_type,
                input_text=test_case.input_text,
                expected_output=test_case.expected_template,
                actual_output="",
                response_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def evaluate_end_to_end(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate end-to-end functionality (translation + classification + RAG)"""
        start_time = time.time()
        
        try:
            # Create a suggestion request
            request = SuggestionRequest(
                user_id=self.config["test_user_id"],
                message=test_case.input_text,
                prompt_type="auto"
            )
            
            # Process through the full pipeline
            result = await process_rag_suggestion(request)
            response_time = time.time() - start_time
            
            # Evaluate success based on multiple criteria
            success = (
                result.get("status") == "success" and
                "suggestions" in result and
                len(result["suggestions"]) > 0
            )
            
            # Check template matching if expected
            if test_case.expected_template:
                template_match = result.get("template_used") == test_case.expected_template
                success = success and template_match
            
            # Check if translation occurred for non-English input
            translation_occurred = result.get("translated_message") is not None
            
            return EvaluationResult(
                test_id=test_case.id,
                test_type=test_case.test_type,
                input_text=test_case.input_text,
                expected_output=test_case.expected_template,
                actual_output=result.get("template_used", ""),
                rag_result=result,
                translation_result=result.get("translated_message"),
                response_time=response_time,
                success=success
            )
            
        except Exception as e:
            return EvaluationResult(
                test_id=test_case.id,
                test_type=test_case.test_type,
                input_text=test_case.input_text,
                expected_output=test_case.expected_template,
                actual_output="",
                response_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using improved approach for translation"""
        try:
            # Normalize texts
            text1_norm = text1.lower().strip()
            text2_norm = text2.lower().strip()
            
            # If texts are identical, return perfect score
            if text1_norm == text2_norm:
                return 1.0
            
            # Tokenize and clean
            tokens1 = set(text1_norm.replace('?', '').replace('.', '').replace(',', '').split())
            tokens2 = set(text2_norm.replace('?', '').replace('.', '').replace(',', '').split())
            
            if not tokens1 and not tokens2:
                return 1.0
            
            if not tokens1 or not tokens2:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            jaccard = len(intersection) / len(union) if union else 0.0
            
            # Calculate percentage of important words matched
            important_words1 = tokens1 - {'a', 'an', 'the', 'is', 'are', 'to', 'of', 'for', 'in', 'on', 'at', 'by'}
            important_words2 = tokens2 - {'a', 'an', 'the', 'is', 'are', 'to', 'of', 'for', 'in', 'on', 'at', 'by'}
            
            if important_words1 and important_words2:
                important_intersection = important_words1.intersection(important_words2)
                important_coverage = len(important_intersection) / max(len(important_words1), len(important_words2))
            else:
                important_coverage = jaccard
            
            # Return the higher of the two scores (more lenient)
            return max(jaccard, important_coverage)
            
        except Exception:
            return 0.0
    
    async def run_evaluation(self) -> EvaluationSummary:
        """Run the complete evaluation suite"""
        print("🔍 Starting SaklAI Evaluation...")
        print("=" * 60)
        
        test_cases = self.get_test_cases()
        self.results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] Running test: {test_case.id} ({test_case.test_type})")
            
            try:
                if test_case.test_type == "translation":
                    result = await self.evaluate_translation(test_case)
                elif test_case.test_type == "classification":
                    result = await self.evaluate_classification(test_case)
                elif test_case.test_type == "rag":
                    result = await self.evaluate_rag(test_case)
                elif test_case.test_type == "end_to_end":
                    result = await self.evaluate_end_to_end(test_case)
                else:
                    raise ValueError(f"Unknown test type: {test_case.test_type}")
                
                self.results.append(result)
                
                # Print result
                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"    {status} - {result.response_time:.2f}s")
                
                if not result.success and result.error_message:
                    print(f"    Error: {result.error_message}")
                
            except Exception as e:
                print(f"    ❌ FAIL - Unexpected error: {e}")
                
                # Create error result
                error_result = EvaluationResult(
                    test_id=test_case.id,
                    test_type=test_case.test_type,
                    input_text=test_case.input_text,
                    expected_output="",
                    actual_output="",
                    success=False,
                    error_message=str(e)
                )
                self.results.append(error_result)
        
        return self._generate_summary()
    
    def _generate_summary(self) -> EvaluationSummary:
        """Generate evaluation summary from results"""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
        
        response_times = [r.response_time for r in self.results if r.response_time > 0]
        average_response_time = statistics.mean(response_times) if response_times else 0.0
        
        # Test type breakdown
        test_type_breakdown = {}
        for test_type in set(r.test_type for r in self.results):
            type_results = [r for r in self.results if r.test_type == test_type]
            type_success = sum(1 for r in type_results if r.success)
            type_total = len(type_results)
            
            test_type_breakdown[test_type] = {
                "total": type_total,
                "successful": type_success,
                "success_rate": type_success / type_total if type_total > 0 else 0.0,
                "average_response_time": statistics.mean([r.response_time for r in type_results if r.response_time > 0]) or 0.0
            }
        
        # Classification accuracy (for classification and e2e tests)
        classification_results = [r for r in self.results if r.test_type in ["classification", "end_to_end"]]
        classification_accuracy = None
        if classification_results:
            classification_success = sum(1 for r in classification_results if r.success)
            classification_accuracy = classification_success / len(classification_results)
        
        # Translation accuracy (for translation and e2e tests)
        translation_results = [r for r in self.results if r.test_type in ["translation", "end_to_end"]]
        translation_accuracy = None
        if translation_results:
            translation_success = sum(1 for r in translation_results if r.success)
            translation_accuracy = translation_success / len(translation_results)
        
        # Calculate comprehensive quantitative metrics
        quantitative_metrics = None
        try:
            # Convert results to format expected by quantitative analyzer
            analysis_data = []
            for result in self.results:
                analysis_record = {
                    "test_id": result.test_id,
                    "test_type": result.test_type,
                    "expected_output": result.expected_output or "",
                    "actual_output": result.actual_output,
                    "response_time": result.response_time,
                    "success": result.success,
                    "expected_category": result.expected_category,
                    "predicted_category": result.predicted_category,
                    "similarity_score": result.similarity_score
                }
                analysis_data.append(analysis_record)
            
            quantitative_metrics = self.quantitative_analyzer.analyze_evaluation_results(analysis_data)
        except Exception as e:
            print(f"Warning: Could not calculate quantitative metrics: {e}")
        
        return EvaluationSummary(
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            average_response_time=average_response_time,
            test_type_breakdown=test_type_breakdown,
            classification_accuracy=classification_accuracy,
            translation_accuracy=translation_accuracy,
            quantitative_metrics=quantitative_metrics
        )
    
    def print_summary(self, summary: EvaluationSummary):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"Total Tests: {summary.total_tests}")
        print(f"Successful: {summary.successful_tests}")
        print(f"Failed: {summary.failed_tests}")
        print(f"Success Rate: {summary.success_rate:.1%}")
        print(f"Average Response Time: {summary.average_response_time:.2f}s")
        
        if summary.classification_accuracy is not None:
            print(f"Classification Accuracy: {summary.classification_accuracy:.1%}")
        
        if summary.translation_accuracy is not None:
            print(f"Translation Accuracy: {summary.translation_accuracy:.1%}")
        
        # Display quantitative metrics if available
        if summary.quantitative_metrics:
            qm = summary.quantitative_metrics
            print(f"\n🎯 QUANTITATIVE METRICS:")
            print("-" * 40)
            print(f"BLEU Score: {qm.bleu_score:.3f}")
            print(f"ROUGE-1: {qm.rouge_1_score:.3f}")
            print(f"ROUGE-2: {qm.rouge_2_score:.3f}")
            print(f"ROUGE-L: {qm.rouge_l_score:.3f}")
            print(f"Semantic Similarity: {qm.semantic_similarity_score:.3f}")
            print(f"Lexical Diversity: {qm.lexical_diversity:.3f}")
            print(f"Classification F1: {qm.classification_f1:.3f}")
            print(f"95% Response Time: {qm.p95_response_time:.2f}s")
            print(f"Confidence Interval: [{qm.confidence_interval_lower:.3f}, {qm.confidence_interval_upper:.3f}]")
        
        print("\n📈 BY TEST TYPE:")
        print("-" * 40)
        
        for test_type, stats in summary.test_type_breakdown.items():
            print(f"{test_type.upper()}:")
            print(f"  Tests: {stats['successful']}/{stats['total']} ({stats['success_rate']:.1%})")
            print(f"  Avg Time: {stats['average_response_time']:.2f}s")
        
        print("\n🔍 DETAILED RESULTS:")
        print("-" * 40)
        
        for result in self.results:
            status = "✅" if result.success else "❌"
            print(f"{status} {result.test_id} ({result.test_type}) - {result.response_time:.2f}s")
            
            # Show additional quantitative metrics for individual results
            if result.bleu_score is not None:
                print(f"   BLEU: {result.bleu_score:.3f}")
            if result.semantic_similarity is not None:
                print(f"   Semantic Sim: {result.semantic_similarity:.3f}")
            
            if not result.success and result.error_message:
                print(f"   Error: {result.error_message}")
    
    def save_results(self, output_path: str):
        """Save detailed results to JSON file"""
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "summary": asdict(self._generate_summary()),
            "detailed_results": [asdict(result) for result in self.results]
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def export_to_csv(self, output_path: str):
        """Export results to CSV for further analysis"""
        results_data = []
        
        for result in self.results:
            row = {
                "test_id": result.test_id,
                "test_type": result.test_type,
                "input_text": result.input_text,
                "expected_output": result.expected_output,
                "actual_output": result.actual_output,
                "success": result.success,
                "response_time": result.response_time,
                "similarity_score": result.similarity_score,
                "error_message": result.error_message
            }
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        csv_path = Path(output_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"📈 CSV export saved to: {csv_path}")


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SaklAI Evaluation Script")
    parser.add_argument("--config", help="Path to evaluation config file")
    parser.add_argument("--output", default="evaluation/results.json", help="Output file for results")
    parser.add_argument("--csv", help="Export results to CSV file")
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = SaklAIEvaluator(args.config)
        
        # Run evaluation
        summary = await evaluator.run_evaluation()
        
        # Print summary
        evaluator.print_summary(summary)
        
        # Save results
        evaluator.save_results(args.output)
        
        # Export to CSV if requested
        if args.csv:
            evaluator.export_to_csv(args.csv)
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"Overall success rate: {summary.success_rate:.1%}")
        
        # Exit with non-zero code if success rate is below threshold
        if summary.success_rate < 0.8:  # 80% threshold
            print("⚠️  Success rate below 80% threshold")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
