#!/usr/bin/env python3
"""
Quantitative Metrics Module for SaklAI Evaluation

This module provides comprehensive quantitative metrics for:
1. Response Quality Metrics (BLEU, ROUGE, Semantic Similarity)
2. Performance Metrics (Latency, Throughput, Resource Utilization)
3. Accuracy Metrics (Classification, Translation, RAG Retrieval)
4. Statistical Analysis (Confidence Intervals, Distribution Analysis)
5. Business Metrics (Customer Satisfaction Proxy, Task Completion Rate)
"""

import numpy as np
import pandas as pd
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re
import math
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


@dataclass
class QuantitativeMetrics:
    """Comprehensive quantitative metrics container"""
    
    # Response Quality Metrics
    bleu_score: float = 0.0
    rouge_1_score: float = 0.0
    rouge_2_score: float = 0.0
    rouge_l_score: float = 0.0
    semantic_similarity_score: float = 0.0
    lexical_diversity: float = 0.0
    
    # Performance Metrics
    avg_response_time: float = 0.0
    median_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    throughput_qps: float = 0.0
    
    # Accuracy Metrics
    exact_match_accuracy: float = 0.0
    fuzzy_match_accuracy: float = 0.0
    classification_precision: float = 0.0
    classification_recall: float = 0.0
    classification_f1: float = 0.0
    
    # RAG-Specific Metrics
    retrieval_precision_at_k: float = 0.0
    retrieval_recall_at_k: float = 0.0
    retrieval_mrr: float = 0.0  # Mean Reciprocal Rank
    context_relevance_score: float = 0.0
    answer_faithfulness: float = 0.0
    
    # Translation-Specific Metrics
    translation_adequacy: float = 0.0
    translation_fluency: float = 0.0
    character_error_rate: float = 0.0
    word_error_rate: float = 0.0
    
    # Statistical Metrics
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 0.0
    standard_deviation: float = 0.0
    variance: float = 0.0
    
    # Business Metrics
    task_completion_rate: float = 0.0
    user_satisfaction_proxy: float = 0.0
    error_rate: float = 0.0
    system_reliability: float = 0.0


class BusinessMetrics:
    """Business-critical metrics for customer service AI systems"""
    
    def __init__(self):
        self.session_data = []
        self.handoff_times = []
        self.containment_data = []
        self.satisfaction_scores = []
        self.cost_data = []
        self.resolution_data = []
    
    def calculate_average_handoff_time(self, handoff_events: List[Dict]) -> Dict[str, float]:
        """
        Calculate Average Handoff Time - time from bot failure to human agent pickup
        
        Args:
            handoff_events: List of handoff events with timestamps
            Format: [{"bot_end_time": timestamp, "agent_start_time": timestamp, "reason": str}]
        
        Returns:
            Dict with handoff time statistics
        """
        if not handoff_events:
            return {
                "average_handoff_time": 0.0,
                "median_handoff_time": 0.0,
                "max_handoff_time": 0.0,
                "min_handoff_time": 0.0,
                "handoff_count": 0,
                "handoff_reasons": {}
            }
        
        handoff_times = []
        reasons = {}
        
        for event in handoff_events:
            if "bot_end_time" in event and "agent_start_time" in event:
                handoff_time = event["agent_start_time"] - event["bot_end_time"]
                handoff_times.append(handoff_time)
                
                reason = event.get("reason", "unknown")
                reasons[reason] = reasons.get(reason, 0) + 1
        
        if handoff_times:
            return {
                "average_handoff_time": np.mean(handoff_times),
                "median_handoff_time": np.median(handoff_times),
                "max_handoff_time": np.max(handoff_times),
                "min_handoff_time": np.min(handoff_times),
                "handoff_count": len(handoff_times),
                "handoff_reasons": reasons,
                "95th_percentile_handoff": np.percentile(handoff_times, 95)
            }
        
        return {"error": "No valid handoff time data"}
    
    def calculate_bot_containment_rate(self, sessions: List[Dict]) -> Dict[str, float]:
        """
        Calculate Bot Containment Rate - percentage of conversations resolved without human handoff
        
        Args:
            sessions: List of session data
            Format: [{"session_id": str, "resolved_by_bot": bool, "handoff_required": bool}]
        
        Returns:
            Dict with containment rate metrics
        """
        if not sessions:
            return {"bot_containment_rate": 0.0, "total_sessions": 0}
        
        total_sessions = len(sessions)
        bot_resolved = sum(1 for session in sessions if session.get("resolved_by_bot", False))
        handoffs = sum(1 for session in sessions if session.get("handoff_required", False))
        
        containment_rate = (bot_resolved / total_sessions) * 100 if total_sessions > 0 else 0
        handoff_rate = (handoffs / total_sessions) * 100 if total_sessions > 0 else 0
        
        # Calculate by session type
        session_types = {}
        for session in sessions:
            session_type = session.get("type", "general")
            if session_type not in session_types:
                session_types[session_type] = {"total": 0, "contained": 0}
            
            session_types[session_type]["total"] += 1
            if session.get("resolved_by_bot", False):
                session_types[session_type]["contained"] += 1
        
        # Calculate containment by type
        containment_by_type = {}
        for session_type, data in session_types.items():
            containment_by_type[session_type] = (data["contained"] / data["total"]) * 100
        
        return {
            "bot_containment_rate": containment_rate,
            "handoff_rate": handoff_rate,
            "total_sessions": total_sessions,
            "bot_resolved_count": bot_resolved,
            "handoff_count": handoffs,
            "containment_by_type": containment_by_type
        }
    
    def calculate_csat(self, satisfaction_data: List[Dict]) -> Dict[str, float]:
        """
        Calculate Customer Satisfaction (CSAT) metrics
        
        Args:
            satisfaction_data: List of satisfaction ratings
            Format: [{"session_id": str, "rating": int, "feedback": str, "interaction_type": str}]
            Rating scale: 1-5 (1=Very Dissatisfied, 5=Very Satisfied)
        
        Returns:
            Dict with CSAT metrics
        """
        if not satisfaction_data:
            return {"csat_score": 0.0, "total_responses": 0}
        
        ratings = [item["rating"] for item in satisfaction_data if "rating" in item]
        
        if not ratings:
            return {"csat_score": 0.0, "total_responses": 0}
        
        # CSAT calculation - percentage of satisfied customers (rating 4-5)
        satisfied_count = sum(1 for rating in ratings if rating >= 4)
        total_responses = len(ratings)
        csat_percentage = (satisfied_count / total_responses) * 100
        
        # Additional metrics
        average_rating = np.mean(ratings)
        median_rating = np.median(ratings)
        
        # Rating distribution
        rating_distribution = {}
        for i in range(1, 6):
            rating_distribution[f"rating_{i}"] = sum(1 for rating in ratings if rating == i)
        
        # CSAT by interaction type
        csat_by_type = {}
        interaction_types = {}
        
        for item in satisfaction_data:
            interaction_type = item.get("interaction_type", "general")
            if interaction_type not in interaction_types:
                interaction_types[interaction_type] = []
            interaction_types[interaction_type].append(item["rating"])
        
        for interaction_type, type_ratings in interaction_types.items():
            satisfied = sum(1 for rating in type_ratings if rating >= 4)
            total = len(type_ratings)
            csat_by_type[interaction_type] = (satisfied / total) * 100 if total > 0 else 0
        
        return {
            "csat_score": csat_percentage,
            "average_rating": average_rating,
            "median_rating": median_rating,
            "total_responses": total_responses,
            "satisfied_count": satisfied_count,
            "rating_distribution": rating_distribution,
            "csat_by_interaction_type": csat_by_type,
            "nps_promoters": sum(1 for rating in ratings if rating == 5),
            "nps_detractors": sum(1 for rating in ratings if rating <= 2)
        }
    
    def calculate_cost_per_session(self, usage_data) -> Dict[str, float]:
        """
        Calculate Cost per Session and Cost per Inference
        
        Args:
            usage_data: Either aggregated cost data (Dict) or list of usage records
            
            Aggregated format: {
                "total_sessions": int,
                "total_inferences": int, 
                "total_tokens_processed": int,
                "model_costs": {"total_cost": float},
                "infrastructure_costs": {"total_cost": float},
                "operational_costs": {"total_cost": float}
            }
            
            Or List format: [{"session_id": str, "tokens_used": int, "api_calls": int, 
                           "compute_time": float, "model_cost": float}]
        
        Returns:
            Dict with cost metrics
        """
        if not usage_data:
            return {"cost_per_session": 0.0, "cost_per_inference": 0.0}
        
        # Handle aggregated data format
        if isinstance(usage_data, dict) and "total_sessions" in usage_data:
            total_sessions = usage_data.get("total_sessions", 0)
            total_inferences = usage_data.get("total_inferences", 0)
            total_tokens = usage_data.get("total_tokens_processed", 0)
            
            # Calculate total cost from all cost categories
            model_cost = usage_data.get("model_costs", {}).get("total_cost", 0)
            infra_cost = usage_data.get("infrastructure_costs", {}).get("total_cost", 0)
            ops_cost = usage_data.get("operational_costs", {}).get("total_cost", 0)
            total_cost = model_cost + infra_cost + ops_cost
            
            cost_per_session = total_cost / total_sessions if total_sessions > 0 else 0
            cost_per_inference = total_cost / total_inferences if total_inferences > 0 else 0
            cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0
            
            # Calculate cost efficiency score (lower is better)
            # Based on industry benchmarks: excellent < $0.05, good < $0.15, poor > $0.25
            if cost_per_session <= 0.05:
                cost_efficiency_score = 5.0  # Excellent
            elif cost_per_session <= 0.15:
                cost_efficiency_score = 4.0  # Good
            elif cost_per_session <= 0.25:
                cost_efficiency_score = 3.0  # Average
            elif cost_per_session <= 0.50:
                cost_efficiency_score = 2.0  # Below Average
            else:
                cost_efficiency_score = 1.0  # Poor
            
            return {
                "cost_per_session": cost_per_session,
                "cost_per_inference": cost_per_inference,
                "cost_per_token": cost_per_token,
                "total_cost": total_cost,
                "total_sessions": total_sessions,
                "total_inferences": total_inferences,
                "total_tokens": total_tokens,
                "model_cost": model_cost,
                "infrastructure_cost": infra_cost,
                "operational_cost": ops_cost,
                "cost_efficiency_score": cost_efficiency_score,
                "cost_breakdown": {
                    "model_percentage": (model_cost / total_cost * 100) if total_cost > 0 else 0,
                    "infrastructure_percentage": (infra_cost / total_cost * 100) if total_cost > 0 else 0,
                    "operational_percentage": (ops_cost / total_cost * 100) if total_cost > 0 else 0
                }
            }
        
        # Handle list format (original implementation)
        
        total_cost = sum(item.get("model_cost", 0) for item in usage_data)
        total_sessions = len(set(item["session_id"] for item in usage_data))
        total_inferences = sum(item.get("api_calls", 0) for item in usage_data)
        total_tokens = sum(item.get("tokens_used", 0) for item in usage_data)
        total_compute_time = sum(item.get("compute_time", 0) for item in usage_data)
        
        cost_per_session = total_cost / total_sessions if total_sessions > 0 else 0
        cost_per_inference = total_cost / total_inferences if total_inferences > 0 else 0
        cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0
        cost_per_minute = total_cost / (total_compute_time / 60) if total_compute_time > 0 else 0
        
        # Calculate costs by session type
        session_costs = {}
        for item in usage_data:
            session_id = item["session_id"]
            if session_id not in session_costs:
                session_costs[session_id] = {
                    "total_cost": 0,
                    "tokens": 0,
                    "inferences": 0,
                    "compute_time": 0
                }
            session_costs[session_id]["total_cost"] += item.get("model_cost", 0)
            session_costs[session_id]["tokens"] += item.get("tokens_used", 0)
            session_costs[session_id]["inferences"] += item.get("api_calls", 0)
            session_costs[session_id]["compute_time"] += item.get("compute_time", 0)
        
        # Calculate percentiles for cost distribution
        session_cost_values = [data["total_cost"] for data in session_costs.values()]
        
        return {
            "cost_per_session": cost_per_session,
            "cost_per_inference": cost_per_inference,
            "cost_per_token": cost_per_token,
            "cost_per_minute": cost_per_minute,
            "total_cost": total_cost,
            "total_sessions": total_sessions,
            "total_inferences": total_inferences,
            "total_tokens": total_tokens,
            "average_session_cost": np.mean(session_cost_values) if session_cost_values else 0,
            "median_session_cost": np.median(session_cost_values) if session_cost_values else 0,
            "95th_percentile_session_cost": np.percentile(session_cost_values, 95) if session_cost_values else 0,
            "cost_efficiency_score": total_inferences / total_cost if total_cost > 0 else 0
        }
    
    def calculate_first_contact_resolution(self, resolution_data: List[Dict]) -> Dict[str, float]:
        """
        Calculate First Contact Resolution (FCR) rate
        
        Args:
            resolution_data: List of resolution records
            Format: [{"session_id": str, "resolved_first_contact": bool, "issue_type": str, 
                     "resolution_method": str, "follow_up_required": bool}]
        
        Returns:
            Dict with FCR metrics
        """
        if not resolution_data:
            return {"fcr_rate": 0.0, "total_contacts": 0}
        
        total_contacts = len(resolution_data)
        first_contact_resolutions = sum(1 for item in resolution_data 
                                      if item.get("resolved_first_contact", False))
        
        fcr_rate = (first_contact_resolutions / total_contacts) * 100 if total_contacts > 0 else 0
        
        # FCR by issue type
        issue_types = {}
        for item in resolution_data:
            issue_type = item.get("issue_type", "general")
            if issue_type not in issue_types:
                issue_types[issue_type] = {"total": 0, "resolved_first": 0}
            
            issue_types[issue_type]["total"] += 1
            if item.get("resolved_first_contact", False):
                issue_types[issue_type]["resolved_first"] += 1
        
        fcr_by_issue_type = {}
        for issue_type, data in issue_types.items():
            fcr_by_issue_type[issue_type] = (data["resolved_first"] / data["total"]) * 100
        
        # Resolution method analysis
        resolution_methods = {}
        for item in resolution_data:
            method = item.get("resolution_method", "unknown")
            if method not in resolution_methods:
                resolution_methods[method] = {"total": 0, "first_contact": 0}
            
            resolution_methods[method]["total"] += 1
            if item.get("resolved_first_contact", False):
                resolution_methods[method]["first_contact"] += 1
        
        fcr_by_method = {}
        for method, data in resolution_methods.items():
            fcr_by_method[method] = (data["first_contact"] / data["total"]) * 100
        
        # Follow-up requirements
        follow_up_required = sum(1 for item in resolution_data 
                               if item.get("follow_up_required", False))
        
        return {
            "fcr_rate": fcr_rate,
            "total_contacts": total_contacts,
            "first_contact_resolutions": first_contact_resolutions,
            "fcr_by_issue_type": fcr_by_issue_type,
            "fcr_by_resolution_method": fcr_by_method,
            "follow_up_required_count": follow_up_required,
            "follow_up_rate": (follow_up_required / total_contacts) * 100,
            "resolution_efficiency": fcr_rate  # Higher FCR = more efficient
        }


class QuantitativeAnalyzer:
    """Main class for calculating quantitative metrics"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the quantitative analyzer"""
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception as e:
            print(f"Warning: Could not load embedding model {embedding_model}: {e}")
            self.embedding_model = None
        
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score between reference and candidate texts"""
        try:
            # Simple BLEU implementation (1-gram to 4-gram)
            ref_tokens = self._tokenize(reference.lower())
            cand_tokens = self._tokenize(candidate.lower())
            
            if not ref_tokens or not cand_tokens:
                return 0.0
            
            # Calculate n-gram precision for n=1 to 4
            bleu_scores = []
            for n in range(1, 5):
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                cand_ngrams = self._get_ngrams(cand_tokens, n)
                
                if not cand_ngrams:
                    bleu_scores.append(0.0)
                    continue
                
                matches = sum(min(ref_ngrams.get(ngram, 0), cand_ngrams.get(ngram, 0)) 
                             for ngram in cand_ngrams)
                precision = matches / len(cand_ngrams) if len(cand_ngrams) > 0 else 0.0
                bleu_scores.append(precision)
            
            # Geometric mean of n-gram precisions
            if all(score > 0 for score in bleu_scores):
                bleu = math.exp(sum(math.log(score) for score in bleu_scores) / len(bleu_scores))
            else:
                bleu = 0.0
            
            # Brevity penalty
            ref_len = len(ref_tokens)
            cand_len = len(cand_tokens)
            if cand_len > ref_len:
                bp = 1.0
            else:
                bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
            
            return bleu * bp
            
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores"""
        try:
            ref_tokens = self._tokenize(reference.lower())
            cand_tokens = self._tokenize(candidate.lower())
            
            if not ref_tokens or not cand_tokens:
                return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
            
            # ROUGE-1 (unigram overlap)
            ref_unigrams = set(ref_tokens)
            cand_unigrams = set(cand_tokens)
            rouge_1 = len(ref_unigrams & cand_unigrams) / len(ref_unigrams) if ref_unigrams else 0.0
            
            # ROUGE-2 (bigram overlap)
            ref_bigrams = set(self._get_ngrams(ref_tokens, 2).keys())
            cand_bigrams = set(self._get_ngrams(cand_tokens, 2).keys())
            rouge_2 = len(ref_bigrams & cand_bigrams) / len(ref_bigrams) if ref_bigrams else 0.0
            
            # ROUGE-L (Longest Common Subsequence)
            lcs_length = self._longest_common_subsequence(ref_tokens, cand_tokens)
            rouge_l = lcs_length / len(ref_tokens) if ref_tokens else 0.0
            
            return {
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "rouge_l": rouge_l
            }
            
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence embeddings"""
        try:
            if not self.embedding_model:
                # Fallback to TF-IDF cosine similarity
                return self._tfidf_similarity(text1, text2)
            
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return self._tfidf_similarity(text1, text2)
    
    def calculate_lexical_diversity(self, text: str) -> float:
        """Calculate lexical diversity (Type-Token Ratio)"""
        try:
            tokens = self._tokenize(text.lower())
            if not tokens:
                return 0.0
            
            unique_tokens = set(tokens)
            return len(unique_tokens) / len(tokens)
            
        except Exception as e:
            print(f"Error calculating lexical diversity: {e}")
            return 0.0
    
    def calculate_performance_metrics(self, response_times: List[float]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            if not response_times:
                return {
                    "avg_response_time": 0.0,
                    "median_response_time": 0.0,
                    "p95_response_time": 0.0,
                    "p99_response_time": 0.0,
                    "std_dev": 0.0,
                    "variance": 0.0
                }
            
            sorted_times = sorted(response_times)
            n = len(sorted_times)
            
            return {
                "avg_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "p95_response_time": sorted_times[int(0.95 * n)] if n > 0 else 0.0,
                "p99_response_time": sorted_times[int(0.99 * n)] if n > 0 else 0.0,
                "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0.0,
                "variance": statistics.variance(response_times) if len(response_times) > 1 else 0.0
            }
            
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return {}
    
    def calculate_classification_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """Calculate classification accuracy metrics"""
        try:
            if not y_true or not y_pred or len(y_true) != len(y_pred):
                return {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0
                }
            
            # Convert to categorical if needed
            unique_labels = list(set(y_true + y_pred))
            
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
        except Exception as e:
            print(f"Error calculating classification metrics: {e}")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    
    def calculate_retrieval_metrics(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> Dict[str, float]:
        """Calculate retrieval metrics for RAG systems"""
        try:
            if not retrieved_docs or not relevant_docs:
                return {
                    "precision_at_k": 0.0,
                    "recall_at_k": 0.0,
                    "mrr": 0.0
                }
            
            # Take top-k retrieved documents
            top_k_retrieved = retrieved_docs[:k]
            relevant_set = set(relevant_docs)
            
            # Precision@K
            relevant_retrieved = sum(1 for doc in top_k_retrieved if doc in relevant_set)
            precision_at_k = relevant_retrieved / len(top_k_retrieved) if top_k_retrieved else 0.0
            
            # Recall@K
            recall_at_k = relevant_retrieved / len(relevant_set) if relevant_set else 0.0
            
            # Mean Reciprocal Rank (MRR)
            mrr = 0.0
            for i, doc in enumerate(top_k_retrieved):
                if doc in relevant_set:
                    mrr = 1.0 / (i + 1)
                    break
            
            return {
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "mrr": mrr
            }
            
        except Exception as e:
            print(f"Error calculating retrieval metrics: {e}")
            return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
    
    def calculate_translation_metrics(self, reference: str, translation: str) -> Dict[str, float]:
        """Calculate translation-specific metrics"""
        try:
            # Character Error Rate (CER)
            cer = self._calculate_error_rate(reference, translation, level='char')
            
            # Word Error Rate (WER)
            wer = self._calculate_error_rate(reference, translation, level='word')
            
            # Translation adequacy (semantic similarity)
            adequacy = self.calculate_semantic_similarity(reference, translation)
            
            # Translation fluency (lexical diversity and grammar proxy)
            fluency = self.calculate_lexical_diversity(translation)
            
            return {
                "character_error_rate": cer,
                "word_error_rate": wer,
                "adequacy": adequacy,
                "fluency": fluency
            }
            
        except Exception as e:
            print(f"Error calculating translation metrics: {e}")
            return {"character_error_rate": 1.0, "word_error_rate": 1.0, "adequacy": 0.0, "fluency": 0.0}
    
    def calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for a dataset"""
        try:
            if len(data) < 2:
                return (0.0, 0.0)
            
            mean = statistics.mean(data)
            std_err = statistics.stdev(data) / math.sqrt(len(data))
            
            # Use t-distribution for small samples (< 30)
            if len(data) < 30:
                # Simplified t-value approximation
                t_value = 2.0  # Approximate for 95% confidence
            else:
                # Use normal distribution
                from scipy import stats
                t_value = stats.norm.ppf((1 + confidence) / 2)
            
            margin_error = t_value * std_err
            return (mean - margin_error, mean + margin_error)
            
        except Exception as e:
            print(f"Error calculating confidence interval: {e}")
            return (0.0, 0.0)
    
    def analyze_evaluation_results(self, results: List[Dict]) -> QuantitativeMetrics:
        """Analyze comprehensive evaluation results and calculate all metrics"""
        try:
            if not results:
                return QuantitativeMetrics()
            
            # Extract data for analysis
            response_times = [r.get('response_time', 0.0) for r in results]
            success_flags = [r.get('success', False) for r in results]
            
            # Performance metrics
            perf_metrics = self.calculate_performance_metrics(response_times)
            
            # Text quality metrics (for successful results)
            successful_results = [r for r in results if r.get('success', False)]
            quality_scores = {
                'bleu': [],
                'rouge_1': [],
                'rouge_2': [],
                'rouge_l': [],
                'semantic_sim': [],
                'lexical_div': []
            }
            
            for result in successful_results:
                expected = result.get('expected_output', '')
                actual = result.get('actual_output', '')
                
                if expected and actual:
                    # BLEU score
                    bleu = self.calculate_bleu_score(expected, actual)
                    quality_scores['bleu'].append(bleu)
                    
                    # ROUGE scores
                    rouge = self.calculate_rouge_scores(expected, actual)
                    quality_scores['rouge_1'].append(rouge['rouge_1'])
                    quality_scores['rouge_2'].append(rouge['rouge_2'])
                    quality_scores['rouge_l'].append(rouge['rouge_l'])
                    
                    # Semantic similarity
                    sem_sim = self.calculate_semantic_similarity(expected, actual)
                    quality_scores['semantic_sim'].append(sem_sim)
                    
                    # Lexical diversity
                    lex_div = self.calculate_lexical_diversity(actual)
                    quality_scores['lexical_div'].append(lex_div)
            
            # Classification metrics (if applicable)
            expected_categories = [r.get('expected_category') for r in results if r.get('expected_category')]
            predicted_categories = [r.get('predicted_category') for r in results if r.get('predicted_category')]
            
            class_metrics = {}
            if expected_categories and predicted_categories and len(expected_categories) == len(predicted_categories):
                class_metrics = self.calculate_classification_metrics(expected_categories, predicted_categories)
            
            # Statistical analysis
            success_rates = [1.0 if s else 0.0 for s in success_flags]
            ci_lower, ci_upper = self.calculate_confidence_interval(success_rates)
            
            # Compile comprehensive metrics
            metrics = QuantitativeMetrics(
                # Response Quality
                bleu_score=statistics.mean(quality_scores['bleu']) if quality_scores['bleu'] else 0.0,
                rouge_1_score=statistics.mean(quality_scores['rouge_1']) if quality_scores['rouge_1'] else 0.0,
                rouge_2_score=statistics.mean(quality_scores['rouge_2']) if quality_scores['rouge_2'] else 0.0,
                rouge_l_score=statistics.mean(quality_scores['rouge_l']) if quality_scores['rouge_l'] else 0.0,
                semantic_similarity_score=statistics.mean(quality_scores['semantic_sim']) if quality_scores['semantic_sim'] else 0.0,
                lexical_diversity=statistics.mean(quality_scores['lexical_div']) if quality_scores['lexical_div'] else 0.0,
                
                # Performance
                avg_response_time=perf_metrics.get('avg_response_time', 0.0),
                median_response_time=perf_metrics.get('median_response_time', 0.0),
                p95_response_time=perf_metrics.get('p95_response_time', 0.0),
                p99_response_time=perf_metrics.get('p99_response_time', 0.0),
                
                # Accuracy
                exact_match_accuracy=sum(success_flags) / len(success_flags) if success_flags else 0.0,
                classification_precision=class_metrics.get('precision', 0.0),
                classification_recall=class_metrics.get('recall', 0.0),
                classification_f1=class_metrics.get('f1_score', 0.0),
                
                # Statistical
                confidence_interval_lower=ci_lower,
                confidence_interval_upper=ci_upper,
                standard_deviation=perf_metrics.get('std_dev', 0.0),
                variance=perf_metrics.get('variance', 0.0),
                
                # Business
                task_completion_rate=sum(success_flags) / len(success_flags) if success_flags else 0.0,
                error_rate=1.0 - (sum(success_flags) / len(success_flags)) if success_flags else 1.0,
                system_reliability=sum(success_flags) / len(success_flags) if success_flags else 0.0
            )
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing evaluation results: {e}")
            return QuantitativeMetrics()
    
    # Helper methods
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF cosine similarity"""
        try:
            texts = [text1, text2]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def _calculate_error_rate(self, reference: str, hypothesis: str, level: str = 'word') -> float:
        """Calculate error rate at character or word level"""
        try:
            if level == 'char':
                ref_items = list(reference.lower())
                hyp_items = list(hypothesis.lower())
            else:  # word level
                ref_items = self._tokenize(reference.lower())
                hyp_items = self._tokenize(hypothesis.lower())
            
            # Simple edit distance calculation
            if not ref_items:
                return 1.0 if hyp_items else 0.0
            
            # Levenshtein distance approximation
            if len(ref_items) == 0:
                return len(hyp_items)
            if len(hyp_items) == 0:
                return len(ref_items)
            
            # Simple substitution/insertion/deletion count
            substitutions = sum(1 for r, h in zip(ref_items, hyp_items) if r != h)
            insertions = max(0, len(hyp_items) - len(ref_items))
            deletions = max(0, len(ref_items) - len(hyp_items))
            
            total_errors = substitutions + insertions + deletions
            return total_errors / len(ref_items)
            
        except Exception as e:
            print(f"Error calculating error rate: {e}")
            return 1.0


def generate_quantitative_report(metrics: QuantitativeMetrics, results: List[Dict], 
                               business_data: Optional[Dict] = None) -> str:
    """Generate a comprehensive quantitative metrics report with business metrics"""
    
    report_lines = [
        "# Quantitative Evaluation Metrics Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        f"- **Overall Success Rate**: {metrics.task_completion_rate:.1%}",
        f"- **System Reliability**: {metrics.system_reliability:.1%}",
        f"- **Average Response Time**: {metrics.avg_response_time:.2f}s",
        f"- **Semantic Quality Score**: {metrics.semantic_similarity_score:.3f}",
        ""
    ]
    
    # Add business metrics summary if available
    if business_data:
        business_calc = BusinessMetrics()
        report_lines.extend([
            "## Business Performance Overview",
            ""
        ])
        
        # Bot Containment Rate
        if "sessions" in business_data:
            containment = business_calc.calculate_bot_containment_rate(business_data["sessions"])
            report_lines.extend([
                f"- **Bot Containment Rate**: {containment.get('bot_containment_rate', 0):.1f}%",
                f"- **Total Sessions Analyzed**: {containment.get('total_sessions', 0)}",
            ])
        
        # CSAT Score
        if "satisfaction_data" in business_data:
            csat = business_calc.calculate_csat(business_data["satisfaction_data"])
            report_lines.extend([
                f"- **Customer Satisfaction (CSAT)**: {csat.get('csat_score', 0):.1f}%",
                f"- **Average Rating**: {csat.get('average_rating', 0):.1f}/5.0",
            ])
        
        # Cost Metrics
        if "usage_data" in business_data:
            cost = business_calc.calculate_cost_per_session(business_data["usage_data"])
            report_lines.extend([
                f"- **Cost per Session**: ${cost.get('cost_per_session', 0):.3f}",
                f"- **Cost per Inference**: ${cost.get('cost_per_inference', 0):.4f}",
            ])
        
        # FCR Rate
        if "resolution_data" in business_data:
            fcr = business_calc.calculate_first_contact_resolution(business_data["resolution_data"])
            report_lines.extend([
                f"- **First Contact Resolution**: {fcr.get('fcr_rate', 0):.1f}%",
            ])
        
        # Handoff Time
        if "handoff_events" in business_data:
            handoff = business_calc.calculate_average_handoff_time(business_data["handoff_events"])
            if "error" not in handoff:
                report_lines.extend([
                    f"- **Average Handoff Time**: {handoff.get('average_handoff_time', 0):.1f} seconds",
                ])
        
        report_lines.append("")
    
    report_lines.extend([
        "## Response Quality Metrics",
        f"| Metric | Score | Interpretation |",
        f"|--------|-------|----------------|",
        f"| BLEU Score | {metrics.bleu_score:.3f} | Translation/Generation Quality |",
        f"| ROUGE-1 | {metrics.rouge_1_score:.3f} | Unigram Overlap |",
        f"| ROUGE-2 | {metrics.rouge_2_score:.3f} | Bigram Overlap |",
        f"| ROUGE-L | {metrics.rouge_l_score:.3f} | Longest Common Subsequence |",
        f"| Semantic Similarity | {metrics.semantic_similarity_score:.3f} | Meaning Preservation |",
        f"| Lexical Diversity | {metrics.lexical_diversity:.3f} | Vocabulary Richness |",
        "",
        "## Performance Metrics",
        f"| Metric | Value | Unit |",
        f"|--------|-------|------|",
        f"| Average Response Time | {metrics.avg_response_time:.2f} | seconds |",
        f"| Median Response Time | {metrics.median_response_time:.2f} | seconds |",
        f"| 95th Percentile | {metrics.p95_response_time:.2f} | seconds |",
        f"| 99th Percentile | {metrics.p99_response_time:.2f} | seconds |",
        f"| Standard Deviation | {metrics.standard_deviation:.2f} | seconds |",
        "",
        "## Accuracy Metrics",
        f"| Metric | Score | Description |",
        f"|--------|-------|-------------|",
        f"| Exact Match Accuracy | {metrics.exact_match_accuracy:.1%} | Perfect Matches |",
        f"| Classification Precision | {metrics.classification_precision:.3f} | Positive Predictive Value |",
        f"| Classification Recall | {metrics.classification_recall:.3f} | Sensitivity |",
        f"| Classification F1-Score | {metrics.classification_f1:.3f} | Harmonic Mean |",
        "",
        "## Statistical Analysis",
        f"- **Confidence Interval (95%)**: [{metrics.confidence_interval_lower:.3f}, {metrics.confidence_interval_upper:.3f}]",
        f"- **Variance**: {metrics.variance:.4f}",
        f"- **Error Rate**: {metrics.error_rate:.1%}",
        "",
        "## Business Impact Metrics",
        f"- **Task Completion Rate**: {metrics.task_completion_rate:.1%}",
        f"- **System Reliability**: {metrics.system_reliability:.1%}",
        f"- **User Satisfaction Proxy**: {metrics.user_satisfaction_proxy:.1%}",
        "",
        "## Detailed Analysis",
        "",
        "### Quality Assessment",
        "- **BLEU Score Interpretation**:",
        f"  - {metrics.bleu_score:.3f} indicates {'Excellent' if metrics.bleu_score > 0.7 else 'Good' if metrics.bleu_score > 0.5 else 'Fair' if metrics.bleu_score > 0.3 else 'Poor'} translation/generation quality",
        "",
        "- **Semantic Similarity Assessment**:",
        f"  - {metrics.semantic_similarity_score:.3f} shows {'High' if metrics.semantic_similarity_score > 0.8 else 'Moderate' if metrics.semantic_similarity_score > 0.6 else 'Low'} semantic preservation",
        "",
        "### Performance Assessment",
        f"- **Response Time Analysis**: Average {metrics.avg_response_time:.2f}s is {'Excellent' if metrics.avg_response_time < 1.0 else 'Good' if metrics.avg_response_time < 3.0 else 'Acceptable' if metrics.avg_response_time < 5.0 else 'Needs Improvement'}",
        "",
        "### Reliability Assessment",
        f"- **System Reliability**: {metrics.system_reliability:.1%} reliability indicates {'Production Ready' if metrics.system_reliability > 0.95 else 'Near Production' if metrics.system_reliability > 0.90 else 'Development Stage' if metrics.system_reliability > 0.80 else 'Needs Significant Improvement'}",
        ""
    ])
    
    # Add detailed business metrics section if available
    if business_data:
        business_calc = BusinessMetrics()
        report_lines.extend([
            "## Business Metrics Detailed Analysis",
            ""
        ])
        
        # Detailed containment analysis
        if "sessions" in business_data:
            containment = business_calc.calculate_bot_containment_rate(business_data["sessions"])
            report_lines.extend([
                "### Bot Containment Analysis",
                f"| Metric | Value | Status |",
                f"|--------|-------|--------|",
                f"| Bot Containment Rate | {containment.get('bot_containment_rate', 0):.1f}% | {'🟢 Excellent' if containment.get('bot_containment_rate', 0) > 85 else '🟡 Good' if containment.get('bot_containment_rate', 0) > 70 else '🔴 Needs Improvement'} |",
                f"| Total Sessions | {containment.get('total_sessions', 0)} | - |",
                f"| Bot Resolved | {containment.get('bot_resolved_count', 0)} | - |",
                f"| Human Handoffs | {containment.get('handoff_count', 0)} | - |",
                f"| Handoff Rate | {containment.get('handoff_rate', 0):.1f}% | - |",
                ""
            ])
            
            # Containment by type if available
            if containment.get('containment_by_type'):
                report_lines.extend([
                    "#### Containment by Session Type",
                    f"| Session Type | Containment Rate |",
                    f"|--------------|------------------|"
                ])
                for session_type, rate in containment['containment_by_type'].items():
                    report_lines.append(f"| {session_type} | {rate:.1f}% |")
                report_lines.append("")
        
        # Detailed CSAT analysis
        if "satisfaction_data" in business_data:
            csat = business_calc.calculate_csat(business_data["satisfaction_data"])
            report_lines.extend([
                "### Customer Satisfaction Analysis",
                f"| Metric | Value | Status |",
                f"|--------|-------|--------|",
                f"| CSAT Score | {csat.get('csat_score', 0):.1f}% | {'🟢 Excellent' if csat.get('csat_score', 0) > 90 else '🟡 Good' if csat.get('csat_score', 0) > 80 else '🔴 Needs Improvement'} |",
                f"| Average Rating | {csat.get('average_rating', 0):.1f}/5.0 | - |",
                f"| Total Responses | {csat.get('total_responses', 0)} | - |",
                f"| Satisfied Customers | {csat.get('satisfied_count', 0)} | - |",
                f"| NPS Promoters | {csat.get('nps_promoters', 0)} | - |",
                f"| NPS Detractors | {csat.get('nps_detractors', 0)} | - |",
                ""
            ])
        
        # Detailed cost analysis
        if "usage_data" in business_data:
            cost = business_calc.calculate_cost_per_session(business_data["usage_data"])
            report_lines.extend([
                "### Cost Analysis",
                f"| Metric | Value | Efficiency |",
                f"|--------|-------|------------|",
                f"| Cost per Session | ${cost.get('cost_per_session', 0):.3f} | {'🟢 Efficient' if cost.get('cost_per_session', 0) < 0.05 else '🟡 Acceptable' if cost.get('cost_per_session', 0) < 0.15 else '🔴 High'} |",
                f"| Cost per Inference | ${cost.get('cost_per_inference', 0):.4f} | - |",
                f"| Cost per Token | ${cost.get('cost_per_token', 0):.6f} | - |",
                f"| Total Cost | ${cost.get('total_cost', 0):.2f} | - |",
                f"| Total Sessions | {cost.get('total_sessions', 0)} | - |",
                f"| Total Inferences | {cost.get('total_inferences', 0)} | - |",
                f"| Cost Efficiency Score | {cost.get('cost_efficiency_score', 0):.2f} | - |",
                ""
            ])
        
        # Detailed FCR analysis
        if "resolution_data" in business_data:
            fcr = business_calc.calculate_first_contact_resolution(business_data["resolution_data"])
            report_lines.extend([
                "### First Contact Resolution Analysis",
                f"| Metric | Value | Status |",
                f"|--------|-------|--------|",
                f"| FCR Rate | {fcr.get('fcr_rate', 0):.1f}% | {'🟢 Excellent' if fcr.get('fcr_rate', 0) > 85 else '🟡 Good' if fcr.get('fcr_rate', 0) > 75 else '🔴 Needs Improvement'} |",
                f"| Total Contacts | {fcr.get('total_contacts', 0)} | - |",
                f"| First Contact Resolutions | {fcr.get('first_contact_resolutions', 0)} | - |",
                f"| Follow-up Required | {fcr.get('follow_up_required_count', 0)} | - |",
                f"| Follow-up Rate | {fcr.get('follow_up_rate', 0):.1f}% | - |",
                ""
            ])
        
        # Detailed handoff analysis
        if "handoff_events" in business_data:
            handoff = business_calc.calculate_average_handoff_time(business_data["handoff_events"])
            if "error" not in handoff:
                report_lines.extend([
                    "### Handoff Time Analysis",
                    f"| Metric | Value | Status |",
                    f"|--------|-------|--------|",
                    f"| Average Handoff Time | {handoff.get('average_handoff_time', 0):.1f}s | {'🟢 Excellent' if handoff.get('average_handoff_time', 0) < 30 else '🟡 Good' if handoff.get('average_handoff_time', 0) < 60 else '🔴 Slow'} |",
                    f"| Median Handoff Time | {handoff.get('median_handoff_time', 0):.1f}s | - |",
                    f"| 95th Percentile | {handoff.get('95th_percentile_handoff', 0):.1f}s | - |",
                    f"| Max Handoff Time | {handoff.get('max_handoff_time', 0):.1f}s | - |",
                    f"| Total Handoffs | {handoff.get('handoff_count', 0)} | - |",
                    ""
                ])
                
                # Handoff reasons if available
                if handoff.get('handoff_reasons'):
                    report_lines.extend([
                        "#### Handoff Reasons",
                        f"| Reason | Count |",
                        f"|--------|-------|"
                    ])
                    for reason, count in handoff['handoff_reasons'].items():
                        report_lines.append(f"| {reason} | {count} |")
                    report_lines.append("")
    
    report_lines.extend([
        "## Recommendations",
        ""
    ])
    
    # Performance recommendations
    if metrics.avg_response_time > 5.0:
        report_lines.append("### Performance Optimization")
        report_lines.append("- ⚠️ **Response Time**: Consider optimizing model inference or implementing response caching")
    
    if metrics.bleu_score < 0.5:
        report_lines.append("### Content Quality Improvement")
        report_lines.append("- ⚠️ **BLEU Score**: Consider fine-tuning response generation or improving training data")
    
    if metrics.rouge_l_score < 0.4:
        report_lines.append("### Response Relevance")
        report_lines.append("- ⚠️ **ROUGE-L Score**: Improve response alignment with user queries")
    
    if business_data:
        if "sessions" in business_data:
            containment = business_calc.calculate_bot_containment_rate(business_data["sessions"])
            if containment.get('bot_containment_rate', 0) < 70:
                report_lines.append("### Bot Containment Improvement")
                report_lines.append("- ⚠️ **Low Containment**: Enhance bot capabilities to reduce human handoffs")
        
        if "satisfaction_data" in business_data:
            csat = business_calc.calculate_csat(business_data["satisfaction_data"])
            if csat.get('csat_score', 0) < 80:
                report_lines.append("### Customer Satisfaction Enhancement")
                report_lines.append("- ⚠️ **Low CSAT**: Focus on improving response quality and user experience")
    
    return "\n".join(report_lines)
    
    if metrics.semantic_similarity_score < 0.6:
        report_lines.append("- **Quality**: Improve semantic similarity - current score indicates meaning loss")
    
    if metrics.system_reliability < 0.9:
        report_lines.append("- **Reliability**: Address system failures - reliability below 90%")
    
    if metrics.classification_f1 < 0.8:
        report_lines.append("- **Classification**: Enhance classification accuracy - F1 score below optimal threshold")
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    # Example usage
    analyzer = QuantitativeAnalyzer()
    
    # Sample results for testing
    sample_results = [
        {
            "test_id": "test_001",
            "expected_output": "How to open a savings account?",
            "actual_output": "How can I open a savings account?",
            "response_time": 1.5,
            "success": True,
            "expected_category": "banking",
            "predicted_category": "banking"
        },
        {
            "test_id": "test_002", 
            "expected_output": "Credit card application process",
            "actual_output": "Process for applying for credit cards",
            "response_time": 2.1,
            "success": True,
            "expected_category": "credit",
            "predicted_category": "credit"
        }
    ]
    
    metrics = analyzer.analyze_evaluation_results(sample_results)
    report = generate_quantitative_report(metrics, sample_results)
    print(report)
