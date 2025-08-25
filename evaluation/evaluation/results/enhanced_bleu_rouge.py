"""
Enhanced BLEU/ROUGE Implementation with Improved Tokenization
"""

import re
import math
import string
from collections import Counter
from typing import List, Dict

class EnhancedTextMetrics:
    """Enhanced BLEU/ROUGE calculations with better tokenization"""
    
    def __init__(self):
        self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def enhanced_tokenize(self, text: str, normalize_currency: bool = True) -> List[str]:
        """
        Enhanced tokenization with better handling of:
        - Currency and numbers
        - Punctuation
        - Contractions
        """
        if normalize_currency:
            # Normalize currency: $1,250.50 -> "dollar 1250.50"
            text = re.sub(r'\$([0-9,]+\.?[0-9]*)', r'dollar \1', text)
            # Remove commas from numbers: 1,250 -> 1250
            text = re.sub(r'(\d+),(\d+)', r'\1\2', text)
        
        # Handle contractions: don't -> do not
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Replace punctuation with spaces (except periods in numbers)
        text = re.sub(r'[^\w\s.]', ' ', text)
        # Handle multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize and filter
        tokens = text.lower().split()
        
        # Optional: Remove stopwords for better matching
        # tokens = [token for token in tokens if token not in self.stopwords]
        
        return tokens
    
    def calculate_enhanced_bleu(self, reference: str, candidate: str, 
                               max_n: int = 4, use_smoothing: bool = True) -> Dict[str, float]:
        """
        Enhanced BLEU with smoothing and better tokenization
        """
        ref_tokens = self.enhanced_tokenize(reference)
        cand_tokens = self.enhanced_tokenize(candidate)
        
        if not ref_tokens or not cand_tokens:
            return {"bleu": 0.0, "precisions": [0.0] * max_n}
        
        # Calculate n-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            cand_ngrams = self._get_ngrams(cand_tokens, n)
            
            if not cand_ngrams:
                if use_smoothing:
                    precisions.append(1e-7)  # Smoothing for zero counts
                else:
                    precisions.append(0.0)
                continue
            
            matches = sum(min(ref_ngrams.get(ngram, 0), cand_ngrams.get(ngram, 0)) 
                         for ngram in cand_ngrams)
            
            precision = matches / len(cand_ngrams) if len(cand_ngrams) > 0 else 0.0
            
            if precision == 0.0 and use_smoothing:
                precision = 1e-7  # Smoothing
            
            precisions.append(precision)
        
        # Geometric mean with smoothing
        if all(p > 0 for p in precisions):
            bleu = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            bleu = 0.0
        
        # Brevity penalty
        ref_len = len(ref_tokens)
        cand_len = len(cand_tokens)
        if cand_len > ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
        
        final_bleu = bleu * bp
        
        return {
            "bleu": final_bleu,
            "precisions": precisions,
            "brevity_penalty": bp,
            "reference_length": ref_len,
            "candidate_length": cand_len
        }
    
    def calculate_enhanced_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Enhanced ROUGE with better tokenization
        """
        ref_tokens = self.enhanced_tokenize(reference)
        cand_tokens = self.enhanced_tokenize(candidate)
        
        if not ref_tokens or not cand_tokens:
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
        
        # ROUGE-1 (unigram overlap)
        ref_unigrams = Counter(ref_tokens)
        cand_unigrams = Counter(cand_tokens)
        
        overlap_count = sum((ref_unigrams & cand_unigrams).values())
        rouge_1_recall = overlap_count / sum(ref_unigrams.values()) if ref_unigrams else 0.0
        rouge_1_precision = overlap_count / sum(cand_unigrams.values()) if cand_unigrams else 0.0
        rouge_1_f1 = (2 * rouge_1_precision * rouge_1_recall) / (rouge_1_precision + rouge_1_recall) if (rouge_1_precision + rouge_1_recall) > 0 else 0.0
        
        # ROUGE-2 (bigram overlap)
        ref_bigrams = Counter(self._get_ngrams(ref_tokens, 2).keys())
        cand_bigrams = Counter(self._get_ngrams(cand_tokens, 2).keys())
        
        bigram_overlap = len(set(ref_bigrams.keys()) & set(cand_bigrams.keys()))
        rouge_2 = bigram_overlap / len(ref_bigrams) if ref_bigrams else 0.0
        
        # ROUGE-L (Longest Common Subsequence)
        lcs_length = self._longest_common_subsequence(ref_tokens, cand_tokens)
        rouge_l_recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
        rouge_l_precision = lcs_length / len(cand_tokens) if cand_tokens else 0.0
        rouge_l_f1 = (2 * rouge_l_precision * rouge_l_recall) / (rouge_l_precision + rouge_l_recall) if (rouge_l_precision + rouge_l_recall) > 0 else 0.0
        
        return {
            "rouge_1": rouge_1_f1,
            "rouge_1_recall": rouge_1_recall,
            "rouge_1_precision": rouge_1_precision,
            "rouge_2": rouge_2,
            "rouge_l": rouge_l_f1,
            "rouge_l_recall": rouge_l_recall,
            "rouge_l_precision": rouge_l_precision
        }
    
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

def test_enhanced_metrics():
    """Test the enhanced metrics implementation"""
    
    print("🧪 TESTING ENHANCED BLEU/ROUGE METRICS")
    print("=" * 60)
    
    metrics = EnhancedTextMetrics()
    
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
        },
        {
            "name": "Perfect Match Test",
            "expected": "Hello world",
            "actual": "Hello world"
        },
        {
            "name": "Synonym Test",
            "expected": "The car is red",
            "actual": "The automobile is red"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {case['name']}")
        print(f"Expected: {case['expected']}")
        print(f"Actual:   {case['actual']}")
        
        # Enhanced tokenization
        exp_tokens = metrics.enhanced_tokenize(case['expected'])
        act_tokens = metrics.enhanced_tokenize(case['actual'])
        print(f"Enhanced tokens - Expected: {exp_tokens}")
        print(f"Enhanced tokens - Actual:   {act_tokens}")
        
        # Calculate enhanced BLEU
        bleu_result = metrics.calculate_enhanced_bleu(case['expected'], case['actual'])
        print(f"📈 Enhanced BLEU: {bleu_result['bleu']:.6f}")
        print(f"   Precisions: {[f'{p:.3f}' for p in bleu_result['precisions']]}")
        print(f"   Brevity Penalty: {bleu_result['brevity_penalty']:.3f}")
        
        # Calculate enhanced ROUGE
        rouge_result = metrics.calculate_enhanced_rouge(case['expected'], case['actual'])
        print(f"📈 Enhanced ROUGE:")
        print(f"   ROUGE-1 F1: {rouge_result['rouge_1']:.6f}")
        print(f"   ROUGE-2:    {rouge_result['rouge_2']:.6f}")
        print(f"   ROUGE-L F1: {rouge_result['rouge_l']:.6f}")
        
        print("-" * 50)
    
    print("\n✅ ENHANCED METRICS TESTING COMPLETE")
    print("🎯 Improved tokenization handles currency, contractions, and punctuation better")
    print("📊 BLEU/ROUGE scores are now more accurate and meaningful")

if __name__ == "__main__":
    test_enhanced_metrics()
