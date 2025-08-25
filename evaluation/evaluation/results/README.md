# SaklAI Evaluation Suite

This directory contains comprehensive evaluation scripts for testing the SaklAI RAG and Translation systems.

## Overview

The evaluation suite includes:

1. **Functional Testing** - Tests core functionality of RAG, translation, and classification
2. **Performance Benchmarking** - Load testing and performance metrics
3. **Comprehensive Testing** - Extended test cases with edge cases
4. **Automated Reporting** - Generates detailed reports and metrics

## Files

### Core Scripts

- `evaluation_script.py` - Main functional evaluation script
- `performance_benchmark.py` - Performance and load testing
- `run_evaluation.py` - Unified runner for all tests

### Configuration

- `config.json` - Main configuration file
- `test_cases.json` - Extended test cases and scenarios
- `requirements.txt` - Additional Python dependencies

### Output

- `results/` - Directory for all evaluation results
- Generated reports in Markdown, JSON, and CSV formats

## Quick Start

### 1. Install Dependencies

```bash
# From the main saklAI directory
pip install -r evaluation/requirements.txt
```

### 2. Run Basic Evaluation

```bash
cd evaluation
python run_evaluation.py --mode functional
```

### 3. Run Performance Tests

```bash
python run_evaluation.py --mode performance --duration 30 --concurrent-users 3
```

### 4. Run Complete Evaluation Suite

```bash
python run_evaluation.py --mode all
```

## Detailed Usage

### Functional Testing

Test the core functionality of the system:

```bash
# Basic functional tests
python evaluation_script.py

# With custom config
python evaluation_script.py --config config.json --output results/my_test.json

# Export to CSV
python evaluation_script.py --csv results/my_test.csv
```

**What it tests:**

- RAG system accuracy and response quality
- Translation accuracy (Tagalog/Taglish to English)
- Classification accuracy (message routing to correct templates)
- End-to-end workflow (translation + classification + RAG)

### Performance Benchmarking

Test system performance under load:

```bash
# Basic performance test
python performance_benchmark.py

# Custom load parameters
python performance_benchmark.py --concurrent-users 10 --duration 120 --output perf_results.json
```

**What it measures:**

- Response times (average, 95th percentile, 99th percentile)
- Operations per second
- Memory usage
- CPU utilization
- Error rates under concurrent load

### Comprehensive Testing

Run extended test suites with edge cases:

```bash
python run_evaluation.py --mode comprehensive
```

This uses the extended test cases from `test_cases.json` including:

- Banking domain-specific scenarios
- Complex translation cases
- Edge cases and error handling
- Mixed language inputs

## Configuration

### Main Config (`config.json`)

```json
{
  "vector_store_path": "./rag_store_eval",
  "test_user_id": "eval_user_001",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "top_k": 5,
  "similarity_threshold": 0.7,
  "timeout_seconds": 30,
  "thresholds": {
    "minimum_success_rate": 0.8,
    "maximum_response_time": 10.0
  }
}
```

### Extended Test Cases (`test_cases.json`)

Contains categorized test suites:

- `rag_comprehensive` - RAG system tests
- `translation_comprehensive` - Translation tests
- `classification_edge_cases` - Classification boundary tests
- `end_to_end_complex` - Integration tests
- `error_handling` - Robustness tests

## Test Types

### 1. RAG Tests

- Tests document retrieval accuracy
- Validates response generation
- Checks template routing
- Measures response quality

Example:

```
Input: "What are the requirements for opening a savings account?"
Expected: Uses savings_accounts.yaml template, returns relevant banking info
```

### 2. Translation Tests

- Tagalog to English translation
- Taglish (mixed language) handling
- Banking domain terminology
- Accuracy measurement

Example:

```
Input: "Paano mag-open ng savings account?"
Expected: "How to open a savings account?"
```

### 3. Classification Tests

- Message categorization
- Template selection
- Confidence scoring
- Edge case handling

Example:

```
Input: "Hello, how are you?"
Expected: config.yaml template (general conversation)
```

### 4. End-to-End Tests

- Complete workflow testing
- Translation → Classification → RAG
- Multi-language support
- Integration validation

## Metrics and Scoring

### Success Criteria

- **Functional Success Rate**: ≥80%
- **Translation Accuracy**: ≥85%
- **Classification Accuracy**: ≥80%
- **Response Time**: ≤10 seconds average
- **Performance Success Rate**: ≥95% under load

### Similarity Scoring

- Token-based similarity for translation accuracy
- Configurable threshold (default: 0.7)
- Accounts for domain-specific terminology

### Performance Metrics

- Operations per second
- Response time percentiles
- Memory usage patterns
- Error rates and types

## Output Files

### JSON Results

Detailed results with:

- Individual test outcomes
- Performance metrics
- Error messages and stack traces
- System information

### CSV Export

Tabular data for:

- Test results analysis
- Performance trending
- Statistical analysis

### Markdown Reports

Human-readable reports with:

- Executive summary
- Detailed metrics
- Recommendations
- Trend analysis

## Troubleshooting

### Common Issues

1. **API Key Not Found**

   ```
   Error: API key not found. Please set 'api_key' environment variable.
   ```

   Solution: Ensure your `.env` file has the correct API key

2. **Module Import Errors**

   ```
   ModuleNotFoundError: No module named 'services.rag'
   ```

   Solution: Run from the evaluation directory, scripts automatically add server path

3. **Memory Issues**

   ```
   Out of memory during vector store operations
   ```

   Solution: Reduce concurrent users or test duration

4. **Timeout Errors**
   ```
   Request timeout after 30 seconds
   ```
   Solution: Increase timeout_seconds in config or check API connectivity

### Performance Tips

1. **For faster testing**: Use the "fast" embedding model in config
2. **For accuracy**: Use "quality" embedding model (slower)
3. **For load testing**: Start with low concurrent users (2-3) and increase gradually
4. **For CI/CD**: Use shorter duration tests (30-60 seconds)

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
- name: Run SaklAI Evaluation
  run: |
    cd evaluation
    python run_evaluation.py --mode functional --duration 30
```

The scripts return appropriate exit codes:

- 0: Success (all tests pass)
- 1: Failure (tests fail or errors occur)
- 130: Interrupted by user

## Contributing

When adding new test cases:

1. Add to `test_cases.json` in appropriate suite
2. Include expected outcomes
3. Set appropriate priority levels
4. Test locally before committing

When modifying evaluation logic:

1. Update corresponding test cases
2. Verify backward compatibility
3. Update documentation
4. Run full evaluation suite

## Advanced Usage

### Custom Test Cases

Create your own test suite:

```python
from evaluation_script import TestCase, SaklAIEvaluator

custom_tests = [
    TestCase(
        id="custom_001",
        input_text="Your test input",
        expected_template="expected_template.yaml",
        test_type="rag"
    )
]

evaluator = SaklAIEvaluator()
evaluator.get_test_cases = lambda: custom_tests
summary = await evaluator.run_evaluation()
```

### Performance Monitoring

For continuous monitoring:

```python
from performance_benchmark import PerformanceBenchmark

benchmark = PerformanceBenchmark()
results = await benchmark.benchmark_rag(duration=30, concurrent_users=2)

# Check if performance degrades
if results.average_response_time > 5.0:
    alert("Performance degradation detected")
```

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review logs in the results directory
3. Verify configuration settings
4. Test with minimal cases first
