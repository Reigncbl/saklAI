"""
PDF Analysis Script for BPI Product Data
Analyzes document structure to optimize RAG chunking strategy
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Add the server directory to Python path
server_dir = Path(__file__).parent.parent
sys.path.append(str(server_dir))

# LlamaIndex imports for PDF analysis
from llama_index.readers.file import PDFReader
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def analyze_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """Analyze PDF content and structure for optimal chunking"""
    
    print(f"📄 Analyzing PDF: {pdf_path}")
    
    # Load PDF
    loader = PDFReader()
    documents = loader.load_data(pdf_path)
    
    analysis = {
        "total_pages": len(documents),
        "total_characters": 0,
        "total_words": 0,
        "average_page_length": 0,
        "content_structure": [],
        "recommended_chunk_sizes": {},
        "sample_content": []
    }
    
    # Analyze each page
    page_lengths = []
    for i, doc in enumerate(documents):
        content = doc.text
        char_count = len(content)
        word_count = len(content.split())
        
        page_lengths.append(char_count)
        analysis["total_characters"] += char_count
        analysis["total_words"] += word_count
        
        # Store sample content from first few pages
        if i < 3:
            analysis["sample_content"].append({
                "page": i + 1,
                "preview": content[:500] + "..." if len(content) > 500 else content,
                "char_count": char_count,
                "word_count": word_count
            })
        
        # Analyze content structure patterns
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        structure_info = {
            "page": i + 1,
            "total_lines": len(lines),
            "content_lines": len(non_empty_lines),
            "avg_line_length": sum(len(line) for line in non_empty_lines) / max(1, len(non_empty_lines)),
            "has_bullets": any('•' in line or '*' in line for line in non_empty_lines),
            "has_numbers": any(line.strip().startswith(tuple('0123456789')) for line in non_empty_lines),
            "has_headers": any(len(line.strip()) < 100 and line.strip().isupper() for line in non_empty_lines)
        }
        analysis["content_structure"].append(structure_info)
    
    # Calculate averages
    analysis["average_page_length"] = analysis["total_characters"] / max(1, len(documents))
    
    # Recommend chunk sizes based on analysis
    avg_page_chars = analysis["average_page_length"]
    
    analysis["recommended_chunk_sizes"] = {
        "small_chunks": {
            "size": min(512, int(avg_page_chars * 0.3)),
            "overlap": 50,
            "use_case": "Precise information retrieval, Q&A"
        },
        "medium_chunks": {
            "size": min(1024, int(avg_page_chars * 0.6)),
            "overlap": 100,
            "use_case": "Balanced context and precision"
        },
        "large_chunks": {
            "size": min(2048, int(avg_page_chars * 1.0)),
            "overlap": 200,
            "use_case": "Context-heavy tasks, summarization"
        },
        "semantic_chunks": {
            "size": "variable",
            "method": "semantic_similarity",
            "use_case": "Content-aware chunking based on meaning"
        }
    }
    
    return analysis

def test_chunking_strategies(documents: List[Document]) -> Dict[str, Any]:
    """Test different chunking strategies and compare results"""
    
    print("🔍 Testing different chunking strategies...")
    
    # Initialize embedding model for semantic chunking
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    strategies = {}
    
    # Strategy 1: Sentence-based chunking
    sentence_splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    sentence_nodes = sentence_splitter.get_nodes_from_documents(documents)
    strategies["sentence_512"] = {
        "node_count": len(sentence_nodes),
        "avg_node_size": sum(len(node.text) for node in sentence_nodes) / len(sentence_nodes),
        "sample_node": sentence_nodes[0].text[:200] + "..." if sentence_nodes else ""
    }
    
    # Strategy 2: Token-based chunking
    token_splitter = TokenTextSplitter(
        chunk_size=256,
        chunk_overlap=50
    )
    token_nodes = token_splitter.get_nodes_from_documents(documents)
    strategies["token_256"] = {
        "node_count": len(token_nodes),
        "avg_node_size": sum(len(node.text) for node in token_nodes) / len(token_nodes),
        "sample_node": token_nodes[0].text[:200] + "..." if token_nodes else ""
    }
    
    # Strategy 3: Larger chunks for context
    large_splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=100
    )
    large_nodes = large_splitter.get_nodes_from_documents(documents)
    strategies["sentence_1024"] = {
        "node_count": len(large_nodes),
        "avg_node_size": sum(len(node.text) for node in large_nodes) / len(large_nodes),
        "sample_node": large_nodes[0].text[:200] + "..." if large_nodes else ""
    }
    
    # Strategy 4: Semantic chunking (if available)
    try:
        semantic_splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=1,
            breakpoint_percentile_threshold=95
        )
        semantic_nodes = semantic_splitter.get_nodes_from_documents(documents)
        strategies["semantic"] = {
            "node_count": len(semantic_nodes),
            "avg_node_size": sum(len(node.text) for node in semantic_nodes) / len(semantic_nodes),
            "sample_node": semantic_nodes[0].text[:200] + "..." if semantic_nodes else ""
        }
    except Exception as e:
        strategies["semantic"] = {
            "error": f"Semantic chunking failed: {str(e)}"
        }
    
    return strategies

def main():
    """Main analysis function"""
    
    # Path to BPI PDF (corrected path)
    pdf_path = Path(__file__).parent.parent / "server" / "BPI" / "BPI Product Data for RAG_.pdf"
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    # Analyze PDF content
    analysis = analyze_pdf_content(str(pdf_path))
    
    # Load documents for chunking tests
    loader = PDFReader()
    documents = loader.load_data(str(pdf_path))
    
    # Test chunking strategies
    chunking_results = test_chunking_strategies(documents)
    
    # Combine results
    full_analysis = {
        "pdf_analysis": analysis,
        "chunking_strategies": chunking_results,
        "recommendations": generate_recommendations(analysis, chunking_results)
    }
    
    # Save analysis results
    output_path = Path(__file__).parent / "pdf_analysis_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Analysis complete! Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total pages: {analysis['total_pages']}")
    print(f"Total characters: {analysis['total_characters']:,}")
    print(f"Total words: {analysis['total_words']:,}")
    print(f"Average page length: {analysis['average_page_length']:.0f} characters")
    
    print("\n📋 RECOMMENDED CHUNK SIZES:")
    for name, config in analysis['recommended_chunk_sizes'].items():
        print(f"  {name}: {config['size']} chars, overlap: {config.get('overlap', 'N/A')}")
    
    print("\n🧪 CHUNKING STRATEGY RESULTS:")
    for strategy, results in chunking_results.items():
        if 'error' not in results:
            print(f"  {strategy}: {results['node_count']} chunks, avg size: {results['avg_node_size']:.0f} chars")
        else:
            print(f"  {strategy}: {results['error']}")

def generate_recommendations(pdf_analysis: Dict, chunking_results: Dict) -> Dict[str, Any]:
    """Generate optimization recommendations based on analysis"""
    
    recommendations = {
        "optimal_strategy": "",
        "reasoning": "",
        "configuration": {},
        "implementation_notes": []
    }
    
    # Analyze document characteristics
    avg_page_length = pdf_analysis["average_page_length"]
    total_pages = pdf_analysis["total_pages"]
    
    # Choose optimal strategy based on content characteristics
    if avg_page_length < 1000:
        # Short pages - use medium chunks
        recommendations["optimal_strategy"] = "sentence_medium"
        recommendations["configuration"] = {
            "chunk_size": 768,
            "chunk_overlap": 75,
            "method": "SentenceSplitter"
        }
        recommendations["reasoning"] = "Short pages detected. Medium chunks provide good balance."
    elif avg_page_length > 3000:
        # Long pages - use smaller chunks for precision
        recommendations["optimal_strategy"] = "sentence_small"
        recommendations["configuration"] = {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "method": "SentenceSplitter"
        }
        recommendations["reasoning"] = "Long pages detected. Smaller chunks improve retrieval precision."
    else:
        # Medium pages - use balanced approach
        recommendations["optimal_strategy"] = "sentence_balanced"
        recommendations["configuration"] = {
            "chunk_size": 1024,
            "chunk_overlap": 100,
            "method": "SentenceSplitter"
        }
        recommendations["reasoning"] = "Balanced page length. Standard chunking approach recommended."
    
    # Add implementation notes
    recommendations["implementation_notes"] = [
        "Use SentenceSplitter for natural language boundaries",
        "Consider semantic chunking for complex documents",
        "Monitor retrieval quality and adjust chunk size if needed",
        "Use overlap to maintain context between chunks",
        f"Recommended top_k retrieval: {min(10, max(3, total_pages // 2))}"
    ]
    
    return recommendations

if __name__ == "__main__":
    main()
