# Embedding Model Configuration
# Choose based on your performance vs quality needs

# Available models (all are already downloaded):
EMBEDDING_MODELS = {
    # Fastest - Best for development/testing
    "fast": "sentence-transformers/all-MiniLM-L6-v2",
    
    # Balanced - Good speed + multilingual support  
    "balanced": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    
    # Best Quality - Slower but most accurate
    "quality": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    
    # Alternative fast option
    "e5_large": "intfloat/multilingual-e5-large"
}

# Current selection - change this to switch models
CURRENT_MODEL = "fast"  # Options: "fast", "balanced", "quality", "e5_large"

# Performance comparison:
# fast:     ~5x faster,   80MB,  excellent for English
# balanced: ~3x faster,   420MB, good multilingual support  
# quality:  baseline,     420MB, best accuracy
# e5_large: ~2x faster,   1.1GB, very good quality

def get_embedding_model():
    """Get the currently configured embedding model"""
    return EMBEDDING_MODELS[CURRENT_MODEL]

def get_model_info():
    """Get information about the current model"""
    return {
        "model": EMBEDDING_MODELS[CURRENT_MODEL],
        "profile": CURRENT_MODEL,
        "available_profiles": list(EMBEDDING_MODELS.keys())
    }
