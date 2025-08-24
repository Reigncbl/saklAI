"""
RAG (Retrieval Augmented Generation) routes
"""

from fastapi import APIRouter, HTTPException
from dto.models import SuggestionRequest
from business.rag_processor import process_rag_suggestion

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/suggestions")
async def get_suggestions(request: SuggestionRequest):
    """
    Generate intelligent suggestions using RAG (Retrieval Augmented Generation) system
    with automatic classification and response routing.
    """
    try:
        result = await process_rag_suggestion(request)
        
        if result.get("status") == "error":
            # Return error response with fallback suggestions
            return result
        
        return result
        
    except Exception as e:
        print(f"RAG suggestions error: {e}")
        return {
            "status": "error",
            "message": f"RAG processing failed: {str(e)}",
            "fallback_suggestions": [
                {"analysis": "Error", "category": "System", "suggestion": "I'm experiencing technical difficulties. Please try again or contact support."}
            ]
        }
