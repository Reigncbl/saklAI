"""
Chat history related API routes
"""

from fastapi import APIRouter, HTTPException
from dto.models import ChatHistoryResponse, UserSummaryResponse
from services.chat_history_service import chat_history_service

router = APIRouter()


@router.get("/history/{user_id}", response_model=ChatHistoryResponse)
async def get_chat_history(user_id: str, limit: int = 20):
    """Get chat history for a specific user"""
    try:
        history = chat_history_service.get_history(user_id, limit)
        return ChatHistoryResponse(
            user_id=user_id,
            history=history,
            total_count=len(history)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")


@router.delete("/history/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear chat history for a specific user"""
    try:
        success = chat_history_service.clear_history(user_id)
        if success:
            return {"status": "success", "message": f"Chat history cleared for user {user_id}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear chat history")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing chat history: {str(e)}")


@router.get("/summary/{user_id}", response_model=UserSummaryResponse)
async def get_user_summary(user_id: str):
    """Get summary statistics for a user's chat history"""
    try:
        summary = chat_history_service.get_user_summary(user_id)
        if "error" in summary:
            raise HTTPException(status_code=500, detail=summary["error"])
        
        return UserSummaryResponse(
            user_id=user_id,
            **summary
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user summary: {str(e)}")


@router.get("/context/{user_id}")
async def get_conversation_context(user_id: str, length: int = 5):
    """Get conversation context for a user"""
    try:
        context = chat_history_service.get_conversation_context(user_id, length)
        return {
            "user_id": user_id,
            "context": context,
            "context_length": length
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversation context: {str(e)}")
