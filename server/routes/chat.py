"""
Chat history related API routes
"""
import os, json
from fastapi import APIRouter, HTTPException
from dto.models import ChatHistoryResponse, UserSummaryResponse
from services.chat_history_service import chat_history_service
from datetime import datetime
from pathlib import Path

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

CHAT_HISTORY_DIR = Path(__file__).parent.parent.parent / "chat_history"

@router.get("/chat/active")
async def get_active_chats():
    """Summarize all active chats from JSON history files"""
    try:
        conversations = []

        for filename in os.listdir(CHAT_HISTORY_DIR):
            if not filename.endswith(".json"):
                continue

            path = CHAT_HISTORY_DIR / filename
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Support both old (list) and new (dict) formats
            if isinstance(data, list):
                history = data
                status = "active"  # default for legacy
            else:
                history = data.get("history", [])
                status = data.get("status", "active")

            if not history or status != "active":
                continue


            # use last entry timestamp
            last_entry = history[-1]
            timestamp = last_entry.get("timestamp")

            conversations.append({
                "user_id": filename.replace(".json", ""),
                "last_timestamp": timestamp,
                "message_count": len(history),
                "history": history  # Include full chat history
            })

        # Sort by last_timestamp ascending (oldest first)
        conversations.sort(key=lambda c: c["last_timestamp"] or "")
        return conversations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read chat histories: {str(e)}")

# Endpoint to set chat status (top-level field) for a user
from fastapi import Body

@router.post("/chat/status/{user_id}")
async def set_chat_status(user_id: str, status: str = Body(..., embed=True)):
    """Set the top-level status field for a user's chat history file"""
    try:
        path = CHAT_HISTORY_DIR / f"{user_id}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Chat history file not found")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            # Upgrade to dict format
            data = {"history": data, "status": status}
        else:
            data["status"] = status
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {"status": "success", "user_id": user_id, "new_status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set chat status: {str(e)}")
