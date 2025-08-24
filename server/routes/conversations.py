"""
Conversation-related API routes
"""

from fastapi import APIRouter, Depends
from sqlmodel import Session, select
from typing import List
from datetime import datetime, timedelta

from model.messages import Message
from utils.con_db import get_session
from dto.models import ConversationSummary

router = APIRouter()


@router.get("/list", response_model=List[ConversationSummary])
def get_conversations_list(session: Session = Depends(get_session)):
    """Get list of all conversations with summary info"""
    # Get unique conversation pairs
    statement = select(Message).order_by(Message.timestamp.desc())
    all_messages = session.exec(statement).all()
    
    conversations = {}
    for msg in all_messages:
        # Create a consistent conversation key (alphabetically ordered)
        participants = sorted([msg.sender_id, msg.receiver_id])
        conv_key = f"{participants[0]}-{participants[1]}"
        
        if conv_key not in conversations:
            conversations[conv_key] = {
                "participant1": participants[0],
                "participant2": participants[1],
                "last_message": msg.message,
                "last_timestamp": msg.timestamp,
                "message_count": 1
            }
        else:
            conversations[conv_key]["message_count"] += 1
            # Keep the latest message
            if msg.timestamp > conversations[conv_key]["last_timestamp"]:
                conversations[conv_key]["last_message"] = msg.message
                conversations[conv_key]["last_timestamp"] = msg.timestamp
    
    return list(conversations.values())


@router.get("/active", response_model=List[ConversationSummary])
def get_active_conversations(hours: int = 24, session: Session = Depends(get_session)):
    """Get conversations that have been active in the last N hours"""
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    statement = select(Message).where(Message.timestamp >= cutoff_time).order_by(Message.timestamp.desc())
    recent_messages = session.exec(statement).all()
    
    conversations = {}
    for msg in recent_messages:
        participants = sorted([msg.sender_id, msg.receiver_id])
        conv_key = f"{participants[0]}-{participants[1]}"
        
        if conv_key not in conversations:
            conversations[conv_key] = {
                "participant1": participants[0],
                "participant2": participants[1],
                "last_message": msg.message,
                "last_timestamp": msg.timestamp,
                "message_count": 1
            }
        else:
            conversations[conv_key]["message_count"] += 1
            if msg.timestamp > conversations[conv_key]["last_timestamp"]:
                conversations[conv_key]["last_message"] = msg.message
                conversations[conv_key]["last_timestamp"] = msg.timestamp
    
    return list(conversations.values())
