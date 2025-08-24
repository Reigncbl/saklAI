"""
Message-related API routes
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import List
from datetime import datetime, timedelta

from model.messages import Message
from utils.con_db import get_session
from dto.models import SendMessageDTO, MessageResponse, ConversationSummary

router = APIRouter()


@router.post("/send", response_model=MessageResponse)
def send_message(payload: SendMessageDTO, session: Session = Depends(get_session)):
    """Send a new message"""
    msg = Message(
        sender_id=payload.sender_id,
        receiver_id=payload.receiver_id,
        message=payload.message
    )
    session.add(msg)
    session.commit()
    session.refresh(msg)
    return msg


@router.get("/all", response_model=List[MessageResponse])
def get_all_messages(limit: int = 100, session: Session = Depends(get_session)):
    """Get all messages (for admin)"""
    statement = select(Message).order_by(Message.timestamp.desc()).limit(limit)
    return session.exec(statement).all()


@router.get("/conversation/{user1}/{user2}", response_model=List[MessageResponse])
def get_conversation(user1: str, user2: str, session: Session = Depends(get_session)):
    """Get conversation between two users"""
    statement = select(Message).where(
        ((Message.sender_id == user1) & (Message.receiver_id == user2)) |
        ((Message.sender_id == user2) & (Message.receiver_id == user1))
    ).order_by(Message.timestamp)
    
    results = session.exec(statement).all()
    if not results:
        raise HTTPException(status_code=404, detail="No conversation found")
    return results
