"""
Data models for the SaklAI Chat API.
"""

from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional, Any


class SendMessageDTO(BaseModel):
    sender_id: str
    receiver_id: str
    message: str


class MessageResponse(BaseModel):
    id: int
    sender_id: str
    receiver_id: str
    message: str
    timestamp: datetime


class ConversationSummary(BaseModel):
    participant1: str
    participant2: str
    last_message: str
    last_timestamp: datetime
    message_count: int


class SuggestionRequest(BaseModel):
    user_id: str
    message: str = ""  # User's inquiry message for automatic classification
    prompt_type: str = "auto"  # "auto" for automatic detection, or specific type
    include_context: bool = True  # Whether to include conversation context


class ChatHistoryEntry(BaseModel):
    timestamp: str
    type: str  # "user" or "assistant"
    message: str
    response: Optional[Dict] = None
    template_used: Optional[str] = None
    processing_method: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    user_id: str
    history: List[ChatHistoryEntry]
    total_count: int


class UserSummaryResponse(BaseModel):
    user_id: str
    total_messages: int
    total_responses: int
    first_interaction: Optional[str]
    last_interaction: Optional[str]
    most_used_templates: List[tuple]
    conversation_length: int
