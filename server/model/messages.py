from sqlmodel import SQLModel, Field
from datetime import datetime

class Message(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    sender_id: str
    receiver_id: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
