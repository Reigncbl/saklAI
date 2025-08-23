from fastapi import FastAPI, Depends, HTTPException
from sqlmodel import Session, select
from typing import List
from datetime import datetime
from model.messages import Message
from utils.con_db import get_session, init_db
from pydantic import BaseModel
import uvicorn
app = FastAPI()

# ✅ DTOs
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

# ✅ Initialize DB on startup
@app.on_event("startup")
def on_startup():
    init_db()

# ✅ Send message
@app.post("/messages/send", response_model=MessageResponse)
def send_message(payload: SendMessageDTO, session: Session = Depends(get_session)):
    msg = Message(
        sender_id=payload.sender_id,
        receiver_id=payload.receiver_id,
        message=payload.message
    )
    session.add(msg)
    session.commit()
    session.refresh(msg)
    return msg

# ✅ Get conversation between two users
@app.get("/messages/conversation/{user1}/{user2}", response_model=List[MessageResponse])
def get_conversation(user1: str, user2: str, session: Session = Depends(get_session)):
    statement = select(Message).where(
        ((Message.sender_id == user1) & (Message.receiver_id == user2)) |
        ((Message.sender_id == user2) & (Message.receiver_id == user1))
    ).order_by(Message.timestamp)

    results = session.exec(statement).all()
    if not results:
        raise HTTPException(status_code=404, detail="No conversation found")
    return results

# ✅ Get all messages (for admin/debugging)
@app.get("/messages/all", response_model=List[MessageResponse])
def get_all_messages(limit: int = 100, session: Session = Depends(get_session)):
    """Get all messages with optional limit"""
    statement = select(Message).order_by(Message.timestamp.desc()).limit(limit)
    results = session.exec(statement).all()
    return results

# ✅ Get messages by sender
@app.get("/messages/from/{sender}", response_model=List[MessageResponse])
def get_messages_by_sender(sender: str, session: Session = Depends(get_session)):
    """Get all messages sent by a specific user"""
    statement = select(Message).where(Message.sender_id == sender).order_by(Message.timestamp.desc())
    results = session.exec(statement).all()
    if not results:
        raise HTTPException(status_code=404, detail=f"No messages found from {sender}")
    return results

# ✅ Get messages by receiver
@app.get("/messages/to/{receiver}", response_model=List[MessageResponse])
def get_messages_by_receiver(receiver: str, session: Session = Depends(get_session)):
    """Get all messages received by a specific user"""
    statement = select(Message).where(Message.receiver_id == receiver).order_by(Message.timestamp.desc())
    results = session.exec(statement).all()
    if not results:
        raise HTTPException(status_code=404, detail=f"No messages found for {receiver}")
    return results

# ✅ Get recent messages (last N messages)
@app.get("/messages/recent", response_model=List[MessageResponse])
def get_recent_messages(count: int = 10, session: Session = Depends(get_session)):
    """Get the most recent messages"""
    statement = select(Message).order_by(Message.timestamp.desc()).limit(count)
    results = session.exec(statement).all()
    return results

# ✅ Add CORS middleware for frontend
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve static files (for your admin interface)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "..", "public")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    @app.get("/admin")
    def admin_page():
        """Serve the admin interface"""
        admin_file = os.path.join(static_path, "admin.html")
        if os.path.exists(admin_file):
            return FileResponse(admin_file)
        raise HTTPException(status_code=404, detail="Admin page not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
