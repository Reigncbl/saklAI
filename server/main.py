from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlmodel import Session, select
from typing import List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import uvicorn
import os
from pathlib import Path

from model.messages import Message
from utils.con_db import get_session, init_db
from dto.models import SendMessageDTO, MessageResponse, ConversationSummary, SuggestionRequest, ChatHistoryResponse, UserSummaryResponse

# RAG functionality
from services.rag import suggestion_generation

# Import separated services
from services.classification_service import classify_with_langchain_agent, should_use_rag
from services.translation_service import translate_to_english
from services.response_service import generate_direct_groq_response
from services.chat_history_service import chat_history_service

# Configuration constants
DEFAULT_PROMPT_FILE = "config.yaml"
RAG_STORE_PATH = "./rag_store"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    yield
    # Shutdown (if needed)

app = FastAPI(title="SaklAI Chat API", version="1.0.0", lifespan=lifespan)

# Mount static files from the public directory
current_dir = Path(__file__).parent
public_dir = current_dir.parent / "public"
app.mount("/static", StaticFiles(directory=public_dir), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve chat interface
@app.get("/")
async def chat_interface():
    """Serve the chat interface"""
    current_dir = Path(__file__).parent
    chat_file = current_dir.parent / "public" / "chat.html"
    return FileResponse(chat_file)

# Core endpoints
@app.post("/messages/send", response_model=MessageResponse)
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

@app.get("/messages/all", response_model=List[MessageResponse])
def get_all_messages(limit: int = 100, session: Session = Depends(get_session)):
    """Get all messages (for admin)"""
    statement = select(Message).order_by(Message.timestamp.desc()).limit(limit)
    return session.exec(statement).all()

@app.get("/messages/conversation/{user1}/{user2}", response_model=List[MessageResponse])
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

@app.get("/conversations/list", response_model=List[ConversationSummary])
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

@app.get("/conversations/active", response_model=List[ConversationSummary])
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

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "message": "Server is running"}

# Chat history endpoints - keeping the properly typed versions

@app.post("/rag/suggestions")
async def get_rag_suggestions(request: SuggestionRequest):
    """
    Intelligent RAG endpoint - automatically detects inquiry type and returns 
    BPI banking suggestions based on PDF content using the most appropriate template.
    """
    try:
        current_dir = Path(__file__).parent
        prompts_dir = current_dir / "Prompts"
        
        # Get API key from environment
        api_key = os.getenv("api_key")
        if not api_key:
            return {
                "status": "error", 
                "message": "API key not configured",
                "fallback_suggestions": [
                    {"analysis": "Error", "category": "System", "suggestion": "Please contact support for assistance."}
                ]
            }
        
        # Translate message to English if needed
        original_message = request.message
        if request.message:
            translated_message = translate_to_english(request.message, api_key)
            # Use the translated message for classification and RAG processing
            request.message = translated_message
        
        # Add user message to chat history
        if request.message:
            chat_history_service.add_message(
                user_id=request.user_id,
                message=original_message,
                role="user"
            )
        
        # Get conversation context and memory for better responses
        conversation_context = chat_history_service.get_conversation_context(request.user_id)
        memory_context = chat_history_service.get_memory_context(request.user_id)
        
        # Log context for debugging
        if conversation_context:
            print(f"Conversation context: {conversation_context[:200]}...")
        if memory_context:
            print(f"Memory context: {memory_context}")
        
        # Determine which template to use
        yaml_file = DEFAULT_PROMPT_FILE  # Default to config.yaml
        classification_info = None
        use_rag = False  # Flag to determine if we should use RAG or direct Groq
        
        if request.prompt_type == "auto" and request.message:
            # Use LangChain agent for intelligent classification
            try:
                classification = await classify_with_langchain_agent(request.message, api_key)
                yaml_file = classification["template"]
                classification_info = classification
                
                # Determine if we should use RAG based on the template
                use_rag = should_use_rag(yaml_file)
                
                # Log the classification decision
                print(f"LangChain Classification: {classification['category']} -> {classification['template']} (confidence: {classification['confidence']}, method: {classification['method']}, use_rag: {use_rag})")
                
            except Exception as e:
                print(f"LangChain classification error: {e}")
                yaml_file = DEFAULT_PROMPT_FILE  # Fallback to default
                use_rag = False
                
        else:
            # Use manual prompt type selection (backward compatibility)
            prompt_files = {
                "config": DEFAULT_PROMPT_FILE,
                "investing_equities": "investing_equities.yaml", 
                "investing_funds": "investing_funds.yaml",
                "investing_portfolio": "investing_portfolio.yaml",
                "savings_accounts": "savings_accounts.yaml",
                "credit_cards": "credit_cards.yaml",
                "loans": "loans.yaml",
                "remittances_ofw": "remittances_ofw.yaml",
                "digital_banking": "digital_banking.yaml",
                "account_services": "account_services.yaml",
                "general_banking": "general_banking.yaml"
            }
            yaml_file = prompt_files.get(request.prompt_type, DEFAULT_PROMPT_FILE)
        
        yaml_path = prompts_dir / yaml_file
        
        if not yaml_path.exists():
            return {
                "status": "error",
                "message": f"Prompt file not found: {yaml_file}",
                "fallback_suggestions": [
                    {"analysis": "Error", "category": "System", "suggestion": "Please contact support for assistance."}
                ]
            }

        # Handle response generation based on whether we need RAG or not
        if use_rag:
            # Use RAG system for specific banking topics
            print(f"Using RAG system for {yaml_file}")
            
            # Get conversation context and memory for better responses
            context_parts = []
            if memory_context:
                context_parts.append(f"User background: {memory_context}")
            if conversation_context:
                context_parts.append(f"Recent conversation:\n{conversation_context}")
            context = "\n".join(context_parts) if context_parts else ""
            
            # Import embedding configuration
            try:
                from config.embedding_config import get_embedding_model
                embedding_model = get_embedding_model()
            except ImportError:
                # Fallback to fast model if config not available
                embedding_model = DEFAULT_EMBEDDING_MODEL
            
            try:
                result = await suggestion_generation(
                    user_id=request.user_id,
                    yaml_path=str(yaml_path),
                    groq_api_key=api_key,
                    vector_store_path=RAG_STORE_PATH,
                    reset_index=False,  # Don't reset index by default for better performance
                    top_k=5,
                    embedding_model=embedding_model,
                    conversation_context=context  # Pass context to RAG
                )
            except Exception as rag_error:
                print(f"RAG generation error: {rag_error}")
                error_response = {
                    "status": "error",
                    "message": f"RAG processing failed: {str(rag_error)}",
                    "fallback_suggestions": [
                        {"analysis": "Error", "category": "System", "suggestion": "I'm having trouble processing your request right now. Please try again or contact support."}
                    ]
                }
                # Save error response to history
                chat_history_service.add_message(
                    user_id=request.user_id,
                    message="I'm having trouble processing your request right now. Please try again or contact support.",
                    role="assistant",
                    response=error_response
                )
                return error_response
            
            # Check if there was an error in RAG processing
            if "error" in result:
                error_response = {
                    "status": "error",
                    "message": result.get("detail", "RAG processing failed"),
                    "error_details": result,
                    "fallback_suggestions": [
                        {"analysis": "Error", "category": "System", "suggestion": "I'm experiencing technical difficulties. Please try again or contact support."}
                    ]
                }
                # Save error response to history
                chat_history_service.add_message(
                    user_id=request.user_id,
                    message="I'm experiencing technical difficulties. Please try again or contact support.",
                    role="assistant",
                    response=error_response
                )
                return error_response
                
        else:
            # Use direct Groq response for general conversation
            print(f"Using direct Groq response for {yaml_file}")
            
            # Get conversation context and memory for better responses
            context_parts = []
            if memory_context:
                context_parts.append(f"User background: {memory_context}")
            if conversation_context:
                context_parts.append(f"Recent conversation:\n{conversation_context}")
            context = "\n".join(context_parts) if context_parts else ""
            
            try:
                result = await generate_direct_groq_response(
                    message=request.message,
                    yaml_path=str(yaml_path),
                    groq_api_key=api_key,
                    conversation_context=context  # Pass context to direct response
                )
            except Exception as groq_error:
                print(f"Direct Groq generation error: {groq_error}")
                error_response = {
                    "status": "error",
                    "message": f"Response generation failed: {str(groq_error)}",
                    "fallback_suggestions": [
                        {"analysis": "Error", "category": "System", "suggestion": "I'm having trouble processing your request right now. Please try again or contact support."}
                    ]
                }
                # Save error response to history
                chat_history_service.add_message(
                    user_id=request.user_id,
                    message="I'm having trouble processing your request right now. Please try again or contact support.",
                    role="assistant",
                    response=error_response
                )
                return error_response

        # This block is now handled within the if/else for use_rag
        # The result from either RAG or direct Groq is already in the 'result' variable
        
        response_data = {
            "status": "success",
            "user_id": request.user_id,
            "message": original_message,  # Return the original message
            "translated_message": request.message if request.message != original_message else None,
            "template_used": yaml_file,
            "suggestions": result,
            "processing_method": "rag" if use_rag else "direct_groq"
        }
        
        # Include classification info if available
        if classification_info:
            response_data["classification"] = {
                "detected_category": classification_info["category"],
                "confidence": classification_info["confidence"],
                "method": classification_info["method"]
            }
        
        # Save assistant response to chat history
        # Get the first suggestion as the main response text
        assistant_message = ""
        if result and len(result) > 0:
            if isinstance(result, list) and len(result) > 0:
                assistant_message = result[0].get('suggestion', '') if isinstance(result[0], dict) else str(result[0])
            elif isinstance(result, dict):
                assistant_message = result.get('response', result.get('content', str(result)))
            else:
                assistant_message = str(result)
        
        chat_history_service.add_message(
            user_id=request.user_id,
            message=assistant_message,
            role="assistant",
            response=response_data
        )
        
        return response_data
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "fallback_suggestions": [
                {"analysis": "Error", "category": "System", "suggestion": "I'm experiencing technical difficulties. Please try again or contact support."}
            ]
        }


@app.get("/chat/history/{user_id}", response_model=ChatHistoryResponse)
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


@app.delete("/chat/history/{user_id}")
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


@app.get("/chat/summary/{user_id}", response_model=UserSummaryResponse)
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


@app.get("/chat/context/{user_id}")
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

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )
