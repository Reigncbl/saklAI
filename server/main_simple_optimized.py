# Add these imports for database support
from sqlalchemy import create_engine, Column, String, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import or_, and_
from datetime import datetime

import asyncio
import aiofiles
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os
import json
import yaml
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=4)

# Async file operations helpers
async def read_json_file_async(file_path: Path) -> dict:
    """Async JSON file reader with error handling"""
    try:
        if not file_path.exists():
            return {"history": [], "status": "active"}
        
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return {"history": [], "status": "active"}

async def write_json_file_async(file_path: Path, data: dict) -> bool:
    """Async JSON file writer with error handling"""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=2))
        return True
    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}")
        return False

# File batching for bulk operations
async def read_multiple_chat_files_async(chat_history_dir: Path) -> List[dict]:
    """Read multiple chat files concurrently"""
    if not chat_history_dir.exists():
        return []
    
    json_files = [f for f in chat_history_dir.glob("*.json") if f.name.startswith("chat_history_")]
    
    # Read files concurrently
    tasks = []
    for file_path in json_files:
        task = asyncio.create_task(_read_single_chat_file(file_path))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and empty results
    conversations = []
    for result in results:
        if isinstance(result, dict) and result.get("user_id"):
            conversations.append(result)
    
    return conversations

async def _read_single_chat_file(file_path: Path) -> dict:
    """Helper to read a single chat file"""
    try:
        data = await read_json_file_async(file_path)
        
        # Handle both old and new formats
        if isinstance(data, list):
            history = data
            status = "active"
        else:
            history = data.get("history", [])
            status = data.get("status", "active")
        
        if not history:
            return {}
        
        user_id = file_path.name.replace("chat_history_", "").replace(".json", "")
        last_entry = history[-1] if history else {}
        timestamp = last_entry.get("timestamp", "")
        
        # Find last user message
        last_message = ""
        for entry in reversed(history):
            if entry.get("role") == "user":
                last_message = entry.get("content", "")
                break
        
        return {
            "user_id": user_id,
            "last_timestamp": timestamp,
            "last_message": last_message,
            "message_count": len(history),
            "status": status,
            "history": history
        }
    except Exception as e:
        logger.error(f"Error reading chat file {file_path}: {e}")
        return {}

# Simple data models
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    status: str
    suggestions: list

class StatusUpdate(BaseModel):
    status: str

class AgentMessage(BaseModel):
    message: str

# Input validation helpers
def validate_user_id(user_id: str) -> str:
    """Validate and sanitize user ID"""
    if not user_id or len(user_id) > 50:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    # Remove any potentially dangerous characters
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', user_id)
    if not sanitized:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    return sanitized

# Connection pooling for Groq API
from functools import lru_cache

@lru_cache(maxsize=1)
def get_groq_client():
    """Create a singleton Groq client"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not configured")
    return Groq(api_key=api_key)

async def async_groq_call(prompt: str, temperature: float = 0.7, max_tokens: int = 1000):
    """Async wrapper for Groq API calls"""
    try:
        client = get_groq_client()
        
        # Run the blocking API call in thread pool
        response = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API error: {e}")
        raise

# Helper function for concurrent file reading
async def read_multiple_files_async(file_paths: List[str]) -> List[dict]:
    """Read multiple files concurrently"""
    tasks = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            task = asyncio.create_task(_read_single_file_async(file_path))
            tasks.append(task)
    
    if not tasks:
        return []
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and return valid data
    valid_results = []
    for result in results:
        if isinstance(result, dict) and result:
            valid_results.append(result)
    
    return valid_results

async def _read_single_file_async(file_path: str) -> dict:
    """Helper to read a single file asynchronously"""
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return {}

# Enhanced async GROQ API calls with RAG support
async def make_groq_request_async(api_key: str, conversation_context: str, knowledge_context: str, user_intent: str, conversation_summary: str):
    """Enhanced async GROQ API call for RAG-powered recommendations"""
    try:
        # Build enhanced prompt with RAG context
        enhanced_prompt = f"""You are an AI assistant helping human banking agents provide personalized customer service for BPI.

Context Analysis:
- User Intent: {user_intent}
- Conversation Summary: {conversation_summary}

Recent Conversation:
{conversation_context}

Knowledge Base Context:
{knowledge_context}

Generate exactly 3 contextual message recommendations in this JSON format:
[
  {{
    "type": "CLARIFY",
    "message": "A specific clarifying question based on the conversation",
    "reasoning": "Why this clarification would help"
  }},
  {{
    "type": "PRODUCT_RECOMMENDATION", 
    "message": "A BPI product recommendation tailored to their needs",
    "reasoning": "Why this product fits their situation"
  }},
  {{
    "type": "NEXT_STEPS",
    "message": "A specific next action to move toward resolution",
    "reasoning": "How this progresses their request"
  }}
]

Guidelines:
1. Be specific to this customer's conversation
2. Reference conversation details when possible
3. Suggest appropriate BPI products/services
4. Maintain professional, helpful tone
5. Focus on actionable responses

Return only valid JSON."""

        # Make async API call
        client = get_groq_client()
        response = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=0.3,
                max_tokens=1200
            )
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            if response_text.startswith("```"):
                response_text = response_text.strip("`").lstrip("json").strip()
            recommendations = json.loads(response_text)
            
            if isinstance(recommendations, list) and len(recommendations) > 0:
                return {
                    "status": "success",
                    "recommendations": recommendations,
                    "context": {
                        "user_intent": user_intent,
                        "conversation_length": len(conversation_context),
                        "knowledge_available": bool(knowledge_context.strip()),
                        "processing_method": "groq_rag_enhanced"
                    }
                }
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
        
        # Fallback if parsing fails
        return get_fallback_recommendations(user_intent, conversation_context[-200:])
        
    except Exception as e:
        print(f"Groq API error: {e}")
        return get_fallback_recommendations(user_intent, conversation_context[-200:])

def get_fallback_recommendations(user_intent, last_message):
    """Provide fallback recommendations when AI service is unavailable"""
    base_recommendations = {
        "loan_inquiry": [
            {
                "type": "CLARIFY",
                "message": "What specific type of loan are you interested in? (Personal, Auto, Home, Business)",
                "reasoning": "Understanding loan type helps provide targeted information"
            },
            {
                "type": "PRODUCT_RECOMMENDATION",
                "message": "Based on your inquiry, I'd recommend checking our Personal Loan with competitive rates starting at 5.99% annually.",
                "reasoning": "Personal loans are most common and have competitive rates"
            },
            {
                "type": "NEXT_STEPS",
                "message": "Would you like me to schedule a consultation with our loan specialist to discuss your specific requirements?",
                "reasoning": "Personal consultation ensures proper loan matching"
            }
        ],
        "savings_investment": [
            {
                "type": "CLARIFY",
                "message": "Are you looking for short-term savings or long-term investment options?",
                "reasoning": "Investment timeframe determines product recommendations"
            },
            {
                "type": "PRODUCT_RECOMMENDATION",
                "message": "Consider our BPI Save Up account with higher interest rates for your savings goals.",
                "reasoning": "Save Up offers competitive returns for growing savings"
            },
            {
                "type": "NEXT_STEPS",
                "message": "I can help you open a savings account online or schedule a branch visit. Which would you prefer?",
                "reasoning": "Offering convenient options for account opening"
            }
        ]
    }
    
    return {
        "status": "success",
        "recommendations": base_recommendations.get(user_intent, base_recommendations["loan_inquiry"]),
        "source": "fallback_templates"
    }

# Create FastAPI app
app = FastAPI(title="SaklAI Simple Chat", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
current_dir = Path(__file__).parent
public_dir = current_dir.parent / "public"
app.mount("/static", StaticFiles(directory=public_dir), name="static")

# Also serve files directly from public directory
app.mount("/css", StaticFiles(directory=public_dir / "css"), name="css")
app.mount("/js", StaticFiles(directory=public_dir / "js"), name="js")

@app.get("/")
async def root():
    """Serve the chat interface"""
    chat_path = public_dir / "chat.html"
    if chat_path.exists():
        return FileResponse(chat_path)
    return {"message": "Chat interface not found"}

@app.get("/admin")
async def admin():
    """Serve the admin interface"""
    admin_path = public_dir / "admin.html"
    if admin_path.exists():
        return FileResponse(admin_path)
    return {"message": "Admin interface not found"}

@app.get("/admin.html")
async def admin_html():
    """Serve admin.html directly"""
    return await admin()

@app.get("/favicon.ico")
async def favicon():
    """Return empty response for favicon to prevent 404 errors"""
    return {"message": "No favicon"}

@app.get("/chat.html")
async def chat_html():
    """Serve chat.html directly"""
    return await root()

@app.post("/rag/suggestions")
async def get_suggestions(request: ChatRequest):
    """Optimized chat endpoint with async operations"""
    try:
        # Validate input
        user_id = validate_user_id(request.user_id)
        if not request.message or len(request.message.strip()) > 10000:
            raise HTTPException(status_code=400, detail="Invalid message")
        
        from datetime import datetime
        
        # Get GROQ API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return {
                "status": "error",
                "message": "GROQ API key not configured",
                "suggestions": [{"suggestion": "Please configure the GROQ API key."}]
            }
        
        # Load prompt template
        prompt_path = current_dir / "Prompts" / "config.yaml"
        try:
            async with aiofiles.open(prompt_path, "r", encoding="utf-8") as f:
                config_content = await f.read()
                config = yaml.safe_load(config_content)
            prompt_template = config.get("prompt", "You are a helpful banking assistant.")
        except:
            prompt_template = "You are SaklAI, a helpful banking assistant for BPI. Provide helpful and professional responses."
        
        # Read chat history asynchronously
        chat_history_dir = Path(__file__).parent.parent / "chat_history"
        file_path = chat_history_dir / f"chat_history_{user_id}.json"
        
        user_history = []
        if file_path.exists():
            try:
                data = await read_json_file_async(file_path)
                
                # Handle both old and new formats
                if isinstance(data, list):
                    history = data
                else:
                    history = data.get("history", [])
                
                # Get last 6 messages for context
                user_history = history[-6:] if len(history) > 6 else history
            except:
                user_history = []
        
        # Build conversation context
        context = ""
        for entry in user_history:
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role == "user":
                context += f"User: {content}\n"
            elif role in ["assistant", "agent"]:
                context += f"Assistant: {content}\n"
        
        # Create prompt
        full_prompt = f"{prompt_template}\n\nConversation Context:\n{context}\n\nUser message: \"{request.message.strip()}\"\n\nProvide your response as a JSON array with this format: [{{\"analysis\": \"inquiry\", \"category\": \"Information\", \"suggestion\": \"your helpful response\"}}]"
        
        # Call Groq API asynchronously
        response_text = await async_groq_call(full_prompt, temperature=0.7, max_tokens=1000)
        
        # Parse JSON response
        try:
            if response_text.startswith("```"):
                response_text = response_text.strip("`").lstrip("json").strip()
            suggestions = json.loads(response_text)
        except:
            # Fallback if JSON parsing fails
            suggestions = [{"analysis": "response", "category": "Information", "suggestion": response_text}]
        
        # Append to chat history file asynchronously
        timestamp = datetime.now().isoformat()
        
        # Prepare the data structure
        if file_path.exists():
            data = await read_json_file_async(file_path)
            
            # Convert old format to new format if needed
            if isinstance(data, list):
                data = {"history": data, "status": "active"}
        else:
            data = {"history": [], "status": "active"}
        
        # Add user message
        user_entry = {
            "timestamp": timestamp,
            "role": "user",
            "content": request.message.strip(),
            "response": None,
            "template_used": None,
            "processing_method": None
        }
        data["history"].append(user_entry)
        
        # Add assistant response
        assistant_content = suggestions[0].get("suggestion", response_text) if suggestions else response_text
        assistant_entry = {
            "timestamp": datetime.now().isoformat(),
            "role": "assistant",
            "content": assistant_content,
            "response": {
                "status": "success",
                "user_id": user_id,
                "message": request.message.strip(),
                "translated_message": None,
                "template_used": "config.yaml",
                "suggestions": suggestions,
                "processing_method": "async_groq"
            },
            "template_used": "config.yaml",
            "processing_method": "async_groq"
        }
        data["history"].append(assistant_entry)
        
        # Keep only last 50 messages
        if len(data["history"]) > 50:
            data["history"] = data["history"][-50:]
        
        # Write back to file asynchronously
        await write_json_file_async(file_path, data)
        
        return {
            "status": "success",
            "suggestions": suggestions
        }
        
    except Exception as e:
        print(f"Chat error: {e}")
        return {
            "status": "error",
            "message": f"Error: {str(e)}",
            "suggestions": [{"suggestion": "I'm experiencing technical difficulties. Please try again."}]
        }

@app.get("/chat/history/{user_id}")
async def get_chat_history(user_id: str):
    """Get chat history for a user - optimized version"""
    try:
        user_id = validate_user_id(user_id)
        chat_history_dir = Path(__file__).parent.parent / "chat_history"
        file_path = chat_history_dir / f"chat_history_{user_id}.json"
        
        if not file_path.exists():
            return {
                "user_id": user_id,
                "history": [],
                "total_count": 0
            }
        
        data = await read_json_file_async(file_path)
        
        # Handle both old and new formats
        if isinstance(data, list):
            history = data
        else:
            history = data.get("history", [])
        
        # Format for frontend
        formatted_history = []
        for entry in history:
            role = entry.get("role", "")
            content = entry.get("content", "")
            timestamp = entry.get("timestamp", "")
            
            formatted_history.append({
                "type": role,
                "message": content,
                "timestamp": timestamp
            })
        
        return {
            "user_id": user_id,
            "history": formatted_history,
            "total_count": len(formatted_history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")

@app.delete("/chat/history/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear chat history for a user - optimized version"""
    try:
        user_id = validate_user_id(user_id)
        chat_history_dir = Path(__file__).parent.parent / "chat_history"
        file_path = chat_history_dir / f"chat_history_{user_id}.json"
        
        if file_path.exists():
            # Clear the history but keep the file structure
            data = {"history": [], "status": "active"}
            await write_json_file_async(file_path, data)
        
        return {"status": "success", "message": f"Chat history cleared for user {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {str(e)}")

@app.post("/chat/recommendations/{user_id}")
async def get_message_recommendations(user_id: str):
    """Get context-aware AI message recommendations - optimized version"""
    try:
        user_id = validate_user_id(user_id)
        from datetime import datetime
        
        # Get GROQ API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return {
                "status": "error",
                "message": "GROQ API key not configured",
                "recommendations": []
            }
        
        # Read user's conversation history asynchronously
        chat_history_dir = Path(__file__).parent.parent / "chat_history"
        file_path = chat_history_dir / f"chat_history_{user_id}.json"
        
        conversation_context = ""
        if file_path.exists():
            try:
                data = await read_json_file_async(file_path)
                
                # Handle both old and new formats
                if isinstance(data, list):
                    history = data
                else:
                    history = data.get("history", [])
                
                # Build comprehensive conversation context
                for entry in history:
                    role = entry.get("role", "")
                    content = entry.get("content", "")
                    timestamp = entry.get("timestamp", "")
                    
                    if role == "user":
                        conversation_context += f"Customer ({timestamp}): {content}\n"
                    elif role in ["assistant", "agent"]:
                        agent_type = "AI Assistant" if role == "assistant" else "Human Agent"
                        conversation_context += f"{agent_type} ({timestamp}): {content}\n"
                
            except Exception as e:
                print(f"Error reading chat history: {e}")
                conversation_context = "No conversation history available."
        else:
            conversation_context = "No conversation history found for this customer."
        
        # Create a prompt for generating context-aware recommendations
        prompt = f"""You are an AI assistant helping a human banking agent provide excellent customer service. 
        
Based on the customer's conversation history below, generate 3 relevant and helpful message recommendations that the human agent can use to continue the conversation effectively.

Customer Conversation History:
{conversation_context}

Generate exactly 3 recommendations in this JSON format:
[
  {{
    "category": "CLARIFY",
    "message": "A clarifying question to better understand the customer's needs",
    "reasoning": "Why this response would be helpful"
  }},
  {{
    "category": "PRODUCT_RECOMMENDATION", 
    "message": "A relevant product or service recommendation based on the conversation",
    "reasoning": "Why this product fits the customer's needs"
  }},
  {{
    "category": "NEXT_STEPS",
    "message": "A concrete next step to move the conversation forward",
    "reasoning": "How this helps progress the customer's request"
  }}
]

Focus on:
- Banking and financial services context
- BPI (Bank of the Philippine Islands) products and services
- Professional, helpful, and personalized responses
- Actionable next steps
- Building customer trust and satisfaction

Respond only with the JSON array, no additional text."""

        # Call Groq API asynchronously
        response_text = await async_groq_call(prompt, temperature=0.7, max_tokens=1500)
        
        # Parse JSON response
        try:
            if response_text.startswith("```"):
                response_text = response_text.strip("`").lstrip("json").strip()
            recommendations = json.loads(response_text)
        except Exception as e:
            print(f"JSON parsing error: {e}")
            # Fallback recommendations if parsing fails
            recommendations = [
                {
                    "category": "CLARIFY",
                    "message": "Could you provide more details about your specific banking needs so I can assist you better?",
                    "reasoning": "Gathering more information helps provide personalized service"
                },
                {
                    "category": "PRODUCT_RECOMMENDATION",
                    "message": "Based on your inquiry, I'd recommend exploring our comprehensive banking solutions.",
                    "reasoning": "Offering relevant products based on customer interest"
                },
                {
                    "category": "NEXT_STEPS", 
                    "message": "Would you like me to schedule a consultation or provide more information about our services?",
                    "reasoning": "Moving the conversation toward concrete action"
                }
            ]
        
        return {
            "status": "success",
            "user_id": user_id,
            "recommendations": recommendations,
            "conversation_summary": f"Analyzed {len(conversation_context.split('Customer'))-1 if conversation_context else 0} customer messages",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Recommendation error: {e}")
        return {
            "status": "error",
            "message": f"Error generating recommendations: {str(e)}",
            "recommendations": []
        }

@app.get("/chat/active")
async def get_active_chats():
    """Get active chats for admin interface - optimized version"""
    try:
        chat_history_dir = Path(__file__).parent.parent / "chat_history"
        conversations = await read_multiple_chat_files_async(chat_history_dir)
        
        # Sort by last_timestamp descending (newest first)
        conversations.sort(key=lambda c: c.get("last_timestamp", ""), reverse=True)
        return conversations
        
    except Exception as e:
        logger.error(f"Error reading chat histories: {e}")
        return []

@app.post("/chat/status/{user_id}")
async def set_chat_status(user_id: str, status: dict):
    """Set chat status for a user - optimized version"""
    try:
        user_id = validate_user_id(user_id)
        chat_history_dir = Path(__file__).parent.parent / "chat_history"
        file_path = chat_history_dir / f"chat_history_{user_id}.json"
        
        # Read existing file asynchronously
        data = await read_json_file_async(file_path)
        
        # Convert old format to new format if needed
        if isinstance(data, list):
            data = {"history": data, "status": status.get("status", "active")}
        else:
            data["status"] = status.get("status", "active")
        
        # Write back to file asynchronously
        success = await write_json_file_async(file_path, data)
        
        if success:
            return {"status": "success", "user_id": user_id, "new_status": status.get("status", "active")}
        else:
            raise HTTPException(status_code=500, detail="Failed to write status update")
    except Exception as e:
        logger.error(f"Error setting chat status for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set chat status: {str(e)}")

@app.post("/chat/message/{user_id}")
async def add_agent_message(user_id: str, message: AgentMessage):
    """Add an agent message to chat history - optimized version"""
    try:
        # Validate inputs
        user_id = validate_user_id(user_id)
        if not message.message or len(message.message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        if len(message.message) > 10000:
            raise HTTPException(status_code=400, detail="Message too long")
        
        from datetime import datetime
        
        message_text = message.message.strip()
        chat_history_dir = Path(__file__).parent.parent / "chat_history"
        file_path = chat_history_dir / f"chat_history_{user_id}.json"
        
        if not file_path.exists():
            # Create new file with empty history
            data = {"history": [], "status": "assigned"}
        else:
            # Read existing file asynchronously
            data = await read_json_file_async(file_path)
            
            # Convert old format to new format if needed
            if isinstance(data, list):
                data = {"history": data, "status": "assigned"}
        
        # Append the new message with enhanced metadata
        data["history"].append({
            "role": "assistant",
            "content": message_text,
            "timestamp": datetime.now().isoformat(),
            "type": "agent_message"
        })
        
        # Write back to file asynchronously
        await write_json_file_async(file_path, data)
        
        return {
            "status": "success", 
            "message": "Agent message added successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error adding agent message: {e}")
        raise HTTPException(status_code=500, detail="Failed to add agent message")

@app.get("/chat/recommendations-rag/{user_id}")
async def get_ai_message_recommendations_with_rag(user_id: str):
    """Get context-aware AI message recommendations using RAG pipeline - optimized version"""
    try:
        # Validate user ID
        user_id = validate_user_id(user_id)
        
        from datetime import datetime
        
        # Get GROQ API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return {
                "status": "error",
                "message": "GROQ API key not configured",
                "recommendations": []
            }
        
        # Read user's conversation history asynchronously
        chat_history_dir = Path(__file__).parent.parent / "chat_history"
        file_path = chat_history_dir / f"chat_history_{user_id}.json"
        
        conversation_context = ""
        user_intent = "general_inquiry"
        last_user_message = ""
        conversation_summary = ""
        
        if file_path.exists():
            try:
                data = await read_json_file_async(file_path)
                
                # Handle both old and new formats
                if isinstance(data, dict) and "history" in data:
                    messages = data["history"]
                elif isinstance(data, list):
                    messages = data
                else:
                    messages = []
                
                # Extract conversation context
                if messages:
                    # Get last few messages for context
                    recent_messages = messages[-10:] if len(messages) > 10 else messages
                    conversation_context = "\n".join([
                        f"{'User' if msg.get('role') == 'user' else 'Agent'}: {msg.get('content', '')}"
                        for msg in recent_messages if msg.get('content')
                    ])
                    
                    # Get last user message
                    user_messages = [msg for msg in messages if msg.get('role') == 'user']
                    if user_messages:
                        last_user_message = user_messages[-1].get('content', '')
                    
                    # Create conversation summary
                    conversation_summary = f"Conversation with {len(messages)} total messages. "
                    if user_messages:
                        conversation_summary += f"User's last message: {last_user_message[:100]}..."
                
            except Exception as e:
                print(f"Error reading chat history: {e}")
                # Continue with empty context
        
        # Read RAG store files concurrently for knowledge base
        rag_files = [
            f"c:/Users/John Carlo/saklAI-1/rag_store_customer_{user_id}.json"
        ]
        
        # Also include some general RAG files for broader context
        base_dir = Path(__file__).parent.parent
        general_rag_files = list(base_dir.glob("rag_store_customer_*.json"))[:5]  # Limit to 5 for performance
        rag_files.extend([str(f) for f in general_rag_files if str(f) not in rag_files])
        
        # Read RAG data concurrently
        rag_data_list = await read_multiple_files_async(rag_files)
        knowledge_context = ""
        
        for rag_data in rag_data_list:
            if rag_data and isinstance(rag_data, dict):
                # Extract relevant information from RAG data
                if "conversations" in rag_data:
                    for conv in rag_data["conversations"][:3]:  # Limit for context size
                        if "summary" in conv:
                            knowledge_context += f"Knowledge: {conv['summary']}\n"
                elif "knowledge" in rag_data:
                    knowledge_context += f"Knowledge: {str(rag_data['knowledge'])[:200]}\n"
        
        # Determine user intent based on context
        if "problem" in last_user_message.lower() or "issue" in last_user_message.lower():
            user_intent = "technical_support"
        elif "question" in last_user_message.lower() or "how" in last_user_message.lower():
            user_intent = "information_request"
        elif "thank" in last_user_message.lower() or "bye" in last_user_message.lower():
            user_intent = "conversation_closing"
        
        # Make async GROQ API request
        response_data = await make_groq_request_async(
            groq_api_key, 
            conversation_context, 
            knowledge_context, 
            user_intent, 
            conversation_summary
        )
        
        return response_data
        
    except Exception as e:
        print(f"Error getting AI recommendations: {e}")
        return {
            "status": "error",
            "message": f"Failed to get recommendations: {str(e)}",
            "recommendations": []
        }

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "main_simple_optimized:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
