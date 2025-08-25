from fastapi import FastAPI, HTTPException
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

# Simple data models
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    status: str
    suggestions: list

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

# Simple chat history storage (in-memory)
chat_history = {}

@app.get("/")
async def root():
    """Serve the chat interface"""
    chat_path = public_dir / "chat.html"
    if chat_path.exists():
        return FileResponse(chat_path)
    return {"message": "Chat interface not found"}

@app.get("/chat.html")
async def chat_html():
    """Serve chat.html directly"""
    return await root()

@app.post("/rag/suggestions")
async def get_suggestions(request: ChatRequest):
    """Simple chat endpoint that returns AI responses"""
    try:
        # Get GROQ API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return {
                "status": "error",
                "message": "GROQ API key not configured",
                "suggestions": [{"suggestion": "Please configure the GROQ API key."}]
            }
        
        # Load simple prompt
        prompt_path = current_dir / "Prompts" / "config.yaml"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            prompt_template = config.get("prompt", "You are a helpful banking assistant.")
        except:
            prompt_template = "You are SaklAI, a helpful banking assistant for BPI. Provide helpful and professional responses."
        
        # Get conversation context
        user_history = chat_history.get(request.user_id, [])
        context = "\n".join([f"User: {h['message']}\nAssistant: {h['response']}" for h in user_history[-3:]])
        
        # Create prompt
        full_prompt = f"{prompt_template}\n\nConversation Context:\n{context}\n\nUser message: \"{request.message}\"\n\nProvide your response as a JSON array with this format: [{{\"analysis\": \"inquiry\", \"category\": \"Information\", \"suggestion\": \"your helpful response\"}}]"
        
        # Call Groq API
        groq_client = Groq(api_key=groq_api_key)
        response = groq_client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            if response_text.startswith("```"):
                response_text = response_text.strip("`").lstrip("json").strip()
            suggestions = json.loads(response_text)
        except:
            # Fallback if JSON parsing fails
            suggestions = [{"analysis": "response", "category": "Information", "suggestion": response_text}]
        
        # Save to chat history
        if request.user_id not in chat_history:
            chat_history[request.user_id] = []
        
        chat_history[request.user_id].append({
            "message": request.message,
            "response": suggestions[0].get("suggestion", response_text)
        })
        
        # Keep only last 10 messages
        chat_history[request.user_id] = chat_history[request.user_id][-10:]
        
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
    """Get chat history for a user"""
    history = chat_history.get(user_id, [])
    formatted_history = []
    
    for entry in history:
        formatted_history.append({"type": "user", "message": entry["message"]})
        formatted_history.append({"type": "assistant", "message": entry["response"]})
    
    return {
        "user_id": user_id,
        "history": formatted_history,
        "total_count": len(formatted_history)
    }

@app.delete("/chat/history/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear chat history for a user"""
    if user_id in chat_history:
        del chat_history[user_id]
    return {"status": "success", "message": f"Chat history cleared for user {user_id}"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Server is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
