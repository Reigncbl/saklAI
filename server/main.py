from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import uvicorn
from pathlib import Path

from utils.con_db import init_db

# Import route modules
from routes.messages import router as messages_router
from routes.conversations import router as conversations_router
from routes.chat import router as chat_router
from routes.rag import router as rag_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    yield
    # Shutdown (if needed)


app = FastAPI(title="SaklAI Chat API", version="1.0.0", lifespan=lifespan)

# Register route modules
app.include_router(messages_router)
app.include_router(conversations_router)
app.include_router(chat_router)
app.include_router(rag_router)

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


@app.get("/")
async def root():
    """Serve the main chat.html file"""
    try:
        public_dir = Path(__file__).parent.parent / "public"
        chat_path = public_dir / "chat.html"
        if chat_path.exists():
            return FileResponse(chat_path)
        else:
            return {"message": "Welcome to SaklAI Chat API - chat.html not found"}
    except Exception as e:
        return {"message": f"Error serving chat.html: {str(e)}"}


@app.get("/admin")
async def admin():
    """Serve the admin interface"""
    try:
        public_dir = Path(__file__).parent.parent / "public"
        admin_path = public_dir / "admin.html"
        if admin_path.exists():
            return FileResponse(admin_path)
        else:
            return {"message": "Admin interface not found"}
    except Exception as e:
        return {"message": f"Error serving admin interface: {str(e)}"}


@app.get("/admin.html")
async def admin_html():
    """Serve the admin interface with .html extension"""
    return await admin()


@app.get("/chat.html")
async def chat_html():
    """Serve the chat interface with .html extension"""
    try:
        public_dir = Path(__file__).parent.parent / "public"
        chat_path = public_dir / "chat.html"
        if chat_path.exists():
            return FileResponse(chat_path)
        else:
            return {"message": "Chat interface not found"}
    except Exception as e:
        return {"message": f"Error serving chat interface: {str(e)}"}


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "message": "Server is running"}


if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,  # Set to True for development
        log_level="info"
    )
