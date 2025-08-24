from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files from the public directory
current_dir = Path(__file__).parent
public_dir = current_dir / "public"
app.mount("/static", StaticFiles(directory=public_dir), name="static")

@app.get("/")
async def serve_chat():
    """Serve the chat interface"""
    return FileResponse(public_dir / "chat.html")

@app.post("/rag/suggestions")
async def test_suggestions(request: dict):
    """Test endpoint that returns a simple response"""
    print(f"Received request: {request}")
    
    return {
        "status": "success",
        "suggestions": [
            {
                "category": "Test",
                "suggestion": "This is a test response from the server!"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
