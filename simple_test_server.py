from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    uvicorn.run(app, host="127.0.0.1", port=8001)
