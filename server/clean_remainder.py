# Temporary backup of remaining good code
from line 850 onward, let me rewrite cleanly

@app.get("/chat/recommendations/{user_id}")
async def get_ai_message_recommendations(user_id: str):
    """Get context-aware AI message recommendations for a specific user conversation"""
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
        if file_path.exists():
            try:
                data = await read_json_file_async(str(file_path))
                
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

# Helper function for async GROQ API calls with enhanced error handling
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

# Helper function for concurrent file reading
async def read_multiple_chat_files_async(file_paths: List[str]) -> List[dict]:
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

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
