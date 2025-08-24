"""
RAG (Retrieval Augmented Generation) related business logic
"""

import os
from pathlib import Path
from dto.models import SuggestionRequest
from services.rag import suggestion_generation
from services.classification_service import classify_with_langchain_agent, should_use_rag
from services.translation_service import translate_to_english
from services.response_service import generate_direct_groq_response
from services.chat_history_service import chat_history_service

# Configuration constants
DEFAULT_PROMPT_FILE = "config.yaml"
RAG_STORE_PATH = "./rag_store"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


async def process_rag_suggestion(request: SuggestionRequest):
    """
    Process RAG suggestion request with intelligent classification and response generation
    """
    current_dir = Path(__file__).parent.parent
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
        result = await _process_rag_response(request, yaml_path, api_key, conversation_context, memory_context, yaml_file)
    else:
        result = await _process_direct_response(request, yaml_path, api_key, conversation_context, memory_context, yaml_file)
    
    if isinstance(result, dict) and result.get("status") == "error":
        return result
    
    # Build response data
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
    assistant_message = _extract_assistant_message(result)
    chat_history_service.add_message(
        user_id=request.user_id,
        message=assistant_message,
        role="assistant",
        response=response_data
    )
    
    return response_data


async def _process_rag_response(request, yaml_path, api_key, conversation_context, memory_context, yaml_file):
    """Process response using RAG system"""
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
        
        return result
        
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


async def _process_direct_response(request, yaml_path, api_key, conversation_context, memory_context, yaml_file):
    """Process response using direct Groq API"""
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
        return result
        
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


def _extract_assistant_message(result):
    """Extract assistant message from result for chat history"""
    assistant_message = ""
    if result and len(result) > 0:
        if isinstance(result, list) and len(result) > 0:
            assistant_message = result[0].get('suggestion', '') if isinstance(result[0], dict) else str(result[0])
        elif isinstance(result, dict):
            assistant_message = result.get('response', result.get('content', str(result)))
        else:
            assistant_message = str(result)
    return assistant_message
