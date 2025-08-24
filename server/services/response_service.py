"""
Response generation service for handling direct Groq responses.
"""

import json
import os
import yaml
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
model = os.getenv('model')   

async def generate_direct_groq_response(message: str, yaml_path: str, groq_api_key: str, conversation_context: str = "") -> list:
    """
    Generate response using direct Groq API call without RAG, using the prompt from YAML
    """
    try:
        # Load the prompt from YAML file
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        prompt_template = config.get("prompt", "").strip()
        
        if not prompt_template:
            raise ValueError("Prompt template missing or empty in YAML config")
        
        # Check if this is the config.yaml (default template) for single response
        is_config_yaml = yaml_path.endswith("config.yaml")
        
        # Create the full prompt with the user's message and context
        context_section = f"\n\nConversation Context:\n{conversation_context}\n" if conversation_context else ""
        full_prompt = f"{prompt_template}{context_section}\n\nUser message: \"{message}\"\n\nProvide your response as a JSON array following the specified format."
        
        # Use Groq client directly
        groq_client = Groq(api_key=groq_api_key)
        
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Debug logging
        print(f"Raw Groq response: {response_text[:200]}...")
        
        # Parse JSON response
        try:
            # Clean up the response text
            if response_text.startswith("```"):
                response_text = response_text.strip("`").lstrip("json").strip()
            
            # Try to extract JSON from the response if it contains additional text
            if '[' in response_text and ']' in response_text:
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                json_part = response_text[start:end]
                print(f"Extracted JSON part: {json_part}")
                parsed_response = json.loads(json_part)
            else:
                print(f"Parsing full response as JSON: {response_text}")
                parsed_response = json.loads(response_text)
            
            # Ensure it's a list
            if not isinstance(parsed_response, list):
                parsed_response = [parsed_response]
            
            # For config.yaml, ensure only 1 response is returned
            if is_config_yaml and len(parsed_response) > 1:
                print(f"Config.yaml returned {len(parsed_response)} responses, taking only the first one")
                parsed_response = [parsed_response[0]]
            
            print(f"Final parsed response: {parsed_response}")
            return parsed_response
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw response: {response_text}")
            # Fallback response
            return [
                {
                    "analysis": "conversation",
                    "category": "General Response",
                    "suggestion": "I understand what you're saying. How can I help you with your banking needs today?"
                }
            ]
            
    except Exception as e:
        print(f"Direct Groq response error: {e}")
        # Fallback response
        return [
            {
                "analysis": "error",
                "category": "System Error", 
                "suggestion": "I'm having trouble processing your request. Please try again or let me know how I can help with your banking needs."
            }
        ]
