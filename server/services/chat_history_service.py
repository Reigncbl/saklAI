"""
Chat history service for managing conversation context and history.
"""

import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path


class ChatHistoryService:
    """Service to manage chat history for users"""
    
    def __init__(self, storage_path: str = "./chat_history"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def _get_user_history_file(self, user_id: str) -> Path:
        """Get the file path for a user's chat history"""
        return self.storage_path / f"chat_history_{user_id}.json"
    
    def add_message(self, user_id: str, message: str, role: str, response: dict = None) -> None:
        """Add a message to user's chat history
        
        Args:
            user_id: User identifier
            message: The message content
            role: Either "user" or "assistant"
            response: Optional response data (for assistant messages)
        """
        try:
            history = self.get_history(user_id)
            
            # Create message entry
            message_entry = {
                "timestamp": datetime.now().isoformat(),
                "role": role,  # "user" or "assistant"
                "content": message,
                "response": response if role == "assistant" and response else None,
                "template_used": response.get("template_used") if response else None,
                "processing_method": response.get("processing_method") if response else None
            }
            
            history.append(message_entry)
            
            # Keep only last 50 messages to prevent files from getting too large
            if len(history) > 50:
                history = history[-50:]
            
            # Save to file
            history_file = self._get_user_history_file(user_id)
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving chat history for {user_id}: {e}")
    
    def get_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get user's chat history"""
        try:
            history_file = self._get_user_history_file(user_id)
            
            if not history_file.exists():
                return []
            
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            
            # Convert old format to new format for backward compatibility
            normalized_history = []
            for entry in history:
                # Handle old format (type/message) and new format (role/content)
                if "type" in entry and "message" in entry:
                    # Old format
                    normalized_entry = {
                        "timestamp": entry.get("timestamp"),
                        "role": entry["type"],
                        "content": entry["message"],
                        "response": entry.get("response"),
                        "template_used": entry.get("template_used"),
                        "processing_method": entry.get("processing_method")
                    }
                else:
                    # New format (already correct)
                    normalized_entry = entry
                
                normalized_history.append(normalized_entry)
            
            # Return last N messages
            return normalized_history[-limit:] if limit else normalized_history
            
        except Exception as e:
            print(f"Error loading chat history for {user_id}: {e}")
            return []
    
    def get_conversation_context(self, user_id: str, context_length: int = 5) -> str:
        """Get recent conversation context as a formatted string"""
        try:
            history = self.get_history(user_id, limit=context_length * 2)  # Get more to account for both user and assistant messages
            
            if not history:
                return ""
            
            context_messages = []
            for entry in history:
                if entry["role"] == "user":
                    context_messages.append(f"User: {entry['content']}")
                elif entry["role"] == "assistant":
                    # Use content field directly or extract from response
                    if entry.get("content"):
                        context_messages.append(f"Assistant: {entry['content']}")
                    elif entry.get("response", {}).get("suggestions"):
                        # Get the first suggestion as the assistant's response
                        suggestions = entry["response"]["suggestions"]
                        if suggestions and len(suggestions) > 0:
                            context_messages.append(f"Assistant: {suggestions[0].get('suggestion', '')}")
            
            # Return last few exchanges
            return "\n".join(context_messages[-context_length*2:])
            
        except Exception as e:
            print(f"Error getting conversation context for {user_id}: {e}")
            return ""
    
    def get_memory_context(self, user_id: str) -> str:
        """Get memory context for better conversation continuity"""
        try:
            history = self.get_history(user_id, limit=10)  # Get recent messages
            
            if not history:
                return ""
            
            # Extract key information from recent conversation
            user_messages = [h["content"] for h in history if h["role"] == "user"]
            
            # Identify conversation themes and user preferences
            topics_mentioned = []
            user_info = []
            
            for msg in user_messages:
                msg_lower = msg.lower()
                
                # Banking product interests
                if any(word in msg_lower for word in ["savings", "save", "deposit"]):
                    topics_mentioned.append("savings accounts")
                if any(word in msg_lower for word in ["credit card", "card", "credit"]):
                    topics_mentioned.append("credit cards")
                if any(word in msg_lower for word in ["loan", "borrow", "financing"]):
                    topics_mentioned.append("loans")
                if any(word in msg_lower for word in ["remit", "send money", "transfer", "ofw"]):
                    topics_mentioned.append("remittances")
                if any(word in msg_lower for word in ["online", "mobile", "app", "digital"]):
                    topics_mentioned.append("digital banking")
                
                # User characteristics
                if any(word in msg_lower for word in ["student", "college", "university"]):
                    user_info.append("student")
                if any(word in msg_lower for word in ["ofw", "overseas", "abroad"]):
                    user_info.append("OFW")
                if any(word in msg_lower for word in ["business", "entrepreneur", "company"]):
                    user_info.append("business owner")
                if any(word in msg_lower for word in ["senior", "retirement", "retired"]):
                    user_info.append("senior citizen")
            
            # Build memory context
            memory_parts = []
            
            if topics_mentioned:
                unique_topics = list(set(topics_mentioned))
                memory_parts.append(f"User has shown interest in: {', '.join(unique_topics)}")
            
            if user_info:
                unique_info = list(set(user_info))
                memory_parts.append(f"User profile: {', '.join(unique_info)}")
            
            # Add recent conversation summary
            if len(user_messages) > 0:
                latest_message = user_messages[-1]
                memory_parts.append(f"Latest inquiry: {latest_message}")
            
            return " | ".join(memory_parts) if memory_parts else ""
            
        except Exception as e:
            print(f"Error getting memory context for {user_id}: {e}")
            return ""
    
    def clear_history(self, user_id: str) -> bool:
        """Clear user's chat history"""
        try:
            history_file = self._get_user_history_file(user_id)
            if history_file.exists():
                history_file.unlink()
            return True
        except Exception as e:
            print(f"Error clearing chat history for {user_id}: {e}")
            return False
    
    def get_user_summary(self, user_id: str) -> Dict:
        """Get summary information about user's chat history"""
        try:
            history = self.get_history(user_id, limit=None)  # Get all history
            
            if not history:
                return {
                    "total_messages": 0,
                    "first_interaction": None,
                    "last_interaction": None,
                    "most_used_templates": [],
                    "conversation_topics": []
                }
            
            # Analyze history
            user_messages = [h for h in history if h["role"] == "user"]
            assistant_responses = [h for h in history if h["role"] == "assistant"]
            
            # Count template usage
            template_counts = {}
            for response in assistant_responses:
                template = response.get("template_used")
                if template:
                    template_counts[template] = template_counts.get(template, 0) + 1
            
            # Sort templates by usage
            most_used_templates = sorted(template_counts.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "total_messages": len(user_messages),
                "total_responses": len(assistant_responses),
                "first_interaction": history[0]["timestamp"] if history else None,
                "last_interaction": history[-1]["timestamp"] if history else None,
                "most_used_templates": most_used_templates[:5],  # Top 5
                "conversation_length": len(history)
            }
            
        except Exception as e:
            print(f"Error getting user summary for {user_id}: {e}")
            return {"error": str(e)}


# Global instance
chat_history_service = ChatHistoryService()
