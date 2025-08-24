#!/usr/bin/env python3
"""
Test script for chat history functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.chat_history_service import ChatHistoryService

def test_chat_history():
    print("Testing Chat History Service...")
    
    # Initialize the service
    chat_service = ChatHistoryService()
    
    # Test user ID
    test_user = "test_user_123"
    
    # Test adding messages
    print(f"\n1. Adding messages for user: {test_user}")
    chat_service.add_message(test_user, "Hello, how can you help me with banking?", "user")
    chat_service.add_message(test_user, "I can help you with various banking services. What specifically would you like to know?", "assistant")
    chat_service.add_message(test_user, "I want to know about savings accounts", "user")
    chat_service.add_message(test_user, "Our savings accounts offer competitive interest rates and flexible terms...", "assistant")
    
    # Test getting history
    print("\n2. Getting chat history:")
    history = chat_service.get_history(test_user, limit=10)
    for i, msg in enumerate(history, 1):
        print(f"   {i}. [{msg['role']}] {msg['content'][:50]}...")
    
    # Test memory context
    print("\n3. Getting memory context:")
    memory_context = chat_service.get_memory_context(test_user)
    print(f"   User Background: {memory_context}")
    
    # Test conversation context
    print("\n4. Getting conversation context:")
    conv_context = chat_service.get_conversation_context(test_user, context_length=3)
    print(f"   Recent Context: {conv_context}")
    
    # Test user summary
    print("\n5. Getting user summary:")
    summary = chat_service.get_user_summary(test_user)
    print(f"   Summary: {summary}")
    
    print("\n✓ Chat history test completed successfully!")

if __name__ == "__main__":
    test_chat_history()
