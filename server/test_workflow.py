#!/usr/bin/env python3
"""
Simple test for chat history service
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.chat_history_service import ChatHistoryService

def test_full_workflow():
    """Test a complete conversation workflow"""
    print("Testing Complete Chat History Workflow")
    print("=" * 40)
    
    chat_service = ChatHistoryService()
    user_id = "demo_user"
    
    # Clear any existing history
    chat_service.clear_history(user_id)
    print("✓ Cleared previous history")
    
    # Simulate a conversation
    conversations = [
        ("user", "Hi, I want to open a savings account"),
        ("assistant", "Hello! I'd be happy to help you with opening a savings account. BPI offers several savings account options with competitive interest rates and flexible terms."),
        ("user", "What are the requirements?"),
        ("assistant", "For our regular savings account, you'll need: 1) Valid ID, 2) Initial deposit of ₱500, 3) Filled application form. Would you like me to provide more details about our savings account features?"),
        ("user", "What about interest rates?"),
        ("assistant", "Our regular savings account offers 0.25% annual interest rate with daily interest computation. For higher interest rates, you might want to consider our BPI Maxi-Saver which offers up to 1.5% annually."),
        ("user", "How can I apply online?"),
        ("assistant", "You can start your application online through BPI Online or visit any BPI branch. For online applications, you'll need to schedule an appointment for document verification.")
    ]
    
    # Add each message to history
    print("\n📝 Simulating conversation...")
    for role, message in conversations:
        response_data = {
            "suggestions": [{"suggestion": message}],
            "template_used": "savings_accounts.yaml" if role == "assistant" else None,
            "processing_method": "rag" if role == "assistant" else None
        } if role == "assistant" else None
        
        chat_service.add_message(user_id, message, role, response_data)
        print(f"   {role}: {message[:60]}...")
    
    # Test all functions
    print(f"\n📊 Chat History Analysis:")
    
    # 1. Get full history
    history = chat_service.get_history(user_id)
    print(f"   Total messages: {len(history)}")
    
    # 2. Get memory context
    memory = chat_service.get_memory_context(user_id)
    print(f"   Memory context: {memory}")
    
    # 3. Get conversation context
    context = chat_service.get_conversation_context(user_id)
    print(f"   Conversation context: {context[:100]}...")
    
    # 4. Get user summary
    summary = chat_service.get_user_summary(user_id)
    print(f"   User summary: {summary}")
    
    # 5. Test context-aware response building
    print(f"\n🤖 Testing Context Building:")
    print(f"   For a new message: 'Can I get a credit card too?'")
    print(f"   Context would include: {memory}")
    print(f"   Recent conversation: {context}")
    
    print(f"\n✅ Workflow test completed successfully!")
    print(f"📁 History saved to: {chat_service._get_user_history_file(user_id)}")

if __name__ == "__main__":
    test_full_workflow()
