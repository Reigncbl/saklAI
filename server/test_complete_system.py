#!/usr/bin/env python3
"""
Test script for the complete chat system with history
"""

import requests
import json
import time

BASE_URL = "http://localhost:8006"

def test_health():
    """Test if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_chat_endpoints():
    """Test chat history endpoints"""
    test_user = "test_api_user"
    
    # Test getting empty history
    try:
        response = requests.get(f"{BASE_URL}/chat/history/{test_user}")
        print(f"Empty history: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Get history failed: {e}")
    
    # Test getting user summary
    try:
        response = requests.get(f"{BASE_URL}/chat/summary/{test_user}")
        print(f"User summary: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Get summary failed: {e}")
    
    # Test conversation context
    try:
        response = requests.get(f"{BASE_URL}/chat/context/{test_user}")
        print(f"Conversation context: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Get context failed: {e}")

def test_message_with_history():
    """Test sending a message and check if history is saved"""
    test_user = "test_conversation_user"
    
    # Send a test message
    message_data = {
        "user_id": test_user,
        "message": "Hello, I need help with savings accounts",
        "prompt_type": "auto"
    }
    
    try:
        print("Sending test message...")
        response = requests.post(f"{BASE_URL}/rag/suggestions", json=message_data, timeout=30)
        print(f"Message response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result.get('status')}")
            print(f"Template used: {result.get('template_used')}")
            print(f"Processing method: {result.get('processing_method')}")
            if result.get('suggestions'):
                print(f"First suggestion: {result['suggestions'][0][:100]}...")
        else:
            print(f"Error response: {response.text}")
        
        # Check if history was saved
        time.sleep(1)  # Give it a moment to save
        history_response = requests.get(f"{BASE_URL}/chat/history/{test_user}")
        if history_response.status_code == 200:
            history = history_response.json()
            print(f"History saved: {len(history.get('history', []))} messages")
            for msg in history.get('history', []):
                print(f"  - {msg.get('role')}: {msg.get('content', '')[:50]}...")
        
    except Exception as e:
        print(f"Message test failed: {e}")

def main():
    print("Testing Chat System with History Integration")
    print("=" * 50)
    
    # Wait for server to be ready
    print("Waiting for server to start...")
    for i in range(10):
        if test_health():
            print("✓ Server is ready!")
            break
        print(f"  Attempt {i+1}/10...")
        time.sleep(3)
    else:
        print("✗ Server failed to start in time")
        return
    
    print("\n1. Testing Chat History Endpoints...")
    test_chat_endpoints()
    
    print("\n2. Testing Message with History Integration...")
    test_message_with_history()
    
    print("\n✓ Testing completed!")

if __name__ == "__main__":
    main()
