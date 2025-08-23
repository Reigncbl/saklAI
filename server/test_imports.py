#!/usr/bin/env python3
"""
Test script to check if all imports work correctly
"""

print("Testing imports...")

try:
    from fastapi import FastAPI
    print("✓ FastAPI imported successfully")
except ImportError as e:
    print(f"✗ FastAPI import failed: {e}")

try:
    from langchain.agents import initialize_agent
    print("✓ LangChain imported successfully")
except ImportError as e:
    print(f"✗ LangChain import failed: {e}")

try:
    from langchain_groq import ChatGroq
    print("✓ LangChain Groq imported successfully")
except ImportError as e:
    print(f"✗ LangChain Groq import failed: {e}")

try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    print("✓ LlamaIndex imported successfully")
except ImportError as e:
    print(f"✗ LlamaIndex import failed: {e}")

try:
    import torch
    print("✓ PyTorch imported successfully")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

print("Import test completed!")
