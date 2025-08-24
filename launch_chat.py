#!/usr/bin/env python3
"""
SaklAI Chat Launcher
Launches the SaklAI server and opens the chat interface in the browser.
"""

import subprocess
import time
import webbrowser
import sys
import os
import signal
from pathlib import Path


def check_port_available(port=8000):
    """Check if port is available"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False


def kill_existing_server():
    """Kill any existing Python processes that might be running the server"""
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                         capture_output=True, check=False)
        else:  # Unix/Linux/Mac
            subprocess.run(['pkill', '-f', 'server/main.py'], 
                         capture_output=True, check=False)
        time.sleep(1)  # Give time for processes to terminate
    except Exception as e:
        print(f"Warning: Could not kill existing processes: {e}")


def start_server():
    """Start the SaklAI server"""
    script_dir = Path(__file__).parent
    venv_python = script_dir / ".venv" / "Scripts" / "python.exe"
    server_script = script_dir / "server" / "main.py"
    
    # Check if virtual environment exists
    if not venv_python.exists():
        print(f"Error: Virtual environment not found at {venv_python}")
        print("Please ensure the virtual environment is set up correctly.")
        return None
    
    # Check if server script exists
    if not server_script.exists():
        print(f"Error: Server script not found at {server_script}")
        return None
    
    print("🚀 Starting SaklAI server...")
    
    # Start the server process
    try:
        process = subprocess.Popen([
            str(venv_python), 
            str(server_script)
        ], cwd=str(script_dir))
        
        return process
    except Exception as e:
        print(f"Error starting server: {e}")
        return None


def wait_for_server(max_attempts=30):
    """Wait for the server to be ready"""
    import urllib.request
    import urllib.error
    
    for attempt in range(max_attempts):
        try:
            urllib.request.urlopen('http://localhost:8000/health', timeout=1)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            time.sleep(1)
            print(f"⏳ Waiting for server to start... ({attempt + 1}/{max_attempts})")
    
    return False


def open_chat():
    """Open the chat interface in the default browser"""
    chat_url = "http://localhost:8000"
    print(f"🌐 Opening chat interface: {chat_url}")
    webbrowser.open(chat_url)


def main():
    """Main function to launch SaklAI chat"""
    print("=" * 50)
    print("🤖 SaklAI Chat Launcher")
    print("=" * 50)
    
    try:
        # Check if port is available, if not, kill existing processes
        if not check_port_available():
            print("⚠️  Port 8000 is busy. Terminating existing processes...")
            kill_existing_server()
            
            # Wait and check again
            time.sleep(2)
            if not check_port_available():
                print("❌ Could not free port 8000. Please manually stop any running servers.")
                return
        
        # Start the server
        server_process = start_server()
        if not server_process:
            print("❌ Failed to start server")
            return
        
        # Wait for server to be ready
        if wait_for_server():
            print("✅ Server is ready!")
            time.sleep(1)  # Give it an extra moment
            open_chat()
            
            print("\n" + "=" * 50)
            print("🎉 SaklAI Chat is now running!")
            print("📱 Chat interface: http://localhost:8000")
            print("🔧 Admin interface: http://localhost:8000/admin")
            print("💡 Press Ctrl+C to stop the server")
            print("=" * 50)
            
            # Wait for user to stop
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping SaklAI server...")
                server_process.terminate()
                try:
                    server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_process.kill()
                print("✅ Server stopped successfully!")
                
        else:
            print("❌ Server failed to start within expected time")
            server_process.terminate()
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    except KeyboardInterrupt:
        print("\n🛑 Launch cancelled by user")


if __name__ == "__main__":
    main()
