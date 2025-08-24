#!/usr/bin/env python3
"""
Simple SaklAI Chat Launcher
"""

import subprocess
import time
import webbrowser
import sys
import os
from pathlib import Path

def main():
    """Launch SaklAI chat"""
    print("🚀 Starting SaklAI Simple Chat...")
    
    # Get paths
    script_dir = Path(__file__).parent
    server_script = script_dir / "server" / "main_simple.py"
    
    # Check if server file exists
    if not server_script.exists():
        print(f"❌ Error: Server script not found at {server_script}")
        return
    
    # Kill any existing Python processes
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                         capture_output=True, check=False)
        time.sleep(1)
    except:
        pass
    
    try:
        # Start server
        print("⏳ Starting server...")
        
        # Change to server directory and run
        os.chdir(script_dir / "server")
        process = subprocess.Popen([
            sys.executable, "main_simple.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Wait for server to start
        for i in range(15):
            time.sleep(1)
            print(f"⏳ Waiting... ({i+1}/15)")
            
            # Check if server is ready
            try:
                import requests
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    break
            except:
                continue
        
        print("✅ Server is ready!")
        print("🌐 Opening chat: http://localhost:8000")
        
        # Open browser
        webbrowser.open("http://localhost:8000")
        
        print("\n" + "="*50)
        print("🎉 SaklAI Simple Chat is running!")
        print("📱 Chat: http://localhost:8000")
        print("💡 Press Ctrl+C to stop")
        print("="*50)
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        process.terminate()
        print("✅ Server stopped!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
