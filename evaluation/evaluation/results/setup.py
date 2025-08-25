#!/usr/bin/env python3
"""
Setup script for SaklAI Evaluation Suite
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install additional dependencies for evaluation"""
    print("📦 Installing evaluation dependencies...")
    
    # Install additional packages
    requirements = [
        "psutil>=5.9.0",
        "pandas>=2.0.0"
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    current_dir = Path(__file__).parent
    
    # Create results directory
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"✅ Created {results_dir}")
    
    return True

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment...")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  No virtual environment detected (recommended to use one)")
    
    # Check for .env file
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("❌ .env file not found - please create one with your API key")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 SaklAI Evaluation Suite Setup")
    print("=" * 40)
    
    success = True
    success &= check_environment()
    success &= setup_directories()
    success &= install_dependencies()
    
    print("\n" + "=" * 40)
    
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Ensure your .env file has the correct API key")
        print("2. Run: python test_setup.py")
        print("3. Run: python run_evaluation.py --mode functional")
    else:
        print("❌ Setup failed. Please fix the issues above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
