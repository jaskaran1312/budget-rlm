#!/usr/bin/env python3
"""
Setup script for environment variables
Helps create and manage .env file for the RLM Document Analyzer
"""

import os
import sys
from pathlib import Path


def create_env_file():
    """Create a .env file with template values."""
    env_content = """# Environment variables for RLM Document Analyzer
# Add your actual API key below

# Required: Your Gemini API key
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: RLM Configuration
RLM_MAX_ITERATIONS=10
RLM_MAX_RECURSION_DEPTH=3
RLM_MODEL_NAME=gemini-2.5-flash
RLM_LOG_LEVEL=INFO
RLM_LOG_FILE=
RLM_MAX_DOCUMENT_SIZE=10485760
RLM_TIMEOUT_SECONDS=300
"""
    
    env_file = Path('.env')
    
    if env_file.exists():
        print(f".env file already exists at {env_file.absolute()}")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("Keeping existing .env file")
            return
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"Created .env file at {env_file.absolute()}")
        print("\nPlease edit the .env file and add your actual GEMINI_API_KEY")
        print("You can get your API key from: https://ai.google.dev/")
    except Exception as e:
        print(f"Error creating .env file: {e}")


def check_env_file():
    """Check if .env file exists and has required variables."""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("No .env file found")
        return False
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
        
        if 'GEMINI_API_KEY=your_gemini_api_key_here' in content:
            print("⚠️  .env file exists but still has template API key")
            print("Please edit .env file and add your actual GEMINI_API_KEY")
            return False
        
        if 'GEMINI_API_KEY=' in content and 'your_gemini_api_key_here' not in content:
            print("✅ .env file looks good - API key appears to be set")
            return True
        
        print("⚠️  .env file exists but API key format is unclear")
        return False
        
    except Exception as e:
        print(f"Error reading .env file: {e}")
        return False


def test_api_key():
    """Test if the API key is working."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key or api_key == 'your_gemini_api_key_here':
            print("❌ No valid API key found")
            return False
        
        # Try to import and test the Gemini client
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            print("✅ API key appears to be valid")
            return True
        except Exception as e:
            print(f"❌ API key test failed: {e}")
            return False
            
    except ImportError:
        print("⚠️  python-dotenv not installed, cannot test API key")
        return False


def main():
    """Main setup function."""
    print("RLM Document Analyzer - Environment Setup")
    print("=" * 50)
    
    # Check if .env file exists and is properly configured
    if check_env_file():
        print("\nTesting API key...")
        if test_api_key():
            print("\n🎉 Setup complete! You can now run the RLM Document Analyzer")
            return
    
    print("\nSetting up environment file...")
    create_env_file()
    
    print("\nNext steps:")
    print("1. Edit the .env file and add your actual GEMINI_API_KEY")
    print("2. Get your API key from: https://ai.google.dev/")
    print("3. Run this script again to test your setup")
    print("4. Run: python main.py --help to see usage options")


if __name__ == "__main__":
    main()
