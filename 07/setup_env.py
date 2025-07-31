#!/usr/bin/env python3
"""
Helper script to set up the .env file with your Gemini API key.
"""

import os

def create_env_file():
    """Create a .env file with the Gemini API key."""
    env_content = """# Gemini API Configuration
# Replace 'your_gemini_api_key_here' with your actual Gemini API key
GEMINI_API_KEY=your_gemini_api_key_here
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file")
    print("📝 Please edit the .env file and replace 'your_gemini_api_key_here' with your actual Gemini API key")
    print("🔑 You can get your Gemini API key from: https://makersuite.google.com/app/apikey")

if __name__ == "__main__":
    if os.path.exists('.env'):
        print("⚠️  .env file already exists")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("❌ Setup cancelled")
            exit(0)
    
    create_env_file() 