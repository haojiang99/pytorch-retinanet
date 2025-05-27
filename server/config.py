"""
Configuration settings for the mammogram analysis server
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# API keys and credentials
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# If there's no API key in environment, check if there's a key file
if not GEMINI_API_KEY:
    key_file_path = os.path.join(os.path.dirname(__file__), 'gemini_api_key.txt')
    if os.path.exists(key_file_path):
        with open(key_file_path, 'r') as f:
            GEMINI_API_KEY = f.read().strip()

# Server configuration
SERVER_HOST = os.environ.get('SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.environ.get('SERVER_PORT', 5300))
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'True').lower() == 'true'

# Default model configuration
MODEL_CONFIDENCE_THRESHOLD = float(os.environ.get('MODEL_CONFIDENCE_THRESHOLD', 0.1))

# Path configuration
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_final.pt'))

# Gemini configuration
USE_GEMINI_BY_DEFAULT = os.environ.get('USE_GEMINI_BY_DEFAULT', 'True').lower() == 'true'
