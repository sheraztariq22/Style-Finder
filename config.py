"""
Configuration settings for the Style Finder application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google Gemini API configuration
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")  # Get from .env file
GEMINI_MODEL = "gemini-2.0-flash-exp"  # Using latest vision-capable model

# Image processing settings
IMAGE_SIZE = (224, 224)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

# Default similarity threshold
SIMILARITY_THRESHOLD = 0.8

# Number of alternatives to return from search
DEFAULT_ALTERNATIVES_COUNT = 5

# Generation parameters
TEMPERATURE = 0.2
TOP_P = 0.6
MAX_TOKENS = 2000