"""
Configuration File for RAG-based Constitution Q&A Agent

Why do we need a config file?
- Centralizes all settings in one place (easier to maintain)
- Separates configuration from code logic (better organization)
- Makes it easy to change settings without modifying the main code
- Follows best practices for software development
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This reads the OPENAI_API_KEY from the .env file we created
load_dotenv()

# ============================================
# API Configuration
# ============================================

# OpenAI API Key
# This is your secret key to access OpenAI's services
# It's loaded from the .env file for security
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Name
# We're using GPT-4o-mini which is a smaller, faster, and cheaper version
# It's perfect for this use case and provides good quality responses
MODEL_NAME = "gpt-4o-mini"

# Temperature Setting
# Temperature controls randomness in responses (0.0 to 2.0)
# 0.0 = very deterministic and focused (best for factual answers)
# Higher values = more creative but less consistent
TEMPERATURE = 0.0

# ============================================
# Document Configuration
# ============================================

# PDF File Path
# This is the path to the Constitution PDF file
# The path is relative to where the script is run from
# Since the script runs from the Ai folder, we use just the filename
PDF_PATH = "constition_new.pdf"

# Vector Database Path
# This is where we'll save the FAISS vector store after processing the PDF
# FAISS creates an index file that allows fast similarity search
VECTOR_DB_PATH = "vector_store"

# ============================================
# Text Processing Configuration
# ============================================

# Chunk Size
# When we split the PDF into smaller pieces, this is the maximum size of each chunk
# Measured in characters. Increased to 2500 to keep Articles and their footnotes together
# This helps prevent false negatives where Article headers get separated from content
# Larger chunks ensure amendments and footnotes stay with their related articles
CHUNK_SIZE = 2500

# Chunk Overlap
# When splitting text, we overlap chunks by this many characters
# Increased to 400 to ensure we don't lose context at chunk boundaries
# This is critical for legal documents where amendments might appear near boundaries
# Example: If chunk 1 ends at word "Constitution" and chunk 2 starts at "Article",
#          overlap ensures "Constitution Article" appears together
CHUNK_OVERLAP = 400

# ============================================
# Validation
# ============================================

# Check if API key is set
# This will raise an error if the API key is missing, preventing runtime errors
if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
    raise ValueError(
        "Please set your OPENAI_API_KEY in the .env file. "
        "Get your API key from: https://platform.openai.com/api-keys"
    )

