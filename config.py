# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Groq API
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")
    
    # Feature flags (so you can switch back to ChromaDB if needed)
    USE_PINECONE = os.getenv("USE_PINECONE", "true").lower() == "true"
    
    @classmethod
    def validate(cls):
        """Validate required environment variables"""
        required = ["GROQ_API_KEY"]
        
        if cls.USE_PINECONE:
            required.extend(["PINECONE_API_KEY"])
        
        missing = [key for key in required if not getattr(cls, key)]
        
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

# Validate on import (will fail fast if misconfigured)
Config.validate()