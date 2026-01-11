import os
from dotenv import load_dotenv
from pathlib import Path
# File này ở: /AI_Agent_NutriRAG/src/config.py
# BASE_DIR sẽ là: /AI_Agent_NutriRAG/
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class Config:
    # --- API KEYS ---
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_API_KEY = os.getenv("HF_API_KEY")


    # --- DATA PATHS ---
    DATA_DIR = BASE_DIR / "data"

    QDRANT_LOCAL_PATH = str(DATA_DIR / "qdrant_db")
    CACHE_DIR = str(DATA_DIR / "cache")
    
    PATH_TEXTBOOK_EN = str(DATA_DIR / "human-nutrition-text.pdf")
    PATH_FOOD_TABLE_VN = str(DATA_DIR / "VIETNAMESE FOOD COMPOSITION TABLE.pdf")

    # Model
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    #PROD_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
    RERANKER_MODEL = "BAAI/bge-reranker-base"
    CHUNK_SIZE = 450
    CHUNK_OVERLAP = 150

    #Model
    GROQ_MODEL_NAME = "llama-3.1-8b-instant"
    MODEL_JUDGE = "llama-3.3-70b-versatile"

    # VectorDatabase
    #VECTOR_DB_TYPE = "qdrant_local"
    VECTOR_DB_TYPE = "qdrant_cloud"
    #VECTOR_DB_TYPE = "pinecone"

    # Cấu hình phụ cho Pinecone (Chỉ dùng nếu type="pinecone")
    PINECONE_INDEX_NAME = "nutri-agent"
    COLLECTION_NAME = "nutrition_agent"

config = Config()

