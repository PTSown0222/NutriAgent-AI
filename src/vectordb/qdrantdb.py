import os
import sys
from typing import Optional, List, Dict
#from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.config import config


class VectorDB:
    def __init__(self, documents: Optional[List] = None):
        
        # 1. Khởi tạo thuộc tính cơ bản
        self.collection_name = config.COLLECTION_NAME
        self.db_type = config.VECTOR_DB_TYPE
        self.db_path = config.QDRANT_LOCAL_PATH
        
        # Setup Embedding
        print(f"Loading Embedding Model: {config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

        # 2. Thiết lập kết nối dựa trên loại DB
        self.client_config = self._get_client_config()
        
        # 3. Khởi tạo Vector Store
        if documents and len(documents) > 0:
            self.db = self._create_new_db(documents)
        else:
            self.db = self._load_existing_db()
    
    def _get_client_config(self) -> dict:
        
        if self.db_type == "qdrant_cloud":
            if not config.QDRANT_ENDPOINT or not config.QDRANT_API_KEY:
                raise ValueError("Thiếu cấu hình Qdrant Cloud trong .env")
            print(f"Connecting to Qdrant Cloud: {config.QDRANT_ENDPOINT[:12]}...")
            return {
                "url": config.QDRANT_ENDPOINT,
                "api_key": config.QDRANT_API_KEY
            }
        # Ngược lại dùng local
        print(f"Using Local Path: {config.QDRANT_LOCAL_PATH}")
        return {"path": config.QDRANT_LOCAL_PATH}

    def _create_new_db(self, documents: List) -> QdrantVectorStore:
        
        print(f"Đang đẩy {len(documents)} documents vào {self.db_type}...")
        
        if self.db_type == "qdrant_local":
            os.makedirs(self.client_config["path"], exist_ok=True)

        return QdrantVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            force_recreate=True,
            **self.client_config # Unpack tham số (url/api_key hoặc path)
        )
    
    def _load_existing_db(self) -> QdrantVectorStore:
        """Kết nối tới DB đã tồn tại (Retrieval)."""
        # Kiểm tra folder nếu chạy local
        if self.db_type == "qdrant_local" and not os.path.exists(self.client_config["path"]):
            raise FileNotFoundError(f"Chưa thấy DB tại {self.client_config['path']}")

        print(f"Kết nối tới Collection: {self.collection_name}")
        # Khởi tạo client 
        client = QdrantClient(**self.client_config)
        
        return QdrantVectorStore(
            client=client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
    
    def get_retriever(self, search_kwargs: Dict = None): 
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        return self.db.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
            )

