import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.processing.ingestion import ProcessDocuments, TextSplitter
from src.vectordb.qdrantdb import VectorDB
from src.config import config

class Pipeline:
    def __init__(self):
        """Khởi tạo các thành phần cốt lõi của Pipeline"""
        self.processor = ProcessDocuments()
        self.splitter = TextSplitter(
            chunk_size=config.CHUNK_SIZE, 
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        self.vdb = None

    def run_full_pipeline(self):
        """Quy trình chạy toàn bộ từ PDF -> PKL -> VectorDB"""
        print("--- [START] INGESTION PIPELINE ---")
        
        # 1. Trích xuất văn bản (Tạo file .pkl)
        docs = self._extract_step()
        
        # 2. Chia nhỏ văn bản
        chunks = self._split_step(docs)
        
        # 3. Lưu trữ vào Vector Database
        self._load_to_vector_db(chunks)
        print("--- [FINISHED] PIPELINE HOÀN TẤT ---")

    def _extract_step(self):
        print("Trích xuất và sửa lỗi font PDF...")
        en_docs = self.processor.load_english_textbook()
        vn_docs = self.processor.load_vietnamese_table()
        all_docs = en_docs + vn_docs
        print(f"Đã xử lý xong {len(all_docs)} trang tài liệu.")
        return all_docs

    def _split_step(self, documents):
        print("Đang thực hiện Chunking...")
        chunks = self.splitter.split(documents)
        print(f"Tạo ra {len(chunks)} chunks nhỏ.")
        return chunks
    
    def _load_to_vector_db(self, chunks):
        print(f"Đang đẩy dữ liệu vào Vector DB (Qdrant)...")
        # Khởi tạo VectorDB với chunks sẽ kích hoạt force_recreate=True
        self.vdb = VectorDB(documents=chunks)
        print(f"Dữ liệu đã được lưu trữ an toàn tại Local.")


#===== RUN Pipeline to test and create file pkl ====#

if __name__ == "__main__":
    # Khởi tạo đối tượng Pipeline
    pipeline = Pipeline()
    
    # Kích hoạt quy trình
    pipeline.run_full_pipeline()


