import os
import pickle
from llama_parse import LlamaParse
from typing import List
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Internal Import
from src.config import config 
from src.utils.helpers import fix_encoding, clean_broken_layout

class ProcessDocuments:
    def __init__(self):
        self.cache_dir = os.path.join(config.DATA_DIR, "cache_parse")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_from_cache(self, cache_name):
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.pkl")
        if os.path.exists(cache_path):
            print(f"Load cache_parse: {cache_name}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None
    
    def _save_to_cache(self, documents, cache_name):
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump(documents, f)
        print(f"Saved cache_parse: {cache_name}")
    
    # Hàm phụ trợ chuyển đổi LlamaDoc -> LangChainDoc
    def _convert_to_langchain(self, llama_docs, source_name):
        langchain_docs = []
        for doc in llama_docs:
            lc_doc = LangChainDocument(
                page_content=doc.text, # Lấy nội dung text
                metadata={
                    "source": source_name,
                    **doc.metadata # Copy metadata gốc
                }
            )
            langchain_docs.append(lc_doc)
        
        return langchain_docs

    def load_english_textbook(self):
        CACHE_NAME = "textbook_en_parsed"
        
        # Check cache
        cached = self._load_from_cache(CACHE_NAME)
        if cached: return cached

        print(f"Parsing TextBook EN...")
        parser = LlamaParse(
            api_key = config.LLAMA_CLOUD_API_KEY,
            result_type="markdown", 
            verbose=True, 
            language="en"
        )

        # load data sau khi parse              
        llama_docs = parser.load_data(config.PATH_TEXTBOOK_EN)
        
        # convert sang langchain
        final_docs = self._convert_to_langchain(llama_docs, "Human Nutrition Text")
        
        # Save cache
        self._save_to_cache(final_docs, CACHE_NAME)
        return final_docs

    def load_vietnamese_table(self):
        CACHE_NAME = "food_table_vn_parsed"
        
        # 1. Load file Cache cũ lên
        docs = self._load_from_cache(CACHE_NAME)
        
        # --- [LOGIC TỰ ĐỘNG FIX VÀ LƯU] ---
        if docs:
            print(f"📂 Đã tìm thấy Cache '{CACHE_NAME}'. Đang kiểm tra chất lượng dữ liệu...")
            count_fixed = 0
            is_modified = False 
            
            for doc in docs:
                original_text = doc.page_content
                
                # Gọi hàm sửa lỗi
                fixed_text = fix_encoding(original_text)
                cleaned_text = clean_broken_layout(fixed_text)
                
                # Nếu có thay đổi thì cập nhật
                if original_text != cleaned_text:
                    doc.page_content = cleaned_text
                    count_fixed += 1
                    is_modified = True 
            
            if count_fixed > 0:
                print(f"Đã tự động sửa lỗi hiển thị cho {count_fixed} trang tài liệu.")
            
            if is_modified:
                self._save_to_cache(docs, CACHE_NAME)
                print("Đã lưu Cache mới! Lần sau load sẽ không cần sửa nữa.")
            else:
                print("Dữ liệu trong Cache đã sạch đẹp. Không cần xử lý thêm.")

            return docs

        # --- TRƯỜNG HỢP KHÔNG CÓ CACHE (Parse mới từ đầu) ---
        print(f"Không thấy Cache. Bắt đầu Parse Food Table VN từ LlamaCloud...")
        parser = LlamaParse(
            api_key = config.LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            user_prompt="Đây là bảng dinh dưỡng. Chuyển thành Markdown Table chuẩn. Lặp lại header.",
            verbose=True,
            language="vi"
        )

        llama_docs = parser.load_data(config.PATH_FOOD_TABLE_VN)
        
        # Convert sang LangChain
        final_docs = self._convert_to_langchain(llama_docs, "Vietnamese Food Table")
        for doc in final_docs:
            doc.page_content = fix_encoding(doc.page_content)
            doc.page_content = clean_broken_layout(doc.page_content)

        # Lưu cache lại
        self._save_to_cache(final_docs, CACHE_NAME)
        
        return final_docs


class TextSplitter:
    def __init__(self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "|", "##", " ", ""] 
        )

    def split(self, documents : List[str]):
        chunks = self.splitter.split_documents(documents)
        return chunks