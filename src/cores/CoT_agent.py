import logging
import re
from typing import List, Dict, Any
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
# Chuyển hướng sang thư viện classic
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever
# Import Retrievers & Rerankers
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# Internal class
from src.config import config
from src.vectordb.qdrantdb import VectorDB
#from src.cores.prompts import system_prompt

class NutriAgentReseacher:
    def __init__(self, use_reasoning : bool = True):
        self.use_reasoning = use_reasoning 
       
        # khởi tạo LLMs
        self.llm  = ChatGroq(
            model = config.GROQ_MODEL_NAME,
            api_key = config.GROQ_API_KEY,
            temperature=0,
            max_tokens=2048
        )

        # 2. Kết nối Ký ức (VectorDB)
        # Gọi class VectorDB không tham số -> Tự động load DB từ ổ cứng (Local)
        self.vectordb = VectorDB()
        self.retriever = self.vectordb.get_retriever(search_kwargs={"k": 5})

        # RAG nâng cao
        self.advanced_retriever = self._build_advanced_retriever()

        # Tạo Chain xử lý cuối cùng
        self.chain = self._create_conversational_chain()
    
    def _build_advanced_retriever(self):
        """
        techniques: Dense Search + Multi-Query + Reranking
        """
        print("... Đang kích hoạt các module Advanced RAG ...")
        
        # A. Base Retriever: Lấy dữ liệu thô
        # Lấy số lượng lớn (k=20) để không bị sót thông tin, sau đó lọc sau.
        base_retriever = self.vectordb.get_retriever(search_kwargs={"k": 20})

        # B. Kỹ thuật 1: Multi-Query (Đa truy vấn)
        # Agent tự biến đổi câu hỏi của user thành 3-4 câu khác nhau để tìm kiếm rộng hơn.
        
        # Ví dụ: User hỏi "Ức gà tốt ko?" -> AI tìm thêm "Dinh dưỡng ức gà", "Lợi ích ức gà".
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
            include_original=True
        )

        # C. Kỹ thuật 2: Reranker (Sắp xếp lại)
        # Dùng model chuyên dụng để đọc kỹ 20 kết quả trên, chấm điểm và chỉ lấy 5 cái tốt nhất.
        # Model 'BAAI/bge-reranker-base' rất nhẹ và tốt cho đa ngôn ngữ.
        reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=reranker_model, top_n=5)

        #  Multi-Query chạy trước -> Kết quả đưa vào Reranker
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=multi_query_retriever
        )
        return compression_retriever
    
    def _create_conversational_chain(self):
        # --- 1. REPHRASE PROMPT (Tạo câu hỏi độc lập) ---
        contextualize_q_system_prompt = """
        Given a chat history and the latest user question which might reference context in the chat history, 
        formulate a standalone question which can be understood without the chat history.
        
        ### INSTRUCTIONS:
        1. **RESOLVE COREFERENCES:** Replace "it", "they", "that", "nó", "người đó" with specific nouns from history.
        2. **KEEP LANGUAGE:** If User speaks Vietnamese, Output Vietnamese. If English, Output English.
        3. **NO ANSWERING:** Do NOT answer. Just rewrite.
        4. **FORMAT:** Output ONLY the standalone question.
        """
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, 
            self.advanced_retriever, 
            contextualize_q_prompt
        )

        # --- 2. QA PROMPT (Prompt trả lời chính) ---
        # Sử dụng kỹ thuật Chain-of-Thought với XML Tags (<thinking>, <answer>)
        
        if self.use_reasoning:
            instructions = """
            You are a Critical Thinking Nutritionist. Follow this structure strictly:

            1.  **Analyze (<thinking>)**:
                - Identify the User's core Intent.
                - Scan the provided `Context` for keywords.
                - Discard irrelevant info (Context Precision).
                - Check for conflicting information.
            
            2.  **Formulate (<answer>)**:
                - Answer the question DIRECTLY based *only* on the context.
                - If the context is empty or insufficient, state: "Xin lỗi, tài liệu hiện tại không có thông tin về vấn đề này."
                - Do not use outside knowledge.
                - **CITATION REQUIREMENT:** You MUST cite sources using the format ``. Append the citation directly after the relevant sentence.
            
            FORMAT YOUR OUTPUT AS:
            <thinking>
            [Your step-by-step reasoning here]
            </thinking>
            <answer>
            [Your final response to the user here with citations]
            </answer>
            """
        else:
            instructions = """
            You are a helpful Nutrition Assistant.
            - Answer DIRECTLY based on the Context.
            - Do not include internal thoughts.
            - If unknown, say "Xin lỗi, không có thông tin".
            - Wrap your final response in <answer> tags for consistency.
            """
        
        system_prompt = f"""
        You are NutriAgent, an expert assistant. 
        
        ### INSTRUCTIONS:
        {instructions}

        ### CONTEXT:
        {{context}}
        """

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return rag_chain
    
    def research(self, query: str, chat_history: List[BaseMessage] = []) -> Dict[str, Any]:
        """
        Hàm thực thi chính.
        Input: Câu hỏi + Lịch sử chat
        Output: Dictionary chứa câu trả lời, nguồn, và suy luận (nếu có)
        """
        try:
            # Gọi Chain đã khởi tạo
            response = self.chain.invoke({"input": query, "chat_history": chat_history})
            
            raw_text = response["answer"]
            sources = response.get("context", [])
            
            final_answer = raw_text
            model_thinking = ""

            # --- Logic Parse XML Tags (<thinking> ... </thinking>) ---
            if self.use_reasoning:
                # 1. Trích xuất phần suy nghĩ
                think_match = re.search(r'<thinking>(.*?)</thinking>', raw_text, re.DOTALL)
                if think_match:
                    model_thinking = think_match.group(1).strip()
                
                # 2. Trích xuất câu trả lời
                ans_match = re.search(r'<answer>(.*?)</answer>', raw_text, re.DOTALL)
                if ans_match:
                    final_answer = ans_match.group(1).strip()
                else:
                    # Fallback: Nếu model quên đóng tag answer, lấy phần còn lại sau thinking
                    final_answer = re.sub(r'<thinking>.*?</thinking>', '', raw_text, flags=re.DOTALL).strip()
                    final_answer = final_answer.replace("<answer>", "").replace("</answer>", "").strip()

            else:
                # Nếu không dùng reasoning, model có thể vẫn bọc trong <answer> do prompt yêu cầu consistency
                final_answer = raw_text.replace("<answer>", "").replace("</answer>", "").strip()
            
            return {
                "answer": final_answer,
                "sources": sources,
                "model_thoughts": model_thinking
            }
            
        except Exception as e:
            print(f"Error in research: {e}")
            return {
                "answer": "Xin lỗi, hệ thống đang gặp sự cố khi xử lý câu hỏi.",
                "sources": [],
                "model_thoughts": str(e)
            }
   




