# src/prompts.py
# ==============================================================================
# GROUP 1: QUERY TRANSFORMATION (BIẾN ĐỔI CÂU HỎI)
# Kỹ thuật: Multi-Query, HyDE, Step-back, Rewrite
# ==============================================================================

# 1. Multi-Query Expansion
# Mục đích: Tạo ra nhiều biến thể của câu hỏi để tìm kiếm rộng hơn (tăng Recall).
MULTI_QUERY_PROMPT = """
You are an AI assistant specialized in {domain}.
Your task is to generate {num_versions} different versions of the given user question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search.

Provide these alternative questions separated by newlines.
Original question: {question}
"""

# 2. HyDE (Hypothetical Document Embeddings)
# Mục đích: Giả lập một câu trả lời hoàn hảo (dù có thể sai sự thật) để dùng nó đi tìm kiếm tài liệu thật.
# Lý do: Vector của "Câu trả lời giả" thường gần với "Tài liệu thật" hơn là "Câu hỏi".
HYDE_PROMPT = """
Please write a brief {domain} passage to answer the question below.
Do not worry about accuracy, focus on the vocabulary and structure a typical answer would have.

Question: {question}
Passage:
"""

# 3. Contextualize Question (History Aware)
# Mục đích: Viết lại câu hỏi dựa trên lịch sử chat (xử lý đại từ nhân xưng: nó, cái đó...).
CONTEXTUALIZE_Q_PROMPT = """
Given a chat history and the latest user question which might reference context in the chat history, 
formulate a standalone question which can be understood without the chat history.

RULES:
1. DO NOT answer the question.
2. Just reformulate/rewrite it if needed.
3. KEEP THE SAME LANGUAGE as the user's question.
"""

# ==============================================================================
# GROUP 2: ANSWER GENERATION (TỔNG HỢP CÂU TRẢ LỜI)
# Kỹ thuật: CoT (Chain of Thought), Few-shot, Universal Language
# ==============================================================================

# 4. Universal QA Prompt (Prompt Đa Năng)
# Prompt này xử lý logic chọn nguồn và ngôn ngữ.
UNIVERSAL_QA_PROMPT = """
You are an expert AI assistant in {domain}.
Answer the user's question based ONLY on the provided Context.

### REASONING & SOURCE LOGIC:
1. **Analyze:** specific keywords in the query to decide which part of the context is most relevant.
2. **Conflict Resolution:** If sources conflict, prioritize {priority_source}.
3. **Language:** ALWAYS answer in the same language as the user's question.

### FORMATTING RULES:
- Use Markdown for clarity (bold key numbers, bullet points).
- If the answer is not in the context, say "I don't have enough information".

### CONTEXT:
{context}

### USER QUESTION:
{input}
"""

# 5. One-Shot Prompt (Kèm ví dụ mẫu)
# Dùng khi bạn muốn AI bắt chước giọng điệu cụ thể.
ONE_SHOT_QA_PROMPT = """
You are a helpful assistant. Use the context to answer the question.

### EXAMPLE:
Q: {example_q}
A: {example_a}

### YOUR TURN:
Context: {context}
Question: {input}
"""

# ==============================================================================
# GROUP 3: ROUTING & PLANNING (AGENTIC)
# Kỹ thuật: Router (Phân loại câu hỏi để chọn Tool)
# ==============================================================================

# 6. Router Prompt
# Dùng để quyết định xem câu hỏi này nên tìm trong VectorDB hay tra Google Search hay tính toán.
ROUTER_PROMPT = """
You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to {domain}.

Use the following criteria:
- If the question is about {domain} specifics (ingredients, calories, laws, definitions), use 'vectorstore'.
- If the question is about current events, weather, or generic chit-chat, use 'web_search'.

Return a JSON with a single key 'datasource' and no premable or explaination.
Question to route: {question}
"""
