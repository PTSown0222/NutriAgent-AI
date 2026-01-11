import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.cores.CoT_agent import NutriAgentReseacher

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="NutriAgent Chat",
    page_icon="🥗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. CSS TÙY CHỈNH (GIAO DIỆN SẠCH SẼ) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Ẩn bớt các element thừa */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Input Chat */
    .stChatInputContainer textarea {
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }

    /* Tiêu đề */
    .header-container {
        text-align: center;
        margin-bottom: 30px;
    }
    .header-title {
        font-size: 36px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #059669, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. HÀM HELPER CHUYỂN ĐỔI LỊCH SỬ ---
def convert_history_to_langchain(streamlit_msgs):
    """
    Chuyển đổi format chat của Streamlit sang format LangChain hiểu được.
    """
    history = []
    for msg in streamlit_msgs:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history

# --- 4. KHỞI TẠO AGENT (QUẢN LÝ CACHE THÔNG MINH) ---
@st.cache_resource(show_spinner=False)
def get_agent(use_reasoning_mode):
    """
    Khởi tạo Agent. 
    Tham số 'use_reasoning_mode' giúp Streamlit biết khi nào cần tạo lại Agent mới.
    """
    print(f"🔄 Đang khởi tạo NutriAgent với chế độ Reasoning={use_reasoning_mode}...")
    return NutriAgentReseacher(use_reasoning=use_reasoning_mode)

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình NutriAgent")
    
    # Nút gạt bật tắt chế độ suy luận
    is_reasoning = st.toggle("Kích hoạt Suy luận sâu (CoT)", value=True)
    
    st.divider()
    
    # Nút xóa lịch sử
    if st.button("🗑️ Clear Chat", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("""
    <div style='margin-top: 20px; font-size: 12px; color: grey;'>
    Supported by: <br>
    - Llama 3.3 (Groq)<br>
    - Qdrant VectorDB<br>
    - Advanced RAG
    - PDF Nutrition & Food Viet Nam Table VN
    - Phuong The Son <br>
    </div>
    """, unsafe_allow_html=True)

# Khởi tạo Agent dựa trên nút gạt
agent = get_agent(is_reasoning)

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 6. GIAO DIỆN CHÍNH ---
st.markdown("""
    <div class="header-container">
        <div style="font-size: 50px;">🥗</div>
        <h1 class="header-title">NutriAgent AI</h1>
        <p>Hỏi đáp dinh dưỡng & Tra cứu thành phần thực phẩm</p>
    </div>
""", unsafe_allow_html=True)

# Render lịch sử
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🦜"
    with st.chat_message(msg["role"], avatar=avatar):
        # Nếu là bot và có phần suy luận (ẩn trong metadata) thì hiển thị lại
        if msg["role"] == "assistant" and "thoughts" in msg and msg["thoughts"]:
            with st.expander("🤔 Xem quá trình suy luận"):
                st.info(msg["thoughts"])
        st.markdown(msg["content"])

# --- 7. XỬ LÝ CHAT ---
if prompt := st.chat_input("Ví dụ: 100g ức gà chứa bao nhiêu protein?"):
    
    # 1. Hiển thị User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    # 2. Xử lý Assistant Message
    with st.chat_message("assistant", avatar="🦜"):
        
        # Chuyển đổi lịch sử chat (Bỏ tin nhắn user vừa nhập để tránh trùng lặp trong history)
        lc_history = convert_history_to_langchain(st.session_state.messages[:-1])
        
        # Container hiển thị trạng thái
        with st.status("NutriAgent đang nghiên cứu...", expanded=True) as status:
            st.write("🔍 Đang tìm kiếm trong VectorDB...")
            # Gọi Agent
            response = agent.research(prompt, chat_history=lc_history)
            
            st.write("Đang tổng hợp câu trả lời...")
            status.update(label="Đã xong!", state="complete", expanded=False)
        
        # --- A. HIỂN THỊ SUY LUẬN (Nếu có) ---
        thoughts = response.get("model_thoughts", "")
        if is_reasoning and thoughts:
            with st.expander("🤔 Xem quá trình suy luận (Chain-of-Thought)"):
                st.info(thoughts)
        
        # --- B. HIỂN THỊ CÂU TRẢ LỜI ---
        st.markdown(response["answer"])
        
        # --- C. HIỂN THỊ NGUỒN ---
        sources = response.get("sources", [])
        if sources:
            st.divider()
            st.caption("📚 Nguồn tài liệu tham khảo:")
            
            # Xử lý hiển thị nguồn đẹp mắt
            unique_sources = {}
            for doc in sources:
                src_name = doc.metadata.get('source', 'Tài liệu không tên')
                # Làm sạch tên file (bỏ đường dẫn dài dòng)
                short_name = src_name.split("/")[-1].replace(".pdf", "")
                unique_sources[short_name] = unique_sources.get(short_name, 0) + 1
            
            # Hiển thị dạng Chips
            cols = st.columns(len(unique_sources))
            for idx, (name, count) in enumerate(unique_sources.items()):
                # Dùng HTML/CSS nhỏ để hiển thị badge
                st.markdown(f"""
                <div style="background-color: #f0fdf4; padding: 5px 10px; border-radius: 20px; border: 1px solid #bbf7d0; font-size: 12px; color: #166534; display: inline-block;">
                    📄 {name} <span style="font-weight: bold;">(x{count} chunks)</span>
                </div>
                """, unsafe_allow_html=True)

    # 3. Lưu lại vào Session State (Kèm cả phần suy luận để render lại nếu f5)
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response["answer"],
        "thoughts": thoughts # Lưu thêm trường này
    })