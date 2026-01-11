import json
import re
import os
from pathlib import Path
from pypdf import PdfReader
import pandas as pd
from datasets import Dataset

# Ragas imports
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper

# Langchain & Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
# Utility
from rapidfuzz import fuzz

# Internal imports
from src.cores.CoT_agent import NutriAgentReseacher
from src.config import config

# =====================
# PATH SETUP
# =====================
BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data"

PDF_NUTRITION = DATA_DIR / "human-nutrition-text.pdf"
PDF_FOOD_TABLE = DATA_DIR / "VIETNAMESE FOOD COMPOSITION TABLE.pdf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =====================
# HELPERS
# =====================
def read_pdf_page_text(pdf_path: Path, page_number_1based: int) -> str:
    reader = PdfReader(str(pdf_path))
    if page_number_1based < 1 or page_number_1based > len(reader.pages):
        raise ValueError(f"Invalid page number {page_number_1based} for {pdf_path.name}")
    return (reader.pages[page_number_1based - 1].extract_text() or "").strip()

def pick_sentence_optimized(text: str, contains: str) -> str:
    # 1. Tìm chính xác
    if contains.lower() in text.lower():
        idx = text.lower().find(contains.lower())
        start = max(0, idx - 100)
        end = min(len(text), idx + 300)
        return " ".join(text[start:end].split())
    # 2. Tìm mờ (Fuzzy)
    sentences = re.split(r'(?<=[.!?]) +', text)
    best_match = None
    highest_score = 0
    for sent in sentences:
        score = fuzz.partial_ratio(contains.lower(), sent.lower())
        if score > highest_score:
            highest_score = score
            best_match = sent
    if highest_score > 70:
        return best_match
    return ""

def parse_energy_protein(page_text: str):
    kcal, protein = None, None
    m1 = re.search(r"Năng lượng.*?(?:KCal|kcal)\s*[:\.]?\s*([0-9]+(?:\.[0-9]+)?)", page_text, re.IGNORECASE)
    if m1: kcal = float(m1.group(1))
    m2 = re.search(r"Protein.*?(?:g)\s*[:\.]?\s*([0-9]+(?:\.[0-9]+)?)", page_text, re.IGNORECASE)
    if m2: protein = float(m2.group(1))
    return kcal, protein

def parse_stt_and_code(page_text: str):
    stt, code = None, None
    m_stt = re.search(r"\bSTT\b\s*[:\-]?\s*([0-9]{1,5})", page_text, re.IGNORECASE)
    if m_stt: stt = int(m_stt.group(1))
    m_code = re.search(r"(Mã\s*số|Ma\s*so)\s*[:\-]?\s*([0-9]{1,10})", page_text, re.IGNORECASE)
    if m_code: code = m_code.group(2)
    return stt, code

def rag_agent_answer(agent: NutriAgentReseacher, question: str):
    try:
        res = agent.research(question, chat_history=[])
        answer = res.get("answer", "Không có câu trả lời.")
        sources = res.get("sources", [])
        contexts = []
        for doc in sources:
            content = getattr(doc, "page_content", str(doc))
            contexts.append(content)
        return answer, contexts
    except Exception as e:
        print(f"Lỗi Agent: {e}")
        return "Lỗi hệ thống", []

def build_eval_cases():
    cases = []
    
    # 1. PDF Lý thuyết 
    p_nutrition = 49
    try:
        text_nut = read_pdf_page_text(PDF_NUTRITION, p_nutrition)
        gt_def = pick_sentence_optimized(text_nut, "Proteins are macromolecules")
        gt_sources = pick_sentence_optimized(text_nut, "Food sources of proteins include")
        gt_energy = pick_sentence_optimized(text_nut, "Proteins provide four kilocalories")
        
        if not gt_def: gt_def = "Proteins are macromolecules composed of amino acids."

        # Case Tiếng Việt
        cases.extend([
            {"q": "Protein là gì? (định nghĩa theo tài liệu)", "gt": gt_def, "file": PDF_NUTRITION.name},
            {"q": "Nguồn thực phẩm cung cấp protein gồm những gì?", "gt": gt_sources, "file": PDF_NUTRITION.name}
        ])
        # Case Tiếng Anh (Bổ sung)
        cases.extend([
            {"q": "What is the definition of proteins?", "gt": gt_def, "file": PDF_NUTRITION.name},
            {"q": "How much energy does protein provide per gram?", "gt": gt_energy, "file": PDF_NUTRITION.name}
        ])
    except Exception as e:
        print(f"Skip PDF Nutrition: {e}")

    # 2. PDF Bảng biểu (Số liệu)
    pages_food = [145, 146, 147]
    try:
        for p in pages_food:
            t = read_pdf_page_text(PDF_FOOD_TABLE, p)
            stt, code = parse_stt_and_code(t)
            kcal, protein = parse_energy_protein(t)
            
            info_str = f"Năng lượng {kcal} KCal, Protein {protein} g" if kcal else "Thông tin trong bảng."
            gt = f"Trang {p}: {info_str} (theo bảng TPTP VN)."
            
            cases.append({
                "q": f"Trong Bảng TP Thực phẩm VN, trang {p} có năng lượng và protein bao nhiêu?",
                "gt": gt,
                "file": PDF_FOOD_TABLE.name
            })
    except Exception as e:
        print(f"Skip PDF Table: {e}")

    return cases

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    try:
        print("--- 1. Generating Test Cases ---")
        raw_cases = build_eval_cases()
        
        print("--- 2. Running Agent ---")
        agent = NutriAgentReseacher()
        samples = []
        for c in raw_cases:
            print(f"Ask: {c['q']}")
            answer, contexts = rag_agent_answer(agent, c['q'])
            if c['gt']:
                samples.append({
                    "question": c['q'],
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": c['gt']
                })
        
        # Save dataset
        with open(DATA_DIR / "eval_nutrition_data.json", "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        print("--- 3. Running Ragas Evaluation ---")
        
        judge_model = ChatGroq(
            model= config.MODEL_JUDGE, 
            api_key=config.GROQ_API_KEY,
            temperature=0
        )
        judge_llm = LangchainLLMWrapper(judge_model)
        hf_embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
        
        # dataset and metrics
        ragas_ds = Dataset.from_list(samples)
        metrics_list = [faithfulness, context_precision, context_recall]
        
        my_run_config = RunConfig(
            max_workers=1,      
            timeout=120,    
            max_retries=5,  
            max_wait=60
        )
        
        result = evaluate(
            ragas_ds,
            metrics=metrics_list,
            llm=judge_llm,
            embeddings=ragas_embeddings,
            run_config=my_run_config 
        )
            
        print(result)
        # trả về pandas
        df = result.to_pandas()
        df.to_csv(BASE_DIR / "ragas_result.csv", index=False, encoding="utf-8-sig")
        print("Done!")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")