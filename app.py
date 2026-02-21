import os
import time
import json
import tempfile
from dotenv import load_dotenv
import streamlit as st
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. System Config
load_dotenv()
st.set_page_config(page_title="Neural Researcher", page_icon="📜", layout="wide")

# --- STABLE PREMIUM CSS (ORIGINAL & LOCKED) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,700&family=Inter:wght@400;500;600&display=swap');
    
    header[data-testid="stHeader"] { background: #1a1412 !important; }
    footer {visibility: hidden;}
    .stApp { background: #1a1412 !important; color: #e5dada !important; }
    
    [data-testid="stSidebar"] { background-color: #0d0b0a !important; border-right: 1px solid #2d2421; min-width: 320px !important; }
    .neural-archive-title { color: #d4a373; font-family: 'Playfair Display', serif; font-size: 2.2rem !important; font-style: italic; margin-bottom: 2rem; letter-spacing: -1px; }
    .sidebar-label { color: #ffffff !important; text-transform: uppercase; letter-spacing: 3px; font-size: 0.85rem !important; font-weight: 800; margin-top: 25px !important; margin-bottom: 10px !important; }
    
    div[data-testid="stRadio"] label { background-color: #1a1412 !important; border: 1px solid #2d2421 !important; border-radius: 12px !important; padding: 12px 18px !important; margin-bottom: 10px !important; transition: all 0.3s ease !important; }
    div[data-testid="stRadio"] label p { color: #ffffff !important; font-weight: 500 !important; }
    div[data-testid="stRadio"] label:has(input:checked) { border-color: #d4a373 !important; background-color: #261e1b !important; box-shadow: 0 0 15px rgba(212, 163, 115, 0.4) !important; transform: scale(1.02); }
    
    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px dashed #4a3b37 !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: #ffffff !important; }
    [data-testid="stFileUploaderDropzone"] button { background-color: #d4a373 !important; color: #1a1412 !important; font-weight: 700 !important; border: none !important; }

    [data-testid="stBottomBlockContainer"] { background-color: #F5F2ED !important; padding-top: 15px !important; }
    [data-testid="stChatInput"] textarea { background-color: #F5F2ED !important; color: #000000 !important; caret-color: #000000 !important; border: 1px solid #1a1412 !important; }

    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) { background-color: #ffffff !important; border-radius: 15px !important; }
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) p { color: #000000 !important; }
    
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) { background-color: #261e1b !important; border: 1px solid #3d2f2b !important; border-radius: 15px !important; }
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) * { color: #ffffff !important; }
    code { color: #d4a373 !important; background-color: rgba(212, 163, 115, 0.1) !important; padding: 2px 4px !important; border-radius: 4px !important; }
    pre code { color: #e5dada !important; background-color: #0d0b0a !important; display: block !important; padding: 1rem !important; }

    .main-title { font-family: 'Playfair Display', serif; font-size: 4rem; font-style: italic; color: #d4a373; text-align: center; margin-top: 1rem; }
    .sub-title { font-family: 'Inter', sans-serif; text-transform: uppercase; letter-spacing: 4px; font-size: 0.8rem; color: #8c7b75; text-align: center; margin-bottom: 2rem; }
    .cost-text { color: #d4a373; font-size: 0.95rem; font-weight: 600; margin-bottom: 10px; }
    .stButton > button { width: 100% !important; background-color: rgba(255, 255, 255, 0.05) !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- BACKEND UTILS ---
DB_FILE = "neural_vault.json"

def load_vault():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f: return json.load(f)
        except: return {}
    return {}

def save_vault(vault_data):
    with open(DB_FILE, "w") as f: json.dump(vault_data, f)

def generate_report(history):
    report = "=== NEURAL RESEARCH REPORT ===\n\n"
    for m in history:
        role = "USER" if m["r"] == "u" else "NEURAL RESEARCHER"
        report += f"[{role}]:\n{m['c']}\n"
        if "meta" in m: report += f"({m['meta']})\n"
        report += "-"*30 + "\n"
    return report

def estimate_tokens(text):
    return len(text) // 4

@st.cache_resource
def load_engine():
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile", temperature=0.1, streaming=True)
    embeds = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    t_key = os.getenv("TAVILY_API_KEY")
    web_tool = TavilySearchResults(tavily_api_key=t_key, k=5) if t_key else None
    return llm, embeds, web_tool

model, embeddings, web_search = load_engine()

# --- INITIALIZATION ---
vault = load_vault()
if "current_session" not in st.session_state:
    st.session_state.current_session = str(int(time.time()))
    st.session_state.chat_history = []; st.session_state.total_cost = 0.0

curr_id = st.session_state.current_session
if not st.session_state.chat_history and curr_id in vault:
    st.session_state.chat_history = vault[curr_id].get("history", [])
    st.session_state.total_cost = vault[curr_id].get("total_cost", 0.0)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="neural-archive-title">Neural Archive</div>', unsafe_allow_html=True)
    if st.button("＋ New Research"):
        st.session_state.current_session = str(int(time.time()))
        st.session_state.chat_history = []; st.session_state.total_cost = 0.0; st.rerun()

    st.markdown(f'<div class="cost-text">💰 Session Cost: ${st.session_state.total_cost:.6f}</div>', unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        st.download_button(label="📥 Download Research Report", data=generate_report(st.session_state.chat_history), file_name=f"report_{curr_id}.txt", mime="text/plain")

    if st.button("🗑️ Purge Current Session"):
        if curr_id in vault: del vault[curr_id]; save_vault(vault)
        st.session_state.current_session = str(int(time.time()))
        st.session_state.chat_history = []; st.session_state.total_cost = 0.0; st.rerun()

    st.markdown('<p class="sidebar-label">Research Intensity</p>', unsafe_allow_html=True)
    mode = st.radio("Mode", ["Quick Scan", "Deep Research"], label_visibility="collapsed")
    st.markdown('<p class="sidebar-label">Upload Intelligence</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload", type=["pdf", "txt", "py", "js", "cpp", "java", "md"], label_visibility="collapsed")
    
    st.markdown('<p class="sidebar-label">Recent Archives</p>', unsafe_allow_html=True)
    for sid in sorted(vault.keys(), reverse=True):
        h = vault[sid].get("history", [])
        label = h[0]["c"][:25] + "..." if h else f"Archive {sid}"
        if st.button(f"📜 {label}", key=f"v_{sid}"):
            st.session_state.current_session = sid; st.session_state.chat_history = vault[sid]["history"]
            st.session_state.total_cost = vault[sid].get("total_cost", 0.0); st.rerun()

# --- FILE PROCESSING (WITH SOURCE FIX) ---
if uploaded_file and ("active_file" not in st.session_state or st.session_state.active_file != uploaded_file.name):
    with st.spinner("Decoding..."):
        file_ext, original_name = uploaded_file.name.split('.')[-1].lower(), uploaded_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tf:
            tf.write(uploaded_file.getbuffer()); t_path = tf.name
        
        loader = PyPDFLoader(t_path) if file_ext == 'pdf' else TextLoader(t_path)
        docs = loader.load()
        for d in docs: d.metadata["source"] = original_name # Force original name
        
        code_map = {"py": Language.PYTHON, "js": Language.JS, "cpp": Language.CPP, "java": Language.JAVA}
        splitter = RecursiveCharacterTextSplitter.from_language(language=code_map[file_ext], chunk_size=1000, chunk_overlap=150) if file_ext in code_map else RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        
        QdrantVectorStore.from_documents(splitter.split_documents(docs), embeddings, url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), collection_name=f"sess_{curr_id}")
        st.session_state.active_file = original_name; st.toast(f"Synced: {original_name}")

# --- MAIN INTERFACE ---
st.markdown('<div class="main-title">Neural Researcher</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Technical Intelligence & Coding Agent</div>', unsafe_allow_html=True)

for m in st.session_state.chat_history:
    with st.chat_message("user" if m["r"] == "u" else "assistant"):
        st.markdown(m["c"])
        if m["r"] == "a" and "meta" in m: st.caption(m["meta"])

query = st.chat_input("Inquire the neural network...")

if query:
    st.session_state.chat_history.append({"r": "u", "c": query}); st.rerun()

if st.session_state.chat_history and st.session_state.chat_history[-1]["r"] == "u":
    u_q = st.session_state.chat_history[-1]["c"]
    with st.chat_message("assistant"):
        placeholder, full_res, start_t = st.empty(), "", time.time()
        try:
            v_store = QdrantVectorStore.from_existing_collection(embedding=embeddings, url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), collection_name=f"sess_{curr_id}")
            hits = v_store.similarity_search(u_q, k=5)
            # Metadata-based citation
            sources = list(set([h.metadata.get("source", "Unknown") for h in hits]))
            internal = "\n".join([h.page_content for h in hits])
            external = str(web_search.run(u_q)) if mode == "Deep Research" else ""
            
            # Clarifying Question Logic
            is_ambig = len(u_q.split()) < 4 or any(w in u_q.lower() for w in ["it", "this", "fix"])
            instr = "Ask for missing details if vague. Else be direct." if is_ambig else "Technical response."

            for chunk in model.stream(f"{instr}\nContext: {internal}\n{external}\nQuestion: {u_q}"):
                full_res += chunk.content; placeholder.markdown(full_res + " 🖋️")
            
            citation = f"\n\n**Sources Analyzed:** {', '.join(sources)}" if sources else ""
            full_res += citation
            placeholder.markdown(full_res)
            
            lat, toks = round(time.time() - start_t, 2), estimate_tokens(u_q) + estimate_tokens(full_res)
            q_cost = (toks * (0.69/1e6))
            st.session_state.total_cost += q_cost
            meta_str = f"⚡ Latency: {lat}s | 🪙 Tokens: {toks} | Mode: {mode}"
            st.caption(meta_str)
            st.session_state.chat_history.append({"r": "a", "c": full_res, "meta": meta_str})
            vault[curr_id] = {"history": st.session_state.chat_history, "total_cost": st.session_state.total_cost}
            save_vault(vault); st.rerun()
        except Exception as e: st.error(f"Fault: {e}")