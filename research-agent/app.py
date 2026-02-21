import os
import time
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

# --- STABLE PREMIUM CSS (FONT COLOR UPDATED TO WHITE) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,700&family=Inter:wght@400;500;600&display=swap');
    
    header[data-testid="stHeader"] { background: #1a1412 !important; }
    footer {visibility: hidden;}
    .stApp { background: #1a1412 !important; color: #e5dada !important; }
    
    /* SIDEBAR LABELS & ELEMENTS */
    [data-testid="stSidebar"] { background-color: #0d0b0a !important; border-right: 1px solid #2d2421; }
    div[data-testid="stRadio"] label p { color: #ffffff !important; font-weight: 500 !important; }
    [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p { color: #ffffff !important; }
    [data-testid="stFileUploaderDropzone"] div > small { color: #ffffff !important; opacity: 1 !important; }
    .sidebar-label { color: #8c7b75 !important; text-transform: uppercase; letter-spacing: 2px; font-size: 0.75rem !important; font-weight: 700; margin-top: 25px !important; margin-bottom: 10px !important; }
    
    /* RADIO BOXES */
    div[data-testid="stRadio"] label { background-color: #1a1412 !important; border: 1px solid #2d2421 !important; border-radius: 10px !important; padding: 10px 15px !important; margin-bottom: 8px !important; }
    div[data-testid="stRadio"] label:has(input:checked) { border-color: #d4a373 !important; background-color: #261e1b !important; }

    /* CHAT FEED FONT COLOR FIX (PURE WHITE) */
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li, [data-testid="stChatMessage"] span { 
        color: #ffffff !important; 
        font-family: 'Inter', sans-serif !important;
    }
    
    /* FILE UPLOADER */
    [data-testid="stFileUploaderDropzone"] { background-color: rgba(255, 255, 255, 0.03) !important; border: 1px dashed #4a3b37 !important; border-radius: 12px !important; }
    [data-testid="stFileUploaderDropzone"] button { background-color: #d4a373 !important; color: #1a1412 !important; font-weight: 700 !important; }

    /* BEIGE SEARCHBAR */
    [data-testid="stBottomBlockContainer"] { background-color: #F5F2ED !important; padding-top: 15px !important; }
    [data-testid="stChatInput"] textarea { background-color: #F5F2ED !important; color: #000000 !important; caret-color: #000000 !important; border: 1px solid #1a1412 !important; }

    /* HEADERS */
    .main-title { font-family: 'Playfair Display', serif; font-size: 4rem; font-style: italic; color: #d4a373; text-align: center; margin-top: 1rem; }
    .sub-title { font-family: 'Inter', sans-serif; text-transform: uppercase; letter-spacing: 4px; font-size: 0.8rem; color: #8c7b75; text-align: center; margin-bottom: 2rem; }
    [data-testid="stChatMessage"] { background-color: #261e1b !important; border: 1px solid #3d2f2b !important; border-radius: 12px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE ---
@st.cache_resource
def load_engine():
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile", temperature=0.1, streaming=True)
    embeds = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Using 'tavily_api_key' parameter as per recent validation error fix
    web_tool = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"), k=5)
    return llm, embeds, web_tool

model, embeddings, web_search = load_engine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#d4a373; font-family:Playfair Display; font-style:italic;'>Neural Archive</h2>", unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">RESEARCH INTENSITY</p>', unsafe_allow_html=True)
    mode = st.radio("Intensity", ["Quick Scan", "Deep Research"], label_visibility="collapsed")
    st.markdown('<p class="sidebar-label">KNOWLEDGE BASE</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload", type=["pdf", "txt", "py", "js", "cpp", "java", "md"], label_visibility="collapsed")
    
    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if "active_file" not in st.session_state or st.session_state.active_file != uploaded_file.name:
            with st.spinner("Decoding Knowledge..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tf:
                    tf.write(uploaded_file.getbuffer())
                    temp_path = tf.name
                loader = PyPDFLoader(temp_path) if file_ext == 'pdf' else TextLoader(temp_path)
                st.session_state.vector_store = QdrantVectorStore.from_documents(
                    RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(loader.load()), 
                    embeddings, url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"),
                    collection_name="neural_final_pro"
                )
                st.session_state.active_file = uploaded_file.name
                st.toast("Internal Data Synced")

# --- MAIN ---
st.markdown('<div class="main-title">Neural Researcher</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Technical Intelligence & Coding Agent</div>', unsafe_allow_html=True)

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

query = st.chat_input("Inquire the neural network...")

if query:
    st.session_state.chat_history.append(HumanMessage(content=query))
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_res = ""
        start_t = time.time()
        
        try:
            internal_data = ""
            external_data = ""
            if "vector_store" in st.session_state:
                hits = st.session_state.vector_store.similarity_search(query, k=5)
                internal_data = "\n".join([f"[Internal]: {h.page_content}" for h in hits])

            if mode == "Deep Research":
                with st.status("🌐 Syncing Global Intelligence...", expanded=False) as status:
                    # Fixing the previous 'str' indexing fault by using .run()
                    external_data = str(web_search.run(query))
                    status.update(label="Web Insight Secured", state="complete")
                instruction = "Senior Systems Architect. Provide a Thinking Log then a detailed report. Use citations."
            else:
                instruction = "Quick Scan Assistant. Concise bullets only."

            final_prompt = f"{instruction}\n\nContext:\n{internal_data}\n{external_data}\n\nQuestion: {query}"

            for chunk in model.stream(final_prompt):
                full_res += chunk.content
                placeholder.markdown(full_res + " 🖋️")
            
            placeholder.markdown(full_res)
            st.caption(f"⚡ Latency: {round(time.time() - start_t, 2)}s | Mode: {mode}")
            st.session_state.chat_history.append(AIMessage(content=full_res))
            
        except Exception as e:
            st.error(f"Nexus Fault: {e}")