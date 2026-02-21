import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

def start_ingestion():
    print("🚀 SCRIPT START: Reading technical documents...")
    
    if not os.path.exists("./data"):
        os.makedirs("./data")
        print("❌ 'data' folder khali tha, naya banaya hai. PDF dalo!")
        return

    # 1. Load Documents
    loader = DirectoryLoader('./data', glob="./*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    if not docs:
        print("❌ Error: 'data' folder mein PDF nahi mili!")
        return

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Embedding model (Local & Free)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"📦 Splitting into {len(splits)} chunks. Uploading to Qdrant...")

    # 4. Upload to Cloud
    QdrantVectorStore.from_documents(
        splits, 
        embedding=embeddings, 
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="tech_research_agent",
        force_recreate=True
    )
    print("✅ SUCCESS! Knowledge base live ho gaya.")

if __name__ == "__main__":
    start_ingestion()