# 📜 Neural Researcher
### **Advanced Technical Intelligence & Coding Agent**

**Neural Researcher** is a high-performance RAG (Retrieval-Augmented Generation) agent designed specifically for technical research, code analysis, and document intelligence. By leveraging **Llama-3.3** and **Qdrant Vector DB**, the system seamlessly combines your private data with live web results to provide cited, accurate insights.

---

## 🚀 Key Features

* **🧠 Hybrid Knowledge Base:** Seamless integration of local files (PDF, Code, Text) and live internet search via **Tavily AI**.
* **📂 Multi-Format Support:** Ingest and analyze Python, C++, Java, JS, PDF, and Markdown files with language-specific splitting.
* **🔗 Automatic Source Citation:** The agent identifies exactly which file (e.g., `work.py`) was used to generate each part of the response.
* **💰 Live Cost Tracking:** Real-time token usage monitoring and session cost estimation based on Llama-3.3 pricing models.
* **📜 Neural Archive:** Auto-saves every session, allowing you to reload or purge research history at any time.
* **📥 Professional Reports:** Export your entire research session into a structured text report with a single click.
* **🎨 Premium UI:** A sleek, dark-themed Bronze-Beige interface optimized for professional technical workflows.

---

## 🛠️ Tech Stack

* **LLM:** Llama-3.3-70B-Versatile (via Groq)
* **Vector Database:** Qdrant Cloud
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Orchestration:** LangChain & Streamlit
* **Search Engine:** Tavily AI

---

## ⚙️ Workflow Architecture



The system follows a sophisticated pipeline to ensure data accuracy:
1.  **Data Ingestion:** User files are processed and broken into chunks using `RecursiveCharacterTextSplitter` to preserve semantic context.
2.  **Vector Encoding:** Chunks are converted into high-dimensional vectors using HuggingFace embeddings and stored in **Qdrant Cloud**.
3.  **Contextual Retrieval:** Based on the user query, the system performs a similarity search to fetch the Top-K relevant data points.
4.  **Agentic Reasoning:** Llama-3.3 synthesizes the retrieved context and (if "Deep Research" is enabled) live web data to produce a technical response with source attribution.

---

## 🧪 Common Use Cases

* **Bug Bounty & Debugging:** Upload `.py` or `.cpp` files to identify potential vulnerabilities or optimize logic.
* **Literature Review:** Extract specific data points from multiple research papers (PDFs) while tracking citations.
* **Live Market Analysis:** Use "Deep Research" mode to find the latest technical trends and documentation updates in real-time.

---

## ⚠️ Constraints & Disclaimers

* **Context Window:** The system manages up to 128k tokens per session (Llama-3.3 hardware limit).
* **Cost Efficiency:** Each query's cost is tracked live to prevent unexpected API overhead.
* **Data Privacy:** Uploaded files are cleared from the active vault upon purging the session or manual deletion.



---

## 🔧 Installation & Setup

1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/Yogyaa20/research-agent.git](https://github.com/Yogyaa20/research-agent.git)
    cd neural-researcher
    ```

2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_key
    TAVILY_API_KEY=your_key
    QDRANT_URL=your_url
    QDRANT_API_KEY=your_key
    ```

4.  **Launch the App:**
    ```bash
    streamlit run app.py
    ```

---

**Developed by HACKERHUB | Maharaja Surajmal Institute of Technology**
