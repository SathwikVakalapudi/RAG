# 📄 PDF Query Search with RAG (Retrieval-Augmented Generation)

This project is a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to query a PDF document and receive intelligent, context-aware answers using LLMs.

It combines:

* 📚 Document retrieval (Chroma vector DB)
* 🔍 Query rewriting (Groq LLM)
* 🎯 Reranking (Cohere)
* 🤖 Answer generation (Groq LLM)
* 🌐 Interactive UI (Streamlit)

---

## 🚀 Features

* Extracts and processes PDF documents
* Uses **hierarchical chunking** (parent + child chunks)
* Generates **multiple query rewrites** for better retrieval
* Performs **semantic search** using embeddings
* Applies **Cohere reranking** for improved relevance
* Produces **final answers using LLM**
* Includes both:

  * 🖥 CLI pipeline (`main.py`)
  * 🌐 Web UI (`app.py`)

---

## 🏗️ Project Structure

```
RAG/
│
├── llm_utils/
│   ├── cohere_reranker.py     # Reranking with Cohere
│   └── groq_rewriter.py       # Query rewriting using Groq
│
├── app.py                     # Streamlit web app
├── main.py                    # CLI pipeline
├── config.py                  # Configuration settings
├── doc_utils.py               # Document conversion utilities
├── pdf_utils.py               # PDF text extraction
├── retriever_utils.py         # Chunking + retriever setup
├── vectorstore_utils.py       # Chroma vector store utilities
```

---

## ⚙️ How It Works

### 1. PDF Processing

* Extract text from PDF using `pdfplumber`
* Convert into LangChain `Document` objects

### 2. Chunking Strategy

* **Parent chunks** (large context)
* **Child chunks** (fine-grained retrieval)

### 3. Embeddings & Storage

* Uses HuggingFace model:

  ```
  sentence-transformers/all-MiniLM-L6-v2
  ```
* Stored in **Chroma vector database**

### 4. Query Pipeline

1. ✏️ Rewrite query into multiple variations (Groq)
2. 🔎 Retrieve relevant chunks (vector similarity)
3. 🧹 Deduplicate results
4. 🎯 Rerank using Cohere
5. 🤖 Generate final answer (Groq LLM)

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
```

---

## 📦 Installation

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

pip install -r requirements.txt
```

---

## ▶️ Usage

### 🖥 Run CLI version

```bash
python main.py
```

---

### 🌐 Run Streamlit App

```bash
streamlit run app.py
```

Then open the browser and enter your query.

---

## 🧪 Example Query

```
What is climate change?
```

Output:

* ✅ Rewritten queries
* ✅ Retrieved chunks
* ✅ Reranked results
* ✅ Final generated answer

---

## 🛠 Configuration

Edit `config.py`:

```python
PDF_PATH = "path_to_your_pdf"
PERSIST_DIR = "path_to_store_chroma"
COLLECTION_NAME = "child_chunks"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PARENT_CHUNK_SIZE = 1500
CHILD_CHUNK_SIZE = 400
```

---

## 📚 Tech Stack

* **LangChain**
* **ChromaDB**
* **HuggingFace Embeddings**
* **Groq (LLM)**
* **Cohere (Reranker)**
* **Streamlit**

---

## ⚠️ Notes

* Ensure API keys are valid before running
* First run may take time due to embedding generation
* Chroma DB is reset on each run

---

## 🔮 Future Improvements

* Add support for multiple PDFs
* Persistent vector DB without reset
* UI improvements with chat history
* Streaming responses
* Add evaluation metrics

---

## 🤝 Contributing

Pull requests are welcome! Feel free to open issues for suggestions or bugs.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 🙌 Acknowledgements

* Cohere for reranking API
* Groq for fast LLM inference
* LangChain for RAG framework
