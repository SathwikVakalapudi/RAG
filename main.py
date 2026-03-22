# =========================
# Main entry point
# =========================

from config import (
    PDF_PATH, PERSIST_DIR, COLLECTION_NAME,
    EMBED_MODEL, PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE
)

from pdf_utils import extract_text_pdfplumber
from doc_utils import convert_to_langchain_documents
from vectorstore_utils import reset_chroma_directory, init_embeddings, init_vectorstore
from retriever_utils import init_splitters, init_retriever

from llm_utils.groq_rewriter import rewrite_query_groq_with_intents
from llm_utils.cohere_reranker import rerank_results

# =========================
# Step 1: Reset Chroma directory
# =========================
print("Step 1: Resetting Chroma directory")
reset_chroma_directory(PERSIST_DIR)

# =========================
# Step 2: Extract text from PDF
# =========================
print("\nStep 2: Extracting text from PDF")
doc_list, error_msg = extract_text_pdfplumber(PDF_PATH)
if error_msg:
    print(error_msg)
    exit(1)

# =========================
# Step 3: Convert to LangChain Documents
# =========================
print("\nStep 3: Converting to LangChain Documents")
documents = convert_to_langchain_documents(doc_list, PDF_PATH)
if not documents:
    print("No documents converted, exiting")
    exit(1)

# =========================
# Step 4: Initialize embeddings + vectorstore
# =========================
print("\nStep 4: Initializing embeddings and vectorstore")
embeddings = init_embeddings(EMBED_MODEL)
vectorstore = init_vectorstore(COLLECTION_NAME, embeddings, PERSIST_DIR)

# =========================
# Step 5: Initialize splitters and retriever
# =========================
print("\nStep 5: Initializing retriever")
parent_splitter, child_splitter = init_splitters(PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE)
retriever, store = init_retriever(vectorstore, parent_splitter, child_splitter)

# =========================
# Step 6: Add documents
# =========================
print("\nStep 6: Adding documents to retriever")
retriever.add_documents(documents)
print("Documents added successfully!")

# =========================
# Step 7: Example query pipeline
# =========================
query = "What is climate change?"

print("\nRewriting query with Groq...")
rewrites = rewrite_query_groq_with_intents(query, num_variations=5)
print("Rewritten queries:", rewrites)

print("\nRetrieving chunks for rewritten queries...")
retrieved_docs = []
for q in rewrites:
    chunks = vectorstore.similarity_search(q, k=5)
    for chunk in chunks:
        parent_docs = store.mget([chunk.metadata.get("doc_id")])
        retrieved_docs.append({
            "query": q,
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "parent_content": parent_docs[0].page_content if parent_docs and parent_docs[0] else None
        })

print(f"Total retrieved docs before deduplication: {len(retrieved_docs)}")

# Deduplication
seen = set()
deduped_docs = []
for doc in retrieved_docs:
    key = (doc["metadata"].get("doc_id"), doc["metadata"].get("page_number"), doc["content"])
    if key not in seen:
        deduped_docs.append(doc)
        seen.add(key)

print(f"After deduplication: {len(deduped_docs)} docs")

# =========================
# Step 8: Reranking with Cohere
# =========================
print("\nReranking results with Cohere...")
reranked_docs = rerank_results(query, deduped_docs, top_n=3)

print("\nTop reranked results:")
for i, doc in enumerate(reranked_docs, 1):
    print(f"\nRank {i}:")
    print("Page:", doc["metadata"].get("page_number"))
    print("Content:", doc["full_content"][:200], "...")
