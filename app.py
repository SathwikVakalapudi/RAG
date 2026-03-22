import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq  # Corrected import
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

# Load environment variables
load_dotenv()

# Debug: Verify API key
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

def initialize_vectorstore():
    """Initialize the vectorstore and retriever."""
    st.write("Step 1: Resetting Chroma directory")
    reset_chroma_directory(PERSIST_DIR)

    st.write("Step 2: Extracting text from PDF")
    doc_list, error_msg = extract_text_pdfplumber(PDF_PATH)
    if error_msg:
        st.error(error_msg)
        return None, None, None

    st.write("Step 3: Converting to LangChain Documents")
    documents = convert_to_langchain_documents(doc_list, PDF_PATH)
    if not documents:
        st.error("No documents converted")
        return None, None, None

    st.write("Step 4: Initializing embeddings and vectorstore")
    embeddings = init_embeddings(EMBED_MODEL)
    vectorstore = init_vectorstore(COLLECTION_NAME, embeddings, PERSIST_DIR)

    st.write("Step 5: Initializing retriever")
    parent_splitter, child_splitter = init_splitters(PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE)
    retriever, store = init_retriever(vectorstore, parent_splitter, child_splitter)

    st.write("Step 6: Adding documents to retriever")
    retriever.add_documents(documents)
    st.success("Documents added successfully!")
    
    return vectorstore, retriever, store

def generate_answer(query, reranked_docs):
    """Generate an answer using ChatGroq with the query and reranked documents."""
    try:
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Format context from reranked documents
        context = "\n\n".join([doc["full_content"] for doc in reranked_docs])
        prompt = f"""You are a helpful assistant. Based on the following context, provide a concise and accurate answer to the query. If the context doesn't fully address the query, note any limitations and answer to the best of your ability.

Query: {query}

Context:
{context}

Answer:"""

        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        st.error(f"Error generating answer with LLM: {str(e)}")
        return None

def process_query(query, vectorstore, retriever, store):
    """Process the query, retrieve documents, rerank, and generate an answer."""
    st.write("Rewriting query with Groq...")
    rewrites = rewrite_query_groq_with_intents(query, num_variations=5)
    st.write("Rewritten queries:", rewrites)

    st.write("Retrieving chunks for rewritten queries...")
    retrieved_docs = []
    for q in rewrites:
        chunks = vectorstore.similarity_search(q, k=5)
        for chunk in chunks:
            parent_docs = store.mget([chunk.metadata.get("doc_id")])
            retrieved_docs.append({
                "query": q,
                "content": chunk.page_content,
                "full_content": chunk.page_content,
                "metadata": chunk.metadata,
                "parent_content": parent_docs[0].page_content if parent_docs and parent_docs[0] else None
            })

    st.write(f"Total retrieved docs before deduplication: {len(retrieved_docs)}")

    # Deduplication
    seen = set()
    deduped_docs = []
    for doc in retrieved_docs:
        key = (doc["metadata"].get("doc_id"), doc["metadata"].get("page_number"), doc["content"])
        if key not in seen:
            deduped_docs.append(doc)
            seen.add(key)

    st.write(f"After deduplication: {len(deduped_docs)} docs")

    st.write("Reranking results with Cohere...")
    reranked_docs = rerank_results(query, deduped_docs, top_n=3)
    
    # Generate answer with LLM
    st.write("Generating answer with LLM...")
    answer = generate_answer(query, reranked_docs)
    
    return reranked_docs, answer

def main():
    """Main Streamlit app."""
    st.title("PDF Query Search")
    st.write("Enter a query to search the PDF document and get an answer.")

    # Initialize vectorstore and retriever
    if "vectorstore" not in st.session_state:
        with st.spinner("Initializing vectorstore..."):
            vectorstore, retriever, store = initialize_vectorstore()
            if vectorstore is None:
                st.stop()
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.session_state.store = store

    # Query input
    query = st.text_input("Enter your query:", value="What is climate change?")
    
    if st.button("Search"):
        if not query:
            st.warning("Please enter a query.")
            return
        
        with st.spinner("Processing query..."):
            reranked_docs, answer = process_query(
                query, 
                st.session_state.vectorstore, 
                st.session_state.retriever, 
                st.session_state.store
            )
            
            # Display results
            st.subheader("Generated Answer")
            if answer:
                st.markdown(answer)
            else:
                st.warning("No answer generated due to an error.")

            st.subheader("Top Reranked Results")
            for i, doc in enumerate(reranked_docs, 1):
                st.markdown(f"**Rank {i}:**")
                st.write(f"**Page:** {doc['metadata'].get('page_number')}")
                st.write(f"**Content:** {doc['full_content'][:200]}...")
                st.write("---")

if __name__ == "__main__":
    main()