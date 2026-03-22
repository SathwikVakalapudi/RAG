import cohere
from dotenv import load_dotenv
import os
load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))

def rerank_results(query: str, retrieved_docs: list, top_n: int = 3):
    if not retrieved_docs: return []
    inputs = [doc["full_content"] for doc in retrieved_docs]
    rerank_response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=inputs,
        top_n=top_n
    )
    return [retrieved_docs[r.index] for r in rerank_response.results]
