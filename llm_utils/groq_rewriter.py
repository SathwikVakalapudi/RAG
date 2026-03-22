from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

def rewrite_query_groq_with_intents(query, num_variations=5, intents=None):
    if intents is None:
        intents = ['informational', 'comparative', 'procedural', 'causal', 'applicative']

    prompt = f"""
    You are an expert question rewriter.
    Rewrite the following question in {num_variations} diverse and natural ways, 
    each with a different intent: {', '.join(intents)}.

    Question: "{query}"

    Return only the rewritten questions as a numbered list.
    """
    response = llm.invoke(prompt)
    return [line.strip() for line in response.content.split("\n") if line.strip()]
