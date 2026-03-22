from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

def init_splitters(parent_size, child_size):
    
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )

    def tiktoken_len(text):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(text))

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size, chunk_overlap=50,
        length_function=tiktoken_len, separators=["\n", ".", "?", "!", " "]
    )
    return parent_splitter, child_splitter

def init_retriever(vectorstore, parent_splitter, child_splitter):
    store = InMemoryStore()
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
    ), store
