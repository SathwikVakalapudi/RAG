# vectorstore_utils.py

import os
import shutil
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def reset_chroma_directory(persist_dir):
    if os.path.exists(persist_dir):
        try:
            shutil.rmtree(persist_dir)
        except PermissionError:
            new_name = persist_dir + "_old"
            os.rename(persist_dir, new_name)
    os.makedirs(persist_dir, exist_ok=True)
    os.chmod(persist_dir, 0o777)


def init_embeddings(model_name: str):
    """Initialize HuggingFace embeddings model"""
    return HuggingFaceEmbeddings(model_name=model_name)


def init_vectorstore(collection_name, embeddings, persist_dir):
    """Initialize Chroma vectorstore"""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
