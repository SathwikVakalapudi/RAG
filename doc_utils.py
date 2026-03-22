from langchain_core.documents import Document

def convert_to_langchain_documents(doc_list, source_path):
    documents = []
    for doc in doc_list:
        documents.append(
            Document(
                page_content=doc["text"],
                metadata={"page_number": doc["page"], "source": source_path},
            )
        )
    return documents
