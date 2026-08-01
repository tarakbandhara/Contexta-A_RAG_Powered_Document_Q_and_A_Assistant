from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from config import EMBEDDING_MODEL, PERSIST_DIR, COLLECTION_NAME, HF_TOKEN

def get_embeddings():
    return HuggingFaceEndpointEmbeddings(model = EMBEDDING_MODEL, huggingfacehub_api_token=HF_TOKEN)

def build_vectorstore(chunks, collection_name = None):
    """
    Create a fresh chroma vectorstore from document chunks.
    Wipes any existing collection first (only the matching one, if a
    session-specific collection_name is given, so this never touches
    another session's data).
    """

    if collection_name is None :
        collection_name = COLLECTION_NAME
    embeddings = get_embeddings()
    try:
        existing = load_vectorstore(collection_name = collection_name)
        existing.delete_collection()
    except Exception:
        pass
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=PERSIST_DIR
    )

def load_vectorstore(collection_name = None):
    """Reconnect to an already-persisted Chroma vectorstore without re-embedding."""
    if collection_name is None :
        collection_name = COLLECTION_NAME
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )