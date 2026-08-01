import re
import os
from langchain_community.document_loaders import PyPDFium2Loader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from config import chunk_size, chunk_overlap

LOADER_MAP = {".pdf":PyPDFium2Loader, ".docx":Docx2txtLoader, ".txt":lambda path: TextLoader(path, encoding="utf-8", autodetect_encoding=True), ".csv":CSVLoader}

def load_file(file_path):
    """Load a single file into LangChain Document objects, tagged with source filename."""
    file_extention = os.path.splitext(file_path)[1].lower()
    loader_cls = LOADER_MAP.get(file_extention)
    if loader_cls is None:
        raise ValueError(f"{file_extention} is not supported")
    docs = loader_cls(file_path).load()
    filename = os.path.basename(file_path)
    for doc in docs:
        doc.metadata["source"]=filename
    return docs

# function so we dont have to stick to fixed chunking and overlapping and threshold value : 

def get_adaptive_chunk_params(documents):
    """
    Analyze document text to pick a reasonable chunk_size and chunk_overlap.
    Defaults to sentence-based splitting (best for prose). Falls back to
    line-based splitting only when very few sentences are detected, which
    signals a structured, non-prose document (resumes, forms, slides) where
    sentence-ending punctuation is unreliable.
    Returns (chunk_size, chunk_overlap).
    """
    sentence_fragments = []
    line_fragments = []

    for doc in documents:
        # Sentence-based: good for prose-heavy documents
        # collapse wrapped-line newlines into spaces so they don't fragment sentences
        normalized_text = doc.page_content .replace("\n", " ")
        # split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+\s+', normalized_text)
        sentence_fragments.extend(sentences)
        # Line-based: good for structured documents (resumes, slides, forms)
        # that don't reliably use sentence-ending punctuation
        lines = doc.page_content.split("\n")
        line_fragments.extend(lines)
    cleaned_sentences = [s.strip() for s in sentence_fragments if s.strip()]
    cleaned_lines = [l.strip() for l in line_fragments if l.strip()]

    MIN_Sentence_Count = 20

    if len(cleaned_sentences) < MIN_Sentence_Count and cleaned_lines:
        chosen_fragments = cleaned_lines
    else :
        chosen_fragments = cleaned_sentences
    if not chosen_fragments:
        return 800, 100
    avg_sentence_length = sum(len(s) for s in chosen_fragments) / len(chosen_fragments)
    fragments_per_chunk = 5
    raw_chunk_size = avg_sentence_length * fragments_per_chunk
    chunk_size_result = int(min(max(raw_chunk_size, 200), 1500))
    chunk_overlap_result = int(chunk_size_result * 0.13)
    return chunk_size_result, chunk_overlap_result


def chunk_documents(documents, chunk_size = None, chunk_overlap = None):
    """
    Split loaded documents into overlapping chunks for embedding.
     If chunk_size/chunk_overlap aren't provided, computes adaptive values
     based on the document's own structure instead of using fixed constants.
    """
    if chunk_size is None or chunk_overlap is None:
        chunk_size, chunk_overlap = get_adaptive_chunk_params(documents)

    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    return splitter.split_documents(documents)