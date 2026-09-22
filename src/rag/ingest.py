"""
Module 1 - Step 1: Ingest policy PDFs into ChromaDB.

Run this once (or whenever policy PDFs change) to build the vector store
that the RAG chatbot will search at query time.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Config ---
POLICIES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "policies"
CHROMA_DIR = Path(__file__).resolve().parent.parent.parent / "chroma_db"
COLLECTION_NAME = "acko_policies"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_all_pdfs(policies_dir: Path):
    """Load every PDF in the policies folder, tagging each page with its source filename."""
    all_docs = []
    pdf_files = sorted(policies_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {policies_dir}")

    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name} ...")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()  # one Document per page
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
        all_docs.extend(docs)
        print(f"  -> {len(docs)} pages loaded")

    return all_docs


def chunk_documents(documents):
    """Split loaded pages into smaller overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"\nTotal chunks created: {len(chunks)}")
    return chunks


def build_vector_store(chunks):
    """Embed chunks and persist them into a local ChromaDB collection."""
    print(f"\nLoading embedding model: {EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Building ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR} ...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print("Done. Vector store is persisted to disk.")
    return vector_store


def main():
    print("=== Module 1: Policy PDF Ingestion ===\n")
    documents = load_all_pdfs(POLICIES_DIR)
    chunks = chunk_documents(documents)
    build_vector_store(chunks)
    print("\n=== Ingestion complete ===")


if __name__ == "__main__":
    main()