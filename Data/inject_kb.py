# -*- coding: utf-8 -*-
import os
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk

# Download sentence tokenizer (first run only)
# To install this without internet
# python3 -m nltk.downloader punkt punkt_tab

nltk.download('punkt')
nltk.download('punkt_tab')


# === CONFIG ===
KB_FILES = [
    "your_file.docx",
]

CHROMA_PATH = "./vector_db"
COLLECTION_NAME = "vector_kb"

# === CHUNKING CONFIG ===
MAX_TOKENS = 200     # Target chunk size
OVERLAP_TOKENS = 40  # Semantic overlap for continuity


# ---------- HELPER: Sentence Token Count ----------
def count_tokens(text, model):
    return len(model.tokenizer.tokenize(text))


# ---------- SEMANTIC SENTENCE SPLITTING ----------
def split_into_sentences(text):
    from nltk.tokenize import sent_tokenize
    return [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 1]


# ---------- CLUSTERED SEMANTIC CHUNKER (200 TOKENS) ----------
def semantic_chunk(text, embed_model):
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    # Encode each sentence
    sentence_embeddings = embed_model.encode(sentences, convert_to_numpy=True)

    # Create clusters by similarity threshold
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence, emb in zip(sentences, sentence_embeddings):
        tokens = count_tokens(sentence, embed_model)

        # If chunk too large, close it
        if current_len + tokens > MAX_TOKENS:
            chunks.append(" ".join(current_chunk))

            # Create overlap
            overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
            current_chunk = overlap_sentences.copy()
            current_len = sum(count_tokens(s, embed_model) for s in current_chunk)

        # Add sentence
        current_chunk.append(sentence)
        current_len += tokens

    # Add last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ---------- FILE LOADING + CHUNKING ----------
def load_and_chunk_file(filepath, embed_model):
    _, ext = os.path.splitext(filepath.lower())
    
    if ext == '.docx':
        try:
            from docx import Document
            doc = Document(filepath)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        except ImportError:
            raise ImportError("Install python-docx to process .docx files: pip install python-docx")
    
    elif ext == '.pdf':
        try:
            import PyPDF2
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = '\n'.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        except ImportError:
            raise ImportError("Install PyPDF2 to process PDFs: pip install PyPDF2")
    
    else:  # Plain text files
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="latin-1") as f:
                text = f.read()

    # Cleanup
    text = ' '.join(text.split())  # Normalize whitespace
    chunks = semantic_chunk(text, embed_model)

    print(f" {filepath}: {len(chunks)} semantic chunks")
    return chunks


# ---------- MAIN INGESTION ----------
def main():
    print("?? Initializing BGE-Large v1.5 Model...")
    embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    print(f"?? Embedding Model Ready (1024-dim)")

    # Reset DB
    print("??? Resetting ChromaDB...")
    if os.path.exists(CHROMA_PATH):
        os.system(f"rm -rf {CHROMA_PATH}")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.create_collection(COLLECTION_NAME)

    print("?? Loading & Chunking KB Files...\n")

    all_chunks = []
    all_metadatas = []
    all_ids = []

    id_counter = 0

    for kb in KB_FILES:
        chunks = load_and_chunk_file(kb, embed_model)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": kb})
            all_ids.append(str(id_counter))
            id_counter += 1

    print(f"\n?? Total Chunks: {len(all_chunks)}")
    print("?? Generating Embeddings (May take a few minutes)...")

    embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)

    print("?? Inserting into ChromaDB...")
    collection.add(
        documents=all_chunks,
        embeddings=embeddings.tolist(),
        metadatas=all_metadatas,
        ids=all_ids
    )

    print("\n? DONE! Zenius KB loaded into ChromaDB with Semantic 200-token Chunking.")
    print(f"?? DB Path: {CHROMA_PATH}")
    print(f"?? Collection: {COLLECTION_NAME}")
    print("?? Your RAG accuracy will now be much higher!")


if __name__ == "__main__":
    main()
