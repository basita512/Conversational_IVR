# -*- coding: utf-8 -*-
import json
from sentence_transformers import SentenceTransformer
import chromadb

def extract_text_from_json(obj, parent_key=""):
    """
    Recursively extract readable text chunks from nested JSON.
    Combines keys and string values into meaningful text sentences.
    """
    chunks = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{parent_key} > {key}" if parent_key else key
            chunks.extend(extract_text_from_json(value, new_key))

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            chunks.extend(extract_text_from_json(item, parent_key))

    elif isinstance(obj, str):
        # Final text node — build a sentence
        text = f"{parent_key}: {obj}"
        chunks.append(text)

    return chunks


def main():
    print("🚀 Initializing model and ChromaDB...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./zenius_db")
    collection = client.get_or_create_collection("zenius_kb")

    print("🧩 Extracting text chunks from JSON...")
    with open("zenius.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    text_chunks = extract_text_from_json(data)
    print(f"✅ Extracted {len(text_chunks)} chunks from KB")

    added_count = 0
    for i, chunk in enumerate(text_chunks):
        try:
            embedding = model.encode(chunk)
            collection.add(
                documents=[chunk],
                embeddings=[embedding.tolist()],
                metadatas=[{"source": "zenius.json"}],
                ids=[str(i)]
            )
            added_count += 1
        except Exception as e:
            print(f"⚠️ Skipping chunk {i} due to error: {e}")

    print(f"✅ Successfully added {added_count} entries to ChromaDB (zenius_db)")


if __name__ == "__main__":
    main()
