import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "incident_reports"

# Singleton pattern for ChromaDB collection
def get_collection():
    chroma_client = chromadb.Client(Settings(persist_directory=PERSIST_DIR))
    if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
        return chroma_client.get_collection(COLLECTION_NAME)
    return chroma_client.create_collection(COLLECTION_NAME)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def add_chunks_to_vector_store(chunks, source_name):
    collection = get_collection()
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": source_name}]*len(chunks)
    )
    #collection._client.persist()

def retrieve_relevant_chunks(question, top_k=12):
    collection = get_collection()
    question_emb = embedder.encode([question])[0].tolist()
    results = collection.query(query_embeddings=[question_emb], n_results=top_k)
    docs = results['documents'][0] if results['documents'] else []
    return "\n\n".join(docs)
