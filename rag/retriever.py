# import os
# import chromadb
# from sentence_transformers import SentenceTransformer


# DATA_DIR = "data"
# DB_DIR = "vector_db"
# COLLECTION_NAME = "knowledge_base"


# def ingest():
#     """Ingest documents into ChromaDB vector database"""
    
#     # Create directories if they don't exist
#     os.makedirs(DATA_DIR, exist_ok=True)
#     os.makedirs(DB_DIR, exist_ok=True)
    
#     # Use PersistentClient (automatically persists)
#     client = chromadb.PersistentClient(path=DB_DIR)
    
#     # Delete existing collection for fresh start (optional)
#     try:
#         client.delete_collection(name=COLLECTION_NAME)
#         print(f"🗑️ Deleted existing collection")
#     except:
#         pass
    
#     # Create collection
#     collection = client.get_or_create_collection(
#         name=COLLECTION_NAME,
#         metadata={"hnsw:space": "cosine"}
#     )
    
#     print(f"📦 Collection '{COLLECTION_NAME}' created")
    
#     # Load embedding model
#     print("🧠 Loading embedding model...")
#     model = SentenceTransformer("all-MiniLM-L6-v2")
    
#     # Check for text files
#     txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    
#     if not txt_files:
#         print(f"\n⚠️ No .txt files found in '{DATA_DIR}' folder")
#         print(f"Please add your text documents to the '{DATA_DIR}' folder")
#         return
    
#     doc_id = 0
#     total_chunks = 0
    
#     # Process each file
#     for filename in txt_files:
#         path = os.path.join(DATA_DIR, filename)
        
#         print(f"\n📄 Processing: {filename}")
        
#         with open(path, "r", encoding="utf-8") as f:
#             text = f.read()
        
#         # Split into chunks (by double newline)
#         chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        
#         if not chunks:
#             print(f"  ⚠️ No content in {filename}")
#             continue
        
#         # Generate embeddings
#         print(f"  🔢 Generating embeddings for {len(chunks)} chunks...")
#         embeddings = model.encode(chunks, show_progress_bar=False).tolist()
        
#         # Create unique IDs
#         ids = [f"doc_{doc_id + i}" for i in range(len(chunks))]
        
#         # Add to collection
#         collection.add(
#             documents=chunks,
#             embeddings=embeddings,
#             ids=ids,
#             metadatas=[{"source": filename}] * len(chunks)
#         )
        
#         print(f"  ✅ Added {len(chunks)} chunks")
        
#         doc_id += len(chunks)
#         total_chunks += len(chunks)
    
#     # Data is automatically persisted with PersistentClient
#     print(f"\n{'='*60}")
#     print(f"✅ Ingestion complete!")
#     print(f"📊 Total chunks: {total_chunks}")
#     print(f"📁 Total files: {len(txt_files)}")
#     print(f"💾 Database location: {DB_DIR}")
#     print(f"{'='*60}\n")


# def retrieve(query: str, top_k: int = 3):
#     """Retrieve relevant documents from vector database"""
    
#     # Load client
#     client = chromadb.PersistentClient(path=DB_DIR)
    
#     try:
#         collection = client.get_collection(name=COLLECTION_NAME)
#     except:
#         print(f"❌ Collection '{COLLECTION_NAME}' not found!")
#         print(f"Please run ingestion first: python rag/retriever.py")
#         return None
    
#     # Load model
#     model = SentenceTransformer("all-MiniLM-L6-v2")
    
#     # Generate query embedding
#     query_embedding = model.encode([query]).tolist()
    
#     # Search
#     results = collection.query(
#         query_embeddings=query_embedding,
#         n_results=top_k
#     )
    
#     print(f"\n🔍 Query: '{query}'")
#     print(f"{'='*80}\n")
    
#     if results['documents'] and results['documents'][0]:
#         for i, (doc, metadata, distance) in enumerate(zip(
#             results['documents'][0],
#             results['metadatas'][0],
#             results['distances'][0]
#         ), 1):
#             similarity = 1 - distance
#             print(f"📄 Result {i} (Similarity: {similarity:.4f})")
#             print(f"   Source: {metadata.get('source', 'Unknown')}")
#             print(f"   Content: {doc[:150]}...")
#             print(f"{'-'*80}\n")
        
#         return results
#     else:
#         print("❌ No results found\n")
#         return None


# def get_context_for_question(question: str, top_k: int = 3) -> str:
#     """Get formatted context string for a question"""
    
#     results = retrieve(question, top_k)
    
#     if not results or not results['documents'][0]:
#         return "No relevant context found in documents."
    
#     # Format context
#     context_parts = []
#     for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
#         context_parts.append(
#             f"[Source: {metadata.get('source', 'Unknown')}]\n{doc}"
#         )
    
#     return "\n\n---\n\n".join(context_parts)


# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) > 1 and sys.argv[1] == "query":
#         # Query mode
#         if len(sys.argv) > 2:
#             query = " ".join(sys.argv[2:])
#             retrieve(query, top_k=3)
#         else:
#             print("Usage: python rag/retriever.py query <your question>")
#     else:
#         # Ingestion mode (default)
#         ingest()




import os
import chromadb
from sentence_transformers import SentenceTransformer


DATA_DIR = "data"
DB_DIR = "vector_db"
COLLECTION_NAME = "knowledge_base"

# ======================================================
# 🔒 SINGLETON EMBEDDING MODEL (CRITICAL FIX)
# ======================================================
_EMBEDDING_MODEL = None


def get_embedding_model():
    """
    Load SentenceTransformer ONCE per process.
    Prevents torch deadlocks on Windows.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        print("🧠 Loading embedding model (once per process)...")
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


# ======================================================
# INGESTION
# ======================================================
def ingest():
    """Ingest documents into ChromaDB vector database"""

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("🗑️ Deleted existing collection")
    except:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"📦 Collection '{COLLECTION_NAME}' created")

    model = get_embedding_model()

    txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

    if not txt_files:
        print(f"\n⚠️ No .txt files found in '{DATA_DIR}'")
        return

    doc_id = 0
    total_chunks = 0

    for filename in txt_files:
        path = os.path.join(DATA_DIR, filename)
        print(f"\n📄 Processing: {filename}")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

        if not chunks:
            print(f"  ⚠️ No content in {filename}")
            continue

        print(f"  🔢 Generating embeddings for {len(chunks)} chunks...")
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        ids = [f"doc_{doc_id + i}" for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"source": filename}] * len(chunks)
        )

        print(f"  ✅ Added {len(chunks)} chunks")

        doc_id += len(chunks)
        total_chunks += len(chunks)

    print("\n" + "=" * 60)
    print("✅ Ingestion complete!")
    print(f"📊 Total chunks: {total_chunks}")
    print(f"💾 Database location: {DB_DIR}")
    print("=" * 60 + "\n")


# ======================================================
# RETRIEVAL
# ======================================================
def retrieve(query: str, top_k: int = 3):
    """Retrieve relevant documents from vector database"""

    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except:
        print(f"❌ Collection '{COLLECTION_NAME}' not found!")
        return None

    model = get_embedding_model()

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    print(f"\n🔍 Query: '{query}'")
    print("=" * 80 + "\n")

    if results["documents"] and results["documents"][0]:
        for i, (doc, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ),
            1
        ):
            similarity = 1 - distance
            print(f"📄 Result {i} (Similarity: {similarity:.4f})")
            print(f"   Source: {metadata.get('source', 'Unknown')}")
            print(f"   Content: {doc[:150]}...")
            print("-" * 80 + "\n")

        return results

    print("❌ No results found\n")
    return None


# ======================================================
# CONTEXT FORMATTER
# ======================================================
def get_context_for_question(question: str, top_k: int = 3) -> str:
    """Get formatted context string for a question"""

    results = retrieve(question, top_k)

    if not results or not results["documents"][0]:
        return "No relevant context found in documents."

    context_parts = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(
            f"[Source: {metadata.get('source', 'Unknown')}]\n{doc}"
        )

    return "\n\n---\n\n".join(context_parts)


# ======================================================
# CLI ENTRYPOINT
# ======================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "query":
        if len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            retrieve(query, top_k=3)
        else:
            print("Usage: python rag/retriever.py query <your question>")
    else:
        ingest()
