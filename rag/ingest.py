import os
import chromadb
from sentence_transformers import SentenceTransformer


DATA_DIR = "data"
DB_DIR = "vector_db"
COLLECTION_NAME = "knowledge_base"


def ingest():
    # Create directory if it doesn't exist
    os.makedirs(DB_DIR, exist_ok=True)
    
    # Use PersistentClient for newer ChromaDB versions
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Delete existing collection if it exists (optional - for fresh start)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"🗑️ Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass
    
    # Create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    print(f"📦 Collection '{COLLECTION_NAME}' ready")
    
    # Load embedding model
    print("🧠 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check if data directory has files
    txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    
    if not txt_files:
        print(f"⚠️ No .txt files found in '{DATA_DIR}' folder")
        print(f"Please add your text documents to the '{DATA_DIR}' folder")
        return
    
    doc_id = 0
    total_chunks = 0
    
    for filename in txt_files:
        path = os.path.join(DATA_DIR, filename)
        
        print(f"📄 Processing: {filename}")
        
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Split into chunks
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        
        if not chunks:
            print(f"  ⚠️ No content found in {filename}")
            continue
        
        # Generate embeddings
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()
        
        # Create IDs
        ids = [f"doc_{doc_id + i}" for i in range(len(chunks))]
        
        # Add to collection
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"source": filename}] * len(chunks)
        )
        
        print(f"  ✅ Added {len(chunks)} chunks from {filename}")
        
        doc_id += len(chunks)
        total_chunks += len(chunks)
    
    print(f"\n✅ Ingestion complete!")
    print(f"📊 Total chunks ingested: {total_chunks}")
    print(f"💾 Database saved to: {DB_DIR}")
    
    # Verify the database was created
    if os.path.exists(DB_DIR):
        print(f"✅ Vector database folder created successfully")
        db_size = sum(
            os.path.getsize(os.path.join(DB_DIR, f)) 
            for f in os.listdir(DB_DIR) 
            if os.path.isfile(os.path.join(DB_DIR, f))
        )
        print(f"📦 Database size: {db_size / 1024:.2f} KB")
    else:
        print(f"❌ ERROR: Database folder not created!")


if __name__ == "__main__":
    ingest()