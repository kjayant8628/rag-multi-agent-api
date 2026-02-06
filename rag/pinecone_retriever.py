# rag/pinecone_retriever.py
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from config import Config

# Initialize once (module-level singleton)
pc = None
index = None
model = None

def init_pinecone():
    """Initialize Pinecone connection (call once at startup)"""
    global pc, index, model
    
    if pc is not None:
        return  # Already initialized
    
    print("🔌 Initializing Pinecone...")
    
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    
    # Create index if it doesn't exist
    if Config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"📦 Creating Pinecone index: {Config.PINECONE_INDEX_NAME}")
        pc.create_index(
            name=Config.PINECONE_INDEX_NAME,
            dimension=384,  # all-MiniLM-L6-v2 embedding size
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=Config.PINECONE_ENVIRONMENT
            )
        )
    
    index = pc.Index(Config.PINECONE_INDEX_NAME)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("✅ Pinecone initialized")

def get_context_for_question(question: str, top_k: int = 3) -> str:
    """
    Retrieve context from Pinecone for a given question.
    Compatible with your existing retriever interface.
    """
    if index is None:
        init_pinecone()
    
    # Generate embedding for question
    query_embedding = model.encode(question).tolist()
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    if not results['matches']:
        return "No relevant context found."
    
    # Format context (same format as ChromaDB version)
    contexts = []
    for match in results['matches']:
        text = match['metadata'].get('text', '')
        score = match['score']
        contexts.append(f"[Score: {score:.2f}] {text}")
    
    return "\n\n".join(contexts)