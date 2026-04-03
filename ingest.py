import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env
load_dotenv()

def ingest_docs():
    """
    Load documents from the data directory, split them into chunks,
    generate embeddings, and save the index to a local FAISS store.
    """
    
    print("Loading documents...")
    # Load all markdown files from the data directory
    loader = DirectoryLoader('./data', glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents into chunks...")
    # Split documents into manageable chunks
    # Chunk size and overlap are critical for RAG performance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Small enough for precise retrieval
        chunk_overlap=50  # Overlap to maintain context between chunks
    )
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks.")

    print("Generating embeddings and creating FAISS index...")
    # Initialize Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create the vector store from chunks
    vector_store = FAISS.from_documents(texts, embeddings)
    
    print("Saving FAISS index locally...")
    # Save the index for later use in the main application
    vector_store.save_local("faiss_index")
    print("Ingestion complete! Index saved to 'faiss_index'.")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_api_key_here":
        print("Error: GOOGLE_API_KEY not found in environment or .env file.")
    else:
        ingest_docs()
