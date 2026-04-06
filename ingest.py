import os
import hashlib
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

class DocumentLoaderFactory:
    """Factory to return the appropriate loader based on file extension."""
    @staticmethod
    def get_loader(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.md' or ext == '.txt':
            return TextLoader(file_path)
        elif ext == '.pdf':
            return PyPDFLoader(file_path)
        # Add more types here (e.g., .docx, .csv) easily!
        else:
            return None

class IngestionManager:
    def __init__(self, data_dir="./data", index_path="faiss_index", state_file="ingest_state.json"):
        self.data_dir = data_dir
        self.index_path = index_path
        self.state_file = state_file
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    def _get_file_hash(self, file_path):
        """Generate a SHA256 hash of the file content."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def process(self):
        new_docs = []
        files_processed = 0

        print(f"Scanning directory: {self.data_dir}...")
        for filename in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, filename)
            if os.path.isdir(file_path):
                continue

            current_hash = self._get_file_hash(file_path)
            
            # Change detection logic
            if self.state.get(filename) == current_hash:
                print(f"Skipping {filename} (no changes detected).")
                continue

            print(f"Processing {filename}...")
            loader = DocumentLoaderFactory.get_loader(file_path)
            if not loader:
                print(f"Unsupported file type: {filename}")
                continue

            # Load and add metadata
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = filename
                doc.metadata["file_type"] = os.path.splitext(filename)[1][1:]
                doc.metadata["content_hash"] = current_hash
            
            new_docs.extend(docs)
            self.state[filename] = current_hash
            files_processed += 1

        if not new_docs:
            print("No new or updated documents to index.")
            return

        print(f"Splitting {len(new_docs)} documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.split_documents(new_docs)

        print(f"Updating Vector Store with {len(chunks)} new chunks...")
        if os.path.exists(self.index_path):
            # Load existing and merge
            vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
            vector_store.add_documents(chunks)
        else:
            # Create new
            vector_store = FAISS.from_documents(chunks, self.embeddings)

        vector_store.save_local(self.index_path)
        self._save_state()
        print(f"Successfully processed {files_processed} files. Index updated.")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_api_key_here":
        print("Error: GOOGLE_API_KEY not found in .env file.")
    else:
        manager = IngestionManager()
        manager.process()
