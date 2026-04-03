import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

def run_app():
    """
    Main application loop for the library guide.
    """
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_api_key_here":
        print("Please set your GOOGLE_API_KEY in the .env file before running the app.")
        return

    # 1. Load the FAISS index
    # We must provide the embeddings model used during ingestion to load the index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Check if index exists
    if not os.path.exists("faiss_index"):
        print("Vector index not found. Please run 'python ingest.py' first.")
        return
        
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 2. Initialize the LLM (Gemini 1.5 Flash is fast and cheap)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # 3. Define the friendly system prompt
    # In interviews, explaining "system prompts" and "few-shot" is key.
    system_prompt = (
        "You are a friendly and helpful library guide. "
        "Your goal is to assist users with their questions about the library's "
        "database structure, policies, and documents. "
        "Answer in a polite, professional, yet warm tone. "
        "Use the following context to provide your answer. "
        "If you don't know the answer from the context, politely say so. "
        "\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 4. Create the Retrieval Chain
    # Modern LangChain uses create_stuff_documents_chain and create_retrieval_chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    print("--- Welcome to the Library Guide AI! ---")
    print("Ask me anything about library policies or database structure.")
    print("(Type 'exit' to quit)")

    while True:
        user_input = input("\nYour Question: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! Have a great day!")
            break
        
        print("\nSearching and thinking...")
        try:
            response = retrieval_chain.invoke({"input": user_input})
            print(f"\nFriendly Librarian: {response['answer']}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_app()
